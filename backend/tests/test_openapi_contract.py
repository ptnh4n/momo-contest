"""
tests/test_openapi_contract.py

Kiểm tra hợp đồng OpenAPI theo tutorial.md mục 4 & 5:
  [x] File parse được (JSON hợp lệ)
  [x] Có đúng 6 thao tác và đúng path/method
  [x] Mọi $ref đều phân giải được, không ref mồ côi
  [x] Các trường tiền là integer (amount, new_balance, balance, daily_transferred, auto_approve_limit)
  [x] Enum Decision/SourceType/DraftStatus/TrustLevel khớp CHECK trong schema.sql
  [x] DraftView có suggested_alias
  [x] ChatRequest có intent_id, user_id, message trong required
  [x] openapi-spec-validator nếu có (bỏ qua nếu không)
"""

import json
import re
import sys
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Đường dẫn
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).parent.parent
_OPENAPI_PATH = _BACKEND_DIR / "openapi" / "safepay.openapi.json"
_SCHEMA_SQL_PATH = _BACKEND_DIR / "app" / "db" / "schema.sql"


# ---------------------------------------------------------------------------
# Fixture: load spec một lần
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def spec():
    assert _OPENAPI_PATH.exists(), f"Không tìm thấy: {_OPENAPI_PATH}"
    with open(_OPENAPI_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helper: resolve $ref nội bộ
# ---------------------------------------------------------------------------
def _resolve_ref(spec: dict, ref: str) -> dict:
    """Phân giải '#/components/schemas/Foo' → dict schema."""
    assert ref.startswith("#/"), f"Chỉ hỗ trợ internal $ref, got: {ref}"
    parts = ref.lstrip("#/").split("/")
    node = spec
    for p in parts:
        assert p in node, f"$ref không phân giải được: {ref} (thiếu key '{p}')"
        node = node[p]
    return node


def _collect_refs(obj, found=None):
    """Thu thập tất cả $ref trong spec (đệ quy)."""
    if found is None:
        found = set()
    if isinstance(obj, dict):
        if "$ref" in obj:
            found.add(obj["$ref"])
        for v in obj.values():
            _collect_refs(v, found)
    elif isinstance(obj, list):
        for item in obj:
            _collect_refs(item, found)
    return found


def _collect_defined_schema_names(spec: dict) -> set:
    return set(spec.get("components", {}).get("schemas", {}).keys())


def _collect_defined_param_names(spec: dict) -> set:
    return set(spec.get("components", {}).get("parameters", {}).keys())


def _collect_defined_response_names(spec: dict) -> set:
    return set(spec.get("components", {}).get("responses", {}).keys())


# ---------------------------------------------------------------------------
# 1. File parse được
# ---------------------------------------------------------------------------
class TestFileParseable:
    def test_file_exists(self):
        assert _OPENAPI_PATH.exists(), f"Thiếu file: {_OPENAPI_PATH}"

    def test_valid_json(self, spec):
        assert isinstance(spec, dict), "Root phải là JSON object"

    def test_openapi_version(self, spec):
        assert spec.get("openapi", "").startswith("3."), "Phải là OpenAPI 3.x"

    def test_info_title(self, spec):
        assert "title" in spec.get("info", {}), "Thiếu info.title"

    def test_paths_present(self, spec):
        assert "paths" in spec and len(spec["paths"]) > 0, "Thiếu paths"


# ---------------------------------------------------------------------------
# 2. Đúng 6 thao tác (tutorial mục 4 & 5)
# ---------------------------------------------------------------------------
EXPECTED_OPERATIONS = [
    ("post", "/api/chat"),
    ("post", "/api/drafts/{tx_id}/confirm"),
    ("post", "/api/drafts/{tx_id}/cancel"),
    ("get",  "/api/wallet/state"),
    ("post", "/api/demo/reset"),
    ("get",  "/health"),
]


class TestOperations:
    def test_has_exactly_6_operations(self, spec):
        paths = spec["paths"]
        ops = [
            (method, path)
            for path, path_item in paths.items()
            for method in path_item
            if method in {"get", "post", "put", "patch", "delete", "head", "options"}
        ]
        assert len(ops) == 6, f"Phải có đúng 6 thao tác, tìm thấy {len(ops)}: {ops}"

    @pytest.mark.parametrize("method,path", EXPECTED_OPERATIONS)
    def test_operation_exists(self, spec, method, path):
        paths = spec["paths"]
        assert path in paths, f"Thiếu path '{path}'"
        assert method in paths[path], f"Thiếu method '{method}' trên '{path}'"

    @pytest.mark.parametrize("method,path", EXPECTED_OPERATIONS)
    def test_operation_has_operationId(self, spec, method, path):
        op = spec["paths"][path][method]
        assert "operationId" in op, f"Thiếu operationId trên {method.upper()} {path}"

    @pytest.mark.parametrize("method,path", EXPECTED_OPERATIONS)
    def test_operation_has_responses(self, spec, method, path):
        op = spec["paths"][path][method]
        assert "responses" in op, f"Thiếu responses trên {method.upper()} {path}"
        assert "200" in op["responses"], f"Thiếu 200 response trên {method.upper()} {path}"


# ---------------------------------------------------------------------------
# 3. $ref đều phân giải được, không có schema mồ côi
# ---------------------------------------------------------------------------
class TestRefs:
    def test_all_refs_resolvable(self, spec):
        """Mọi $ref trong spec đều phân giải được."""
        all_refs = _collect_refs(spec)
        errors = []
        for ref in sorted(all_refs):
            try:
                _resolve_ref(spec, ref)
            except AssertionError as e:
                errors.append(str(e))
        assert not errors, "Có $ref không phân giải được:\n" + "\n".join(errors)

    def test_no_orphan_schemas(self, spec):
        """Mọi schema trong components/schemas đều được tham chiếu ít nhất 1 lần."""
        defined = _collect_defined_schema_names(spec)
        all_refs = _collect_refs(spec)
        # Lấy tên schema từ ref '#/components/schemas/Foo'
        referenced = {
            ref.split("/")[-1]
            for ref in all_refs
            if ref.startswith("#/components/schemas/")
        }
        orphans = defined - referenced
        # ContactView, TransactionView, ClarifyChoice có thể không được ref trực tiếp từ paths
        # nhưng đều được ref từ schemas khác — chỉ cảnh báo nếu không được ref ở BẤT KỲ đâu
        assert not orphans, f"Schema mồ côi (không có $ref nào trỏ tới): {orphans}"

    def test_all_component_params_resolvable(self, spec):
        param_refs = {
            r for r in _collect_refs(spec)
            if r.startswith("#/components/parameters/")
        }
        defined = _collect_defined_param_names(spec)
        for ref in param_refs:
            name = ref.split("/")[-1]
            assert name in defined, f"Parameter $ref không tồn tại: {ref}"

    def test_all_component_responses_resolvable(self, spec):
        response_refs = {
            r for r in _collect_refs(spec)
            if r.startswith("#/components/responses/")
        }
        defined = _collect_defined_response_names(spec)
        for ref in response_refs:
            name = ref.split("/")[-1]
            assert name in defined, f"Response $ref không tồn tại: {ref}"


# ---------------------------------------------------------------------------
# 4. Các trường tiền là integer (tutorial mục 4)
# ---------------------------------------------------------------------------
# Ánh xạ: (schema_name, field_name) -> type phải là integer (hoặc ["integer", "null"])
MONEY_FIELDS = [
    ("DraftView",        "amount"),
    ("TransactionView",  "amount"),
    ("ClarifyChoice",    "amount"),   # ["integer", "null"]
    ("ConfirmResponse",  "new_balance"),
    ("ChatResponse",     "new_balance"),  # ["integer", "null"]
    ("WalletState",      "balance"),
    ("WalletState",      "daily_transferred"),
    ("WalletState",      "auto_approve_limit"),
]


class TestMoneyFields:
    @pytest.mark.parametrize("schema_name,field", MONEY_FIELDS)
    def test_money_field_is_integer(self, spec, schema_name, field):
        schemas = spec["components"]["schemas"]
        assert schema_name in schemas, f"Thiếu schema '{schema_name}'"
        props = schemas[schema_name].get("properties", {})
        assert field in props, f"Thiếu trường '{field}' trong {schema_name}"
        field_type = props[field].get("type")
        # Chấp nhận "integer" hoặc ["integer", "null"]
        if isinstance(field_type, list):
            assert "integer" in field_type, (
                f"{schema_name}.{field}: type phải chứa 'integer', got {field_type}"
            )
        else:
            assert field_type == "integer", (
                f"{schema_name}.{field}: type phải là 'integer', got {field_type!r}"
            )


# ---------------------------------------------------------------------------
# 5. Enum khớp CHECK trong schema.sql (tutorial mục 5)
# ---------------------------------------------------------------------------
def _extract_sql_enum(sql: str, check_pattern: str) -> set:
    """Trích enum từ CHECK constraint trong SQL, VD: CHECK (trust_level IN ('NEW','TRUSTED'))"""
    match = re.search(check_pattern, sql)
    assert match, f"Không tìm thấy pattern '{check_pattern}' trong schema.sql"
    values_str = match.group(1)
    return {v.strip("'") for v in re.findall(r"'[^']++'?", values_str) if v.strip("'")}


@pytest.fixture(scope="module")
def schema_sql():
    assert _SCHEMA_SQL_PATH.exists(), f"Không tìm thấy: {_SCHEMA_SQL_PATH}"
    return _SCHEMA_SQL_PATH.read_text(encoding="utf-8")


class TestEnumConsistency:
    def _get_openapi_enum(self, spec, schema_name):
        schemas = spec["components"]["schemas"]
        assert schema_name in schemas, f"Thiếu schema '{schema_name}'"
        enum_vals = schemas[schema_name].get("enum")
        assert enum_vals is not None, f"Schema '{schema_name}' không có 'enum'"
        return set(enum_vals)

    def test_decision_enum_matches_sql(self, spec, schema_sql):
        openapi_vals = self._get_openapi_enum(spec, "Decision")
        sql_vals = _extract_sql_enum(
            schema_sql,
            r"decision\s+IN\s*\(([^)]+)\)"
        )
        assert openapi_vals == sql_vals, (
            f"Decision enum mâu thuẫn!\nOpenAPI: {openapi_vals}\nSQL:    {sql_vals}"
        )

    def test_source_type_enum_matches_sql(self, spec, schema_sql):
        openapi_vals = self._get_openapi_enum(spec, "SourceType")
        sql_vals = _extract_sql_enum(
            schema_sql,
            r"source_type\s+IN\s*\(([^)]+)\)"
        )
        assert openapi_vals == sql_vals, (
            f"SourceType enum mâu thuẫn!\nOpenAPI: {openapi_vals}\nSQL:    {sql_vals}"
        )

    def test_draft_status_enum_matches_sql(self, spec, schema_sql):
        openapi_vals = self._get_openapi_enum(spec, "DraftStatus")
        sql_vals = _extract_sql_enum(
            schema_sql,
            r"status\s+IN\s*\(([^)]+)\)"
        )
        assert openapi_vals == sql_vals, (
            f"DraftStatus enum mâu thuẫn!\nOpenAPI: {openapi_vals}\nSQL:    {sql_vals}"
        )

    def test_trust_level_enum_matches_sql(self, spec, schema_sql):
        openapi_vals = self._get_openapi_enum(spec, "TrustLevel")
        sql_vals = _extract_sql_enum(
            schema_sql,
            r"trust_level\s+IN\s*\(([^)]+)\)"
        )
        assert openapi_vals == sql_vals, (
            f"TrustLevel enum mâu thuẫn!\nOpenAPI: {openapi_vals}\nSQL:    {sql_vals}"
        )


# ---------------------------------------------------------------------------
# 6. DraftView có suggested_alias; ChatRequest có required fields
# ---------------------------------------------------------------------------
class TestKeyFields:
    def test_draft_view_has_suggested_alias(self, spec):
        """DraftView phải có trường suggested_alias (tutorial mục 0 & 5)."""
        schemas = spec["components"]["schemas"]
        assert "DraftView" in schemas
        props = schemas["DraftView"].get("properties", {})
        assert "suggested_alias" in props, "DraftView thiếu trường 'suggested_alias'"
        # Phải nullable (["string", "null"])
        sa_type = props["suggested_alias"].get("type")
        assert sa_type == ["string", "null"] or "null" in (sa_type if isinstance(sa_type, list) else []), (
            f"suggested_alias phải nullable, got type={sa_type}"
        )

    def test_chat_request_required_fields(self, spec):
        """ChatRequest phải có user_id, intent_id, message trong required."""
        schemas = spec["components"]["schemas"]
        assert "ChatRequest" in schemas
        required = schemas["ChatRequest"].get("required", [])
        for field in ("user_id", "intent_id", "message"):
            assert field in required, f"ChatRequest thiếu '{field}' trong required"

    def test_chat_request_intent_id_uuid_format(self, spec):
        """intent_id phải có format: uuid."""
        schemas = spec["components"]["schemas"]
        props = schemas["ChatRequest"]["properties"]
        assert "intent_id" in props
        assert props["intent_id"].get("format") == "uuid", "intent_id phải có format: uuid"

    def test_draft_view_required_fields(self, spec):
        """DraftView phải có các required field theo tutorial."""
        schemas = spec["components"]["schemas"]
        required = schemas["DraftView"].get("required", [])
        for field in ("tx_id", "recipient_name", "recipient_account", "bank_code", "amount", "source_type", "status"):
            assert field in required, f"DraftView thiếu '{field}' trong required"

    def test_confirm_response_required_fields(self, spec):
        schemas = spec["components"]["schemas"]
        required = schemas["ConfirmResponse"].get("required", [])
        for field in ("status", "new_balance", "tx_id"):
            assert field in required, f"ConfirmResponse thiếu '{field}' trong required"

    def test_error_response_required_fields(self, spec):
        schemas = spec["components"]["schemas"]
        required = schemas["ErrorResponse"].get("required", [])
        for field in ("error_code", "message"):
            assert field in required, f"ErrorResponse thiếu '{field}' trong required"

    def test_wallet_state_required_fields(self, spec):
        schemas = spec["components"]["schemas"]
        required = schemas["WalletState"].get("required", [])
        for field in ("balance", "daily_transferred", "auto_approve_limit", "contacts", "recent_transactions"):
            assert field in required, f"WalletState thiếu '{field}' trong required"


# ---------------------------------------------------------------------------
# 7. openapi-spec-validator (nếu có, không tự cài)
# ---------------------------------------------------------------------------
class TestSpecValidator:
    def test_spec_validator_if_available(self, spec):
        """Chạy openapi-spec-validator nếu cài sẵn, bỏ qua nếu không có."""
        try:
            from openapi_spec_validator import validate
            validate(spec)
        except ImportError:
            pytest.skip("openapi-spec-validator không cài sẵn — bỏ qua")
        except Exception as e:
            pytest.fail(f"openapi-spec-validator báo lỗi: {e}")
