"""
tests/test_t05_orchestrator.py

Kiểm thử toàn diện T05 (LLM Orchestrator & Privilege Separation):
1. Privilege Separation: đúng 6 tool, chặn tool ngoài whitelist, HITL không tự trừ tiền.
2. Zero-PII PrivacyGuard + OutboundAuditLog (E07): mask cả STK/SĐT tách dấu cách/chấm,
   không mask nhầm tiền phân cách hàng nghìn (500.000, 10.000.000), fail-closed nếu lộ PII.
3. Luồng chuyển khoản theo biệt danh ('thằng Nam') -> lookup_contact -> create_draft_transfer -> PIN confirm.
4. Luồng OCR hóa đơn thật bằng RapidOCR (Pillow PNG) + chia bill 4 người -> REQUIRE_HITL,
   source_type='OCR_RECEIPT', suggested_alias=None, '4 người' không bị chặn nhầm thành số tiền mơ hồ.
5. Chống Prompt Injection & Memory Poisoning từ ảnh OCR (chặn upsert_contact_alias, ép OCR_RECEIPT,
   strip suggested_alias, blacklist hard veto).
6. Chặn số tiền mơ hồ ('500') trước khi gọi LLM -> CLARIFY CURRENCY_UNIT -> clarify_choice tạo Draft.
7. Chặn LLM tự đoán người nhận khi lookup_contact trả >= 2 ('Minh') -> CLARIFY CONTACT -> clarify_choice; alias chỉ ghi sau PIN.
8. get_recent_transactions trả về lịch sử Zero-PII.
9. upsert_contact_alias (Phương án B): danh bạ có sẵn lưu ngay (NO_SILENT_OVERWRITE),
   STK thô mới -> CLARIFY xác nhận 2 bước -> 'Đồng ý' mới tạo contact NEW + alias; 'Hủy' không ghi DB; blacklist bị chặn.
10. suggested_alias Post-PIN commit qua chat STK thô trực tiếp.
11. Idempotency: gửi lại cùng intent_id -> trả draft cũ với 0 lần gọi LLM.
12. PlannerUnavailableError (429/503) -> degrade an toàn về CLARIFY, không mất tiền.
"""

from __future__ import annotations

import base64
import io
import os
import re
import sys
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from fastapi.testclient import TestClient
from google.genai import errors as genai_errors
from PIL import Image, ImageDraw
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.connection import get_conn, init_db
from app.db.seed import DEMO_PIN, DEMO_USER_ID, seed_demo
from app.orchestrator.config import OrchestratorConfig
from app.orchestrator.graph import SafePayOrchestrator
from app.orchestrator.llm_client import (
    BasePlannerClient,
    GeminiPlannerClient,
    PlannerTurnResult,
    PlannerUnavailableError,
    ToolCall,
)
from app.orchestrator.ocr import ImageStagingStore, OcrReceiptService
from app.orchestrator.pending_alias import PendingAliasStore
from app.orchestrator.privacy_guard import OutboundAuditLog, PiiLeakBlockedError, PrivacyGuard
from app.orchestrator.tools import (
    ALLOWED_LLM_TOOLS,
    ToolExecutionContext,
    ToolRegistry,
    build_gemini_tool_declarations,
)
from app.wallet.api import create_app
from app.wallet.privacy_proxy import PrivacyProxy
from app.wallet.service import ClarifyContextStore


class ScriptedPlannerClient(BasePlannerClient):
    """Planner giả lập chạy qua cùng PrivacyGuard + OutboundAuditLog như Gemini thật."""

    def __init__(
        self,
        turns: list[PlannerTurnResult | Exception],
        privacy_guard: PrivacyGuard,
        audit_log: OutboundAuditLog,
    ) -> None:
        self._turns = list(turns)
        self.privacy_guard = privacy_guard
        self.audit_log = audit_log
        self.calls: list[list[dict[str, Any]]] = []

    def plan_step(
        self,
        *,
        session_id: str,
        history: list[dict[str, Any]],
        tool_declarations: Any,
        extra_forbidden_pii: set[str] | None = None,
    ) -> PlannerTurnResult:
        preview = self.privacy_guard.assert_zero_pii_outbound(
            session_id,
            history,
            extra_forbidden=extra_forbidden_pii,
        )
        self.audit_log.record(session_id, "scripted-gemini-4-hydrogen", preview)
        self.calls.append(list(history))

        if not self._turns:
            return PlannerTurnResult(text="Đã hoàn tất xử lý.")
        nxt = self._turns.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _make_receipt_png_base64(lines: list[str]) -> str:
    img = Image.new("RGB", (680, 60 + 48 * len(lines)), "white")
    draw = ImageDraw.Draw(img)
    y = 24
    for line in lines:
        draw.text((24, y), line, fill="black")
        y += 48
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


@pytest.fixture
def db_file(tmp_path):
    path = str(tmp_path / "t05_test.db")
    init_db(path)
    conn = get_conn(path)
    seed_demo(conn, DEMO_USER_ID)
    conn.close()
    return path


@pytest.fixture
def db(db_file):
    conn = get_conn(db_file)
    yield conn
    conn.close()


def _build_test_harness(db_file: str, turns: list[PlannerTurnResult | Exception]):
    proxy = PrivacyProxy()
    guard = PrivacyGuard(proxy)
    clarify_store = ClarifyContextStore()
    pending_alias_store = PendingAliasStore()
    audit_log = OutboundAuditLog()
    planner = ScriptedPlannerClient(turns, privacy_guard=guard, audit_log=audit_log)
    orch = SafePayOrchestrator(
        config=OrchestratorConfig(mode="llm", api_keys=("fake-key-for-unit-test",)),
        privacy_guard=guard,
        clarify_store=clarify_store,
        pending_alias_store=pending_alias_store,
        ocr_service=OcrReceiptService(privacy_guard=guard, image_store=ImageStagingStore()),
        planner_client=planner,
        audit_log=audit_log,
    )
    app = create_app(db_path=db_file, seed_if_empty=False, orchestrator=orch)
    return app, orch, planner, audit_log, guard


class TestPrivilegeSeparationAndPrivacy:
    def test_exactly_6_declared_tools_and_no_execution_tools(self):
        tool_obj = build_gemini_tool_declarations()
        declared_names = {fd.name for fd in tool_obj.function_declarations}
        assert declared_names == ALLOWED_LLM_TOOLS
        assert len(declared_names) == 6
        for forbidden in ("confirm_draft", "deduct_balance", "execute_transfer", "reset_demo"):
            assert forbidden not in declared_names

    def test_forbidden_tool_rejected_by_registry(self, db):
        guard = PrivacyGuard(PrivacyProxy())
        ctx = ToolExecutionContext(
            conn=db,
            user_id=DEMO_USER_ID,
            intent_id=str(uuid4()),
            session_id="s1",
            masked_user_message="chuyển 500k",
            has_image=False,
            image_id=None,
            privacy_guard=guard,
            clarify_store=ClarifyContextStore(),
            pending_alias_store=PendingAliasStore(),
            ocr_service=OcrReceiptService(guard),
        )
        res = ToolRegistry().execute(ctx, "confirm_draft", {"tx_id": "tx_1", "pin_code": "123456"})
        assert res["error"] == "FORBIDDEN_TOOL"

    def test_llm_lookup_returns_only_excel_whitelist(self, db):
        guard = PrivacyGuard(PrivacyProxy())
        ctx = ToolExecutionContext(
            conn=db, user_id=DEMO_USER_ID, intent_id=str(uuid4()),
            session_id="lookup-test", masked_user_message="Chuyển 500k cho thằng Nam",
            has_image=False, image_id=None, privacy_guard=guard,
            clarify_store=ClarifyContextStore(), pending_alias_store=PendingAliasStore(),
            ocr_service=OcrReceiptService(guard),
        )
        result = ToolRegistry().execute(ctx, "lookup_contact", {"query": "thằng Nam"})
        assert result == {"matches": [{
            "contact_id": "C_001", "alias": "thằng Nam", "masked_acct": "****567",
        }]}
        assert ctx.last_lookup_matches[0]["bank_code"] == "MB"

    def test_privacy_guard_masks_separated_pii_and_preserves_formatted_money(self):
        guard = PrivacyGuard(PrivacyProxy())
        raw = "Chuyển 10.000.000 cho anh Tuấn STK 1903 1234 5678 90 SĐT 0987.654.321"
        res = guard.mask_text("sess_sep", raw)

        assert "10.000.000" in res.masked_text
        assert "1903" not in res.masked_text
        assert "5678" not in res.masked_text
        assert "0987" not in res.masked_text
        assert "<ACCT_1>" in res.masked_text
        assert "<PHONE_1>" in res.masked_text
        assert guard.resolve_token("sess_sep", "<ACCT_1>") == "19031234567890"
        assert guard.resolve_token("sess_sep", "<PHONE_1>") == "0987654321"

        # Fail-closed nếu cố tình đưa STK thô vào outbound payload
        with pytest.raises(PiiLeakBlockedError):
            guard.assert_zero_pii_outbound(
                "sess_sep",
                {"text": "Chuyển tới 19031234567890"},
            )
        with pytest.raises(PiiLeakBlockedError):
            guard.assert_zero_pii_outbound(
                "sess_sep",
                {"text": "Liên hệ 0987 654 321"},
            )


class TestEndToEndOrchestratorFlows:
    def test_alias_transfer_and_pin_confirm_zero_pii_outbound(self, db_file, db):
        turns = [
            PlannerTurnResult(tool_calls=[ToolCall("lookup_contact", {"query": "thằng Nam"})]),
            PlannerTurnResult(
                tool_calls=[
                    ToolCall(
                        "create_draft_transfer",
                        {
                            "intent_id": "ignored-by-backend",
                            "recipient_ref": "C_001",
                            "bank_code": "MB",
                            "amount": 500_000,
                            "note": "Tiền ăn trưa",
                            "source_type": "CHAT_CONTACT",
                        },
                    )
                ]
            ),
        ]
        app, _orch, planner, audit_log, _guard = _build_test_harness(db_file, turns)

        with TestClient(app) as client:
            res = client.post(
                "/api/chat",
                json={
                    "user_id": DEMO_USER_ID,
                    "intent_id": str(uuid4()),
                    "message": "Chuyển 500k cho thằng Nam tiền ăn trưa",
                },
            )
            assert res.status_code == 200
            body = res.json()
            assert body["decision"] == "REQUIRE_HITL"
            assert body["draft"]["recipient_name"] == "Nguyễn Văn Nam"
            assert body["draft"]["recipient_account"] == "9001234567"
            assert body["draft"]["amount"] == 500_000
            assert body["draft"]["status"] == "DRAFT"

            # Privilege separation: chưa nhập PIN thì chưa trừ tiền
            bal_before = db.execute(
                "SELECT balance FROM users WHERE user_id = ?", (DEMO_USER_ID,)
            ).fetchone()[0]
            assert bal_before == 10_000_000

            # Kiểm chứng 0% PII trong OutboundAuditLog gửi cho LLM
            entries = audit_log.entries()
            assert len(entries) == 2
            joined_outbound = "\n".join(e.payload_preview for e in entries)
            assert "9001234567" not in joined_outbound
            assert "Nguyễn Văn Nam" not in joined_outbound
            assert "****567" in joined_outbound

            # Xác nhận PIN -> trừ tiền thật
            tx_id = body["draft"]["tx_id"]
            conf = client.post(
                f"/api/drafts/{tx_id}/confirm",
                json={"user_id": DEMO_USER_ID, "pin_code": DEMO_PIN},
            )
            assert conf.status_code == 200
            assert conf.json()["new_balance"] == 9_500_000

    def test_real_rapidocr_bill_split_forces_hitl_and_zero_pii(self, db_file, db):
        img_b64 = _make_receipt_png_base64(
            [
                "HOA DON AN TOI",
                "Tong cong: 400.000 VND",
                "STK: 8899776655 MB Bank",
            ]
        )
        # 400.000 / 4 = 100.000 (nằm trong auto_approve_limit=100k của C_001 TRUSTED),
        # nhưng nguồn OCR bắt buộc REQUIRE_HITL (OCR_ALWAYS_HITL)!
        turns = [
            PlannerTurnResult(tool_calls=[ToolCall("parse_receipt_image", {"image_id": "img_auto"})]),
            PlannerTurnResult(tool_calls=[ToolCall("lookup_contact", {"query": "thằng Nam"})]),
            PlannerTurnResult(
                tool_calls=[
                    ToolCall(
                        "create_draft_transfer",
                        {
                            "intent_id": "anything",
                            "recipient_ref": "C_001",
                            "bank_code": "MB",
                            "amount": 100_000,
                            "note": "Chia bill an toi 4 nguoi",
                            "source_type": "CHAT_CONTACT",  # cố tình truyền sai để test backend ép OCR_RECEIPT
                            "suggested_alias": "alias doc hai",  # cố tình truyền để test NO_OCR_ALIAS
                        },
                    )
                ]
            ),
        ]
        app, _orch, planner, audit_log, _guard = _build_test_harness(db_file, turns)

        with TestClient(app) as client:
            res = client.post(
                "/api/chat",
                json={
                    "user_id": DEMO_USER_ID,
                    "intent_id": str(uuid4()),
                    "message": "chia đều bill này cho 4 người rồi chuyển cho thằng Nam",
                    "image_base64": img_b64,
                },
            )
            assert res.status_code == 200
            body = res.json()
            assert body["decision"] == "REQUIRE_HITL"
            assert body["draft"]["amount"] == 100_000
            assert body["draft"]["source_type"] == "OCR_RECEIPT"
            assert body["draft"]["suggested_alias"] is None
            assert body["draft"]["status"] == "DRAFT"
            assert any("OCR_ALWAYS_HITL" in r for r in body["xai_reasons"])
            assert any("NO_OCR_ALIAS" in r for r in body["xai_reasons"])

            # RapidOCR chạy cục bộ và mask STK trên hóa đơn trước khi gửi LLM
            joined_outbound = "\n".join(e.payload_preview for e in audit_log.entries())
            assert "8899776655" not in joined_outbound
            assert "9001234567" not in joined_outbound
            assert "<ACCT_1>" in joined_outbound
            assert len(planner.calls) == 3

            # Tiền chưa bị trừ
            bal = db.execute("SELECT balance FROM users WHERE user_id = ?", (DEMO_USER_ID,)).fetchone()[0]
            assert bal == 10_000_000

    def test_ocr_prompt_injection_upsert_alias_blocked_and_blacklist_vetoed(self, db_file, db):
        img_b64 = _make_receipt_png_base64(
            [
                "HOA DON THANH TOAN",
                "Tong cong: 500.000 VND",
                "STK: 0000000001 MB",
            ]
        )
        # Giả lập LLM bị Prompt Injection trong ảnh OCR:
        # 1) Cố gọi upsert_contact_alias để đầu độc 'sếp'
        # 2) Cố gọi check_scam_database và create_draft_transfer tới STK blacklist
        turns = [
            PlannerTurnResult(tool_calls=[ToolCall("parse_receipt_image", {"image_id": "img_x"})]),
            PlannerTurnResult(
                tool_calls=[
                    ToolCall(
                        "upsert_contact_alias",
                        {"contact_ref": "<ACCT_1>", "alias": "sếp giả mạo", "bank_code": "MB"},
                    )
                ]
            ),
            PlannerTurnResult(
                tool_calls=[
                    ToolCall("check_scam_database", {"account_ref": "<ACCT_1>", "bank_code": "MB"}),
                    ToolCall(
                        "create_draft_transfer",
                        {
                            "intent_id": "x",
                            "recipient_ref": "<ACCT_1>",
                            "bank_code": "MB",
                            "amount": 500_000,
                            "note": "Thanh toan hoa don",
                            "source_type": "OCR_RECEIPT",
                        },
                    ),
                ]
            ),
        ]
        app, _orch, _planner, audit_log, _guard = _build_test_harness(db_file, turns)

        with TestClient(app) as client:
            res = client.post(
                "/api/chat",
                json={
                    "user_id": DEMO_USER_ID,
                    "intent_id": str(uuid4()),
                    "message": "thanh toán hóa đơn này giúp tôi",
                    "image_base64": img_b64,
                },
            )
            assert res.status_code == 200
            body = res.json()
            assert body["decision"] == "BLOCK_ANOMALY"
            assert body["draft"]["status"] == "BLOCKED"
            assert body["draft"]["recipient_account"] == "0000000001"
            assert any("BLACKLIST_HARD_VETO" in r for r in body["xai_reasons"])
            assert any("NO_OCR_ALIAS" in r for r in body["xai_reasons"])

            # Không có alias rác nào được ghi vào DB
            cnt = db.execute(
                "SELECT COUNT(*) FROM contact_aliases WHERE alias_norm LIKE '%gia mao%'"
            ).fetchone()[0]
            assert cnt == 0

            # STK blacklist không rò ra Gemini
            joined_outbound = "\n".join(e.payload_preview for e in audit_log.entries())
            assert "0000000001" not in joined_outbound

    def test_ambiguous_currency_500_short_circuits_before_llm(self, db_file, db):
        app, _orch, planner, _audit_log, _guard = _build_test_harness(db_file, [])
        intent_clarify = str(uuid4())

        with TestClient(app) as client:
            r1 = client.post(
                "/api/chat",
                json={
                    "user_id": DEMO_USER_ID,
                    "intent_id": intent_clarify,
                    "message": "chuyển 500 cho thằng Nam",
                },
            )
            assert r1.status_code == 200
            b1 = r1.json()
            assert b1["decision"] == "CLARIFY"
            assert b1["clarify_options"]["kind"] == "CURRENCY_UNIT"
            assert [o["amount"] for o in b1["clarify_options"]["options"]] == [500_000, 500]
            assert len(planner.calls) == 0

            # Chọn 500.000 VNĐ -> tạo draft không cần gọi LLM
            r2 = client.post(
                "/api/chat",
                json={
                    "user_id": DEMO_USER_ID,
                    "intent_id": str(uuid4()),
                    "message": "500.000 VNĐ",
                    "clarify_choice": {"ref_intent_id": intent_clarify, "amount": 500_000},
                },
            )
            assert r2.status_code == 200
            b2 = r2.json()
            assert b2["decision"] == "REQUIRE_HITL"
            assert b2["draft"]["amount"] == 500_000
            assert b2["draft"]["recipient_account"] == "9001234567"
            assert len(planner.calls) == 0

    def test_ambiguous_contact_defers_alias_expansion_to_pin_confirm(self, db_file, db):
        turns = [
            PlannerTurnResult(tool_calls=[ToolCall("lookup_contact", {"query": "Minh"})]),
            # Giả lập LLM cố tình đoán C_003 dù lookup trả 2 người trùng tên Minh
            PlannerTurnResult(
                tool_calls=[
                    ToolCall(
                        "create_draft_transfer",
                        {
                            "intent_id": "x",
                            "recipient_ref": "C_003",
                            "bank_code": "TCB",
                            "amount": 300_000,
                            "source_type": "CHAT_CONTACT",
                        },
                    )
                ]
            ),
        ]
        app, _orch, _planner, _audit_log, _guard = _build_test_harness(db_file, turns)
        intent_clarify = str(uuid4())

        with TestClient(app) as client:
            r1 = client.post(
                "/api/chat",
                json={
                    "user_id": DEMO_USER_ID,
                    "intent_id": intent_clarify,
                    "message": "chuyển 300k cho anh Minh",
                },
            )
            assert r1.status_code == 200
            b1 = r1.json()
            assert b1["decision"] == "CLARIFY"
            assert b1["draft"] is None
            assert b1["clarify_options"]["kind"] == "CONTACT"
            assert len(b1["clarify_options"]["options"]) == 2
            # Nhãn CLARIFY chỉ hiện masked_acct, không hiện STK đầy đủ
            for opt in b1["clarify_options"]["options"]:
                assert "9001112222" not in opt["label"]
                assert "9003334444" not in opt["label"]

            aliases_before = db.execute(
                "SELECT contact_id, alias_norm, usage_count FROM contact_aliases ORDER BY contact_id, alias_norm"
            ).fetchall()
            r2 = client.post(
                "/api/chat",
                json={
                    "user_id": DEMO_USER_ID,
                    "intent_id": str(uuid4()),
                    "message": "Chọn Lê Quang Minh",
                    "clarify_choice": {"ref_intent_id": intent_clarify, "contact_id": "C_003"},
                },
            )
            assert r2.status_code == 200
            b2 = r2.json()
            assert b2["decision"] == "REQUIRE_HITL"
            assert b2["draft"]["recipient_account"] == "9001112222"
            assert b2["draft"]["amount"] == 300_000
            suggested = b2["draft"]["suggested_alias"]
            assert suggested is not None
            assert db.execute(
                "SELECT contact_id, alias_norm, usage_count FROM contact_aliases ORDER BY contact_id, alias_norm"
            ).fetchall() == aliases_before

            confirmed = client.post(
                f"/api/drafts/{b2['draft']['tx_id']}/confirm",
                json={"user_id": DEMO_USER_ID, "pin_code": DEMO_PIN, "save_alias": True},
            )
            assert confirmed.status_code == 200
            assert confirmed.json()["alias_saved"] == suggested

    def test_get_recent_transactions_zero_pii(self, db_file):
        turns = [
            PlannerTurnResult(tool_calls=[ToolCall("get_recent_transactions", {"limit": 5})]),
            PlannerTurnResult(text="Bạn vừa chuyển 200.000 VNĐ cho thằng Nam (MB ****567)."),
        ]
        app, _orch, planner, audit_log, _guard = _build_test_harness(db_file, turns)

        with TestClient(app) as client:
            res = client.post(
                "/api/chat",
                json={
                    "user_id": DEMO_USER_ID,
                    "intent_id": str(uuid4()),
                    "message": "Xem lịch sử giao dịch gần nhất",
                },
            )
            assert res.status_code == 200
            assert res.json()["decision"] == "PASS_AUTO"
            assert res.json()["draft"] is None
            joined = "\n".join(e.payload_preview for e in audit_log.entries())
            assert "9001234567" not in joined
            assert "****567" in joined
            assert len(planner.calls) == 2

    def test_ambiguous_contact_polite_llm_still_emits_clarify_options(self, db_file):
        """Nếu LLM tuân lời prompt và hỏi lại (không gọi create_draft_transfer), vẫn phải trả ClarifyOptions."""
        turns = [
            PlannerTurnResult(tool_calls=[ToolCall("lookup_contact", {"query": "Minh"})]),
            PlannerTurnResult(text="Có 2 người tên Minh trong danh bạ, bạn muốn chuyển cho ai?"),
        ]
        app, _orch, _planner, _audit_log, _guard = _build_test_harness(db_file, turns)
        intent_clarify = str(uuid4())

        with TestClient(app) as client:
            r1 = client.post(
                "/api/chat",
                json={
                    "user_id": DEMO_USER_ID,
                    "intent_id": intent_clarify,
                    "message": "chuyển 300k cho Minh",
                },
            )
            assert r1.status_code == 200
            b1 = r1.json()
            assert b1["decision"] == "CLARIFY"
            assert b1["clarify_options"]["kind"] == "CONTACT"
            assert len(b1["clarify_options"]["options"]) == 2

            r2 = client.post(
                "/api/chat",
                json={
                    "user_id": DEMO_USER_ID,
                    "intent_id": str(uuid4()),
                    "message": "Chọn Phạm Thanh Minh",
                    "clarify_choice": {"ref_intent_id": intent_clarify, "contact_id": "C_004"},
                },
            )
            assert r2.status_code == 200
            b2 = r2.json()
            assert b2["decision"] == "REQUIRE_HITL"
            assert b2["draft"]["recipient_account"] == "9003334444"
            assert b2["draft"]["amount"] == 300_000

    def test_upsert_contact_alias_option_b_new_and_existing(self, db_file, db):
        # 1) STK mới -> Turn 1 CLARIFY -> Turn 2 "Đồng ý" mới tạo contact NEW + alias
        turns = [
            PlannerTurnResult(
                tool_calls=[
                    ToolCall(
                        "upsert_contact_alias",
                        {"contact_ref": "<ACCT_1>", "alias": "anh Tuấn", "bank_code": "VCB"},
                    )
                ]
            ),
            # Lượt test hủy ('Hủy') cho STK mới thứ 2
            PlannerTurnResult(
                tool_calls=[
                    ToolCall(
                        "upsert_contact_alias",
                        {"contact_ref": "<ACCT_1>", "alias": "bạn Tùng", "bank_code": "TCB"},
                    )
                ]
            ),
            # Lượt test contact đã có (C_001) gắn thêm alias 'sếp' -> NO_SILENT_OVERWRITE
            PlannerTurnResult(tool_calls=[ToolCall("lookup_contact", {"query": "thằng Nam"})]),
            PlannerTurnResult(
                tool_calls=[
                    ToolCall(
                        "upsert_contact_alias",
                        {"contact_ref": "C_001", "alias": "sếp", "bank_code": "MB"},
                    )
                ]
            ),
            PlannerTurnResult(text="Đã thêm biệt danh 'sếp' cho thằng Nam (vẫn giữ danh bạ sếp cũ)."),
        ]
        app, _orch, _planner, audit_log, _guard = _build_test_harness(db_file, turns)

        with TestClient(app) as client:
            # Turn 1: lưu STK mới -> hỏi xác nhận CLARIFY, chưa ghi DB
            r1 = client.post(
                "/api/chat",
                json={
                    "user_id": DEMO_USER_ID,
                    "intent_id": str(uuid4()),
                    "message": "Lưu STK 19038888999900 VCB là anh Tuấn nhé",
                },
            )
            assert r1.status_code == 200
            assert r1.json()["decision"] == "CLARIFY"
            assert "****900" in r1.json()["reply_text"]
            cnt_before = db.execute(
                "SELECT COUNT(*) FROM contacts WHERE account_number = '19038888999900'"
            ).fetchone()[0]
            assert cnt_before == 0

            # Turn 2: Đồng ý -> ghi contacts (trust_level='NEW') + contact_aliases
            r2 = client.post(
                "/api/chat",
                json={
                    "user_id": DEMO_USER_ID,
                    "intent_id": str(uuid4()),
                    "message": "Đồng ý",
                },
            )
            assert r2.status_code == 200
            assert r2.json()["decision"] == "PASS_AUTO"
            c_row = db.execute(
                "SELECT contact_id, trust_level, bank_code FROM contacts WHERE account_number = '19038888999900'"
            ).fetchone()
            assert c_row is not None
            assert c_row["trust_level"] == "NEW"
            assert c_row["bank_code"] == "VCB"
            a_row = db.execute(
                "SELECT alias_raw FROM contact_aliases WHERE contact_id = ?", (c_row["contact_id"],)
            ).fetchone()
            assert a_row["alias_raw"] == "anh Tuấn"

            # Cancel flow -> 0 rows added
            r3 = client.post(
                "/api/chat",
                json={
                    "user_id": DEMO_USER_ID,
                    "intent_id": str(uuid4()),
                    "message": "Lưu STK 556677889900 TCB là bạn Tùng",
                },
            )
            assert r3.json()["decision"] == "CLARIFY"
            r4 = client.post(
                "/api/chat",
                json={
                    "user_id": DEMO_USER_ID,
                    "intent_id": str(uuid4()),
                    "message": "Hủy",
                },
            )
            assert r4.json()["decision"] == "CLARIFY"
            assert (
                db.execute(
                    "SELECT COUNT(*) FROM contacts WHERE account_number = '556677889900'"
                ).fetchone()[0]
                == 0
            )

            # Existing contact C_001 + NO_SILENT_OVERWRITE với alias 'sếp'
            r5 = client.post(
                "/api/chat",
                json={
                    "user_id": DEMO_USER_ID,
                    "intent_id": str(uuid4()),
                    "message": "Gắn thêm biệt danh sếp cho thằng Nam",
                },
            )
            assert r5.status_code == 200
            assert r5.json()["decision"] == "PASS_AUTO"
            sep_cnt = db.execute(
                "SELECT COUNT(*) FROM contact_aliases WHERE alias_norm = 'sep'"
            ).fetchone()[0]
            assert sep_cnt == 2  # giữ nguyên cả C_002 lẫn C_001

            # Kiểm tra 0% PII thô trong audit log
            joined = "\n".join(e.payload_preview for e in audit_log.entries())
            assert "19038888999900" not in joined
            assert "556677889900" not in joined

    def test_post_pin_suggested_alias_from_raw_chat_transfer(self, db_file, db):
        turns = [
            PlannerTurnResult(
                tool_calls=[
                    ToolCall(
                        "create_draft_transfer",
                        {
                            "intent_id": "x",
                            "recipient_ref": "<ACCT_1>",
                            "bank_code": "TCB",
                            "amount": 350_000,
                            "note": "Trả tiền cafe",
                            "source_type": "CHAT_RAW_ACCT",
                            "suggested_alias": "anh Long",
                        },
                    )
                ]
            )
        ]
        app, _orch, _planner, _audit_log, _guard = _build_test_harness(db_file, turns)

        with TestClient(app) as client:
            res = client.post(
                "/api/chat",
                json={
                    "user_id": DEMO_USER_ID,
                    "intent_id": str(uuid4()),
                    "message": "Chuyển 350k cho anh Long STK 19037777666655 TCB",
                },
            )
            assert res.status_code == 200
            body = res.json()
            assert body["decision"] == "REQUIRE_HITL"
            assert body["draft"]["suggested_alias"] == "anh Long"
            assert body["draft"]["recipient_account"] == "19037777666655"

            # Chưa PIN -> chưa lưu vào contact_aliases
            assert (
                db.execute(
                    "SELECT COUNT(*) FROM contact_aliases WHERE alias_norm = 'anh long'"
                ).fetchone()[0]
                == 0
            )

            # Confirm PIN + save_alias=True -> commit vào contact_aliases
            tx_id = body["draft"]["tx_id"]
            conf = client.post(
                f"/api/drafts/{tx_id}/confirm",
                json={"user_id": DEMO_USER_ID, "pin_code": DEMO_PIN, "save_alias": True},
            )
            assert conf.status_code == 200
            assert conf.json()["alias_saved"] == "anh Long"
            assert (
                db.execute(
                    "SELECT COUNT(*) FROM contact_aliases WHERE alias_norm = 'anh long'"
                ).fetchone()[0]
                == 1
            )

    def test_idempotency_skips_llm_on_duplicate_intent_id(self, db_file):
        turns = [
            PlannerTurnResult(tool_calls=[ToolCall("lookup_contact", {"query": "thằng Nam"})]),
            PlannerTurnResult(
                tool_calls=[
                    ToolCall(
                        "create_draft_transfer",
                        {
                            "intent_id": "x",
                            "recipient_ref": "C_001",
                            "bank_code": "MB",
                            "amount": 50_000,
                            "source_type": "CHAT_CONTACT",
                        },
                    )
                ]
            ),
        ]
        app, _orch, planner, _audit_log, _guard = _build_test_harness(db_file, turns)
        intent_id = str(uuid4())
        payload = {
            "user_id": DEMO_USER_ID,
            "intent_id": intent_id,
            "message": "Chuyển 50k cho thằng Nam",
        }

        with TestClient(app) as client:
            r1 = client.post("/api/chat", json=payload)
            assert r1.status_code == 200
            assert r1.json()["decision"] == "PASS_AUTO"
            assert r1.json()["new_balance"] == 9_950_000
            calls_after_first = len(planner.calls)

            r2 = client.post("/api/chat", json=payload)
            assert r2.status_code == 200
            assert r2.json()["draft"]["tx_id"] == r1.json()["draft"]["tx_id"]
            assert r2.json()["new_balance"] == 9_950_000
            assert len(planner.calls) == calls_after_first

    def test_planner_unavailable_degrades_safely_without_moving_money(self, db_file, db):
        turns: list[PlannerTurnResult | Exception] = [
            PlannerUnavailableError("503 UNAVAILABLE")
        ]
        app, _orch, _planner, _audit_log, _guard = _build_test_harness(db_file, turns)

        with TestClient(app) as client:
            res = client.post(
                "/api/chat",
                json={
                    "user_id": DEMO_USER_ID,
                    "intent_id": str(uuid4()),
                    "message": "Chuyển 200k cho thằng Nam",
                },
            )
            assert res.status_code == 200
            body = res.json()
            assert body["decision"] == "CLARIFY"
            assert body["draft"] is None
            assert any("LLM_PLANNER_UNAVAILABLE" in r for r in body["xai_reasons"])

            bal = db.execute("SELECT balance FROM users WHERE user_id = ?", (DEMO_USER_ID,)).fetchone()[0]
            assert bal == 10_000_000


class TestFallbackAndKeyRotation:
    def test_privacy_guard_allows_internal_img_and_tx_ids_with_8_digits(self):
        guard = PrivacyGuard(PrivacyProxy())
        clean = guard.assert_zero_pii_outbound(
            "sess_ids",
            {
                "text": '[IMAGE_ATTACHED: image_id="img_1234567890ab"] ref tx_9876543210fe',
                "uuid": "12345678-1234-4abc-9def-123456789012",
            },
        )
        assert "img_1234567890ab" in clean
        with pytest.raises(PiiLeakBlockedError):
            guard.assert_zero_pii_outbound("sess_ids", {"text": "Chuyển tới 9001234567"})

    def test_gemini_client_falls_back_to_35_flash_lite_on_503(self):
        guard = PrivacyGuard(PrivacyProxy())
        audit = OutboundAuditLog()
        cfg = OrchestratorConfig(
            mode="llm",
            api_keys=("key-1", "key-2"),
            model="gemini-3.6-flash",
            fallback_models=("gemini-3.5-flash-lite",),
            thinking_level=None,
        )
        client = GeminiPlannerClient(cfg, guard, audit)
        called_models: list[str] = []

        def _fake_generate(model: str, contents: Any, config: Any):
            called_models.append(model)
            if model == "gemini-3.6-flash":
                raise genai_errors.ServerError(503, {"error": {"code": 503, "message": "high demand"}})
            fc = SimpleNamespace(name="lookup_contact", args={"query": "thằng Nam"})
            cand = SimpleNamespace(content=SimpleNamespace(role="model", parts=[]))
            return SimpleNamespace(function_calls=[fc], candidates=[cand], text="")

        client._clients = [
            SimpleNamespace(models=SimpleNamespace(generate_content=_fake_generate)),
            SimpleNamespace(models=SimpleNamespace(generate_content=_fake_generate)),
        ]

        res = client.plan_step(
            session_id="s_fb",
            history=[{"role": "user", "text": "Chuyển 250k cho thằng Nam"}],
            tool_declarations=build_gemini_tool_declarations(),
        )
        assert called_models == ["gemini-3.6-flash", "gemini-3.5-flash-lite"]
        assert res.tool_calls == [ToolCall("lookup_contact", {"query": "thằng Nam"})]
        assert audit.entries()[-1].model == "gemini-3.5-flash-lite"

    def test_gemini_client_rotates_api_keys_on_429(self):
        guard = PrivacyGuard(PrivacyProxy())
        audit = OutboundAuditLog()
        cfg = OrchestratorConfig(
            mode="llm",
            api_keys=("key-1", "key-2", "key-3"),
            model="gemini-3.6-flash",
            fallback_models=("gemini-3.5-flash-lite",),
            thinking_level=None,
        )
        client = GeminiPlannerClient(cfg, guard, audit)
        key_hits: list[str] = []

        def _make_fake_client(tag: str, fail_429: bool):
            def _gen(model: str, contents: Any, config: Any):
                key_hits.append(f"{tag}:{model}")
                if fail_429:
                    raise genai_errors.ClientError(429, {"error": {"code": 429, "message": "quota exceeded"}})
                fc = SimpleNamespace(name="lookup_contact", args={"query": "sếp"})
                cand = SimpleNamespace(content=SimpleNamespace(role="model", parts=[]))
                return SimpleNamespace(function_calls=[fc], candidates=[cand], text="")

            return SimpleNamespace(models=SimpleNamespace(generate_content=_gen))

        client._clients = [
            _make_fake_client("k1", True),
            _make_fake_client("k2", False),
            _make_fake_client("k3", False),
        ]

        res = client.plan_step(
            session_id="s_rot",
            history=[{"role": "user", "text": "Chuyển 200k cho sếp"}],
            tool_declarations=build_gemini_tool_declarations(),
        )
        assert key_hits == ["k1:gemini-3.6-flash", "k2:gemini-3.6-flash"]
        assert res.tool_calls == [ToolCall("lookup_contact", {"query": "sếp"})]
        assert audit.entries()[-1].model == "gemini-3.6-flash"
