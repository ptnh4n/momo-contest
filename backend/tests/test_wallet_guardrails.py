"""
tests/test_wallet_guardrails.py

Kiểm thử toàn bộ API ví giả lập + lớp bảo vệ cứng + Privacy Proxy + Zero-PII + Memory Poisoning:
  (1) Idempotency Key chống chuyển đúp (POST /api/chat & POST /api/drafts/{tx_id}/confirm)
  (2) Deterministic Currency Parser (gặp '500' mơ hồ -> CLARIFY, không để LLM đoán)
  (3) Hard Veto Blacklist (lúc tạo draft & lúc confirm)
  (4) Luật cứng OCR_ALWAYS_HITL (nguồn OCR bắt buộc qua HITL, không bao giờ PASS_AUTO)
  (5) Ngưỡng sinh trắc học/PIN NHNN (>10tr/lần, >20tr/ngày)
  (6) Privacy Proxy (Regex Masking/De-masking tạm trong RAM có TTL) & Zero-PII lookup_contact
  (7) 2 luật cứng chống Memory Poisoning: NO_OCR_ALIAS & NO_SILENT_OVERWRITE + Post-PIN Commit
"""

from __future__ import annotations

import os
import re
import sys
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.connection import get_conn, init_db, transaction
from app.db.seed import DEMO_PIN, DEMO_USER_ID, seed_demo
from app.wallet.api import create_app
from app.wallet.currency_parser import parse_currency
from app.wallet.guardrails import (
    NHNN_DAILY_TX_LIMIT,
    NHNN_SINGLE_TX_LIMIT,
    save_contact_alias_checked,
)
from app.wallet.privacy_proxy import ALLOWED_ZERO_PII_KEYS, PrivacyProxy, lookup_contact
from app.wallet.service import confirm_draft, create_draft_transfer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def db_file(tmp_path):
    path = str(tmp_path / "wallet_test.db")
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


@pytest.fixture
def client(db_file):
    app = create_app(db_path=db_file, seed_if_empty=False)
    with TestClient(app) as tc:
        yield tc


# ===========================================================================
# (1) IDEMPOTENCY KEY CHỐNG CHUYỂN ĐÚP
# ===========================================================================
class TestIdempotency:
    def test_duplicate_intent_id_returns_same_draft(self, client, db):
        intent_id = str(uuid4())
        payload = {
            "user_id": DEMO_USER_ID,
            "intent_id": intent_id,
            "message": "chuyển 500k cho thằng Nam",
        }
        r1 = client.post("/api/chat", json=payload)
        r2 = client.post("/api/chat", json=payload)

        assert r1.status_code == 200
        assert r2.status_code == 200
        d1 = r1.json()["draft"]
        d2 = r2.json()["draft"]
        assert d1["tx_id"] == d2["tx_id"]
        count = db.execute(
            "SELECT COUNT(*) FROM draft_transactions WHERE intent_id = ?", (intent_id,)
        ).fetchone()[0]
        assert count == 1

    def test_duplicate_intent_id_pass_auto_deducts_once(self, client, db):
        """PASS_AUTO gửi lặp cùng intent_id chỉ trừ tiền đúng 1 lần."""
        intent_id = str(uuid4())
        payload = {
            "user_id": DEMO_USER_ID,
            "intent_id": intent_id,
            "message": "chuyển 50k cho thằng Nam",
        }
        r1 = client.post("/api/chat", json=payload)
        r2 = client.post("/api/chat", json=payload)

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["decision"] == "PASS_AUTO"
        assert r2.json()["decision"] == "PASS_AUTO"
        assert r1.json()["draft"]["tx_id"] == r2.json()["draft"]["tx_id"]

        balance = db.execute(
            "SELECT balance FROM users WHERE user_id = ?", (DEMO_USER_ID,)
        ).fetchone()[0]
        assert balance == 10_000_000 - 50_000

    def test_confirm_twice_deducts_balance_once(self, client, db):
        """Confirm 2 lần trên cùng draft chỉ trừ tiền 1 lần (tutorial mục 5)."""
        chat_res = client.post(
            "/api/chat",
            json={
                "user_id": DEMO_USER_ID,
                "intent_id": str(uuid4()),
                "message": "chuyển 500k cho thằng Nam",
            },
        )
        assert chat_res.status_code == 200
        tx_id = chat_res.json()["draft"]["tx_id"]

        c1 = client.post(
            f"/api/drafts/{tx_id}/confirm",
            json={"user_id": DEMO_USER_ID, "pin_code": DEMO_PIN},
        )
        c2 = client.post(
            f"/api/drafts/{tx_id}/confirm",
            json={"user_id": DEMO_USER_ID, "pin_code": DEMO_PIN},
        )

        assert c1.status_code == 200
        assert c2.status_code == 200
        assert c1.json()["new_balance"] == 9_500_000
        assert c2.json()["new_balance"] == 9_500_000
        tx_cnt = db.execute(
            "SELECT COUNT(*) FROM transactions WHERE tx_id = ?", (tx_id,)
        ).fetchone()[0]
        assert tx_cnt == 1


# ===========================================================================
# (2) DETERMINISTIC CURRENCY PARSER
# ===========================================================================
class TestDeterministicCurrencyParser:
    @pytest.mark.parametrize("raw_msg,expected_high,expected_low", [
        ("chuyển 500 cho thằng Nam", 500_000, 500),
        ("gửi 200 cho sếp", 200_000, 200),
        ("chuyển 1000 cho Nam", 1_000_000, 1_000),
    ])
    def test_ambiguous_bare_number_forces_clarify(self, raw_msg, expected_high, expected_low):
        res = parse_currency(raw_msg)
        assert res.status == "AMBIGUOUS"
        assert res.amount is None
        amounts = [o.amount for o in res.options]
        assert amounts == [expected_high, expected_low]

    @pytest.mark.parametrize("msg,expected", [
        ("chuyển 500k cho Nam", 500_000),
        ("chuyển 200 nghìn cho sếp", 200_000),
        ("chuyển 1.5tr cho Nam", 1_500_000),
        ("chuyển 1,5 triệu cho Nam", 1_500_000),
        ("chuyển 2tr5 cho sếp", 2_500_000),
        ("chuyển 500.000đ cho Nam", 500_000),
        ("chuyển 500đ cho Nam", 500),
        ("chuyển 10.000.000 vnd cho sếp", 10_000_000),
        ("chuyển 500.000 cho Nam", 500_000),
        ("chuyển 500000 cho Nam", 500_000),
        ("chuyển 500k tới <ACCT_1>", 500_000),
    ])
    def test_unambiguous_amounts_parsed_deterministically(self, msg, expected):
        res = parse_currency(msg)
        assert res.status == "OK"
        assert res.amount == expected

    @pytest.mark.parametrize("msg", ["chuyển 0đ cho Nam", "chuyển -500k cho Nam", "chuyển 0 cho Nam"])
    def test_zero_or_negative_rejected(self, msg):
        res = parse_currency(msg)
        assert res.status == "INVALID"

    def test_chat_api_ambiguous_500_returns_clarify_and_writes_no_db(self, client, db):
        intent_clarify = str(uuid4())
        before_drafts = db.execute("SELECT COUNT(*) FROM draft_transactions").fetchone()[0]

        res = client.post(
            "/api/chat",
            json={
                "user_id": DEMO_USER_ID,
                "intent_id": intent_clarify,
                "message": "chuyển 500 cho thằng Nam",
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["decision"] == "CLARIFY"
        assert body["draft"] is None
        assert body["clarify_options"]["kind"] == "CURRENCY_UNIT"
        assert [o["amount"] for o in body["clarify_options"]["options"]] == [500_000, 500]

        # CLARIFY tuyệt đối không ghi DB
        after_drafts = db.execute("SELECT COUNT(*) FROM draft_transactions").fetchone()[0]
        assert after_drafts == before_drafts

        # User chọn 500.000 VNĐ -> tạo draft bình thường
        res2 = client.post(
            "/api/chat",
            json={
                "user_id": DEMO_USER_ID,
                "intent_id": str(uuid4()),
                "message": "500.000 VNĐ",
                "clarify_choice": {
                    "ref_intent_id": intent_clarify,
                    "amount": 500_000,
                },
            },
        )
        assert res2.status_code == 200
        body2 = res2.json()
        assert body2["decision"] == "REQUIRE_HITL"
        assert body2["draft"]["amount"] == 500_000
        assert body2["draft"]["recipient_account"] == "9001234567"

    def test_amount_exceeding_balance_returns_422(self, client):
        res = client.post(
            "/api/chat",
            json={
                "user_id": DEMO_USER_ID,
                "intent_id": str(uuid4()),
                "message": "chuyển 15tr cho thằng Nam",
            },
        )
        assert res.status_code == 422
        assert res.json()["error_code"] == "INSUFFICIENT_BALANCE"


# ===========================================================================
# (3) HARD VETO BLACKLIST
# ===========================================================================
class TestHardVetoBlacklist:
    def test_blacklisted_account_blocked_immediately(self, client):
        """STK 0000000001 MB trong scam_blacklist -> lập tức BLOCK_ANOMALY & status=BLOCKED."""
        res = client.post(
            "/api/chat",
            json={
                "user_id": DEMO_USER_ID,
                "intent_id": str(uuid4()),
                "message": "chuyển 50k tới STK 0000000001 MB",
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["decision"] == "BLOCK_ANOMALY"
        assert body["risk_score"] == 1.0
        assert body["draft"]["status"] == "BLOCKED"
        assert any("BLACKLIST_HARD_VETO" in r for r in body["xai_reasons"])

        # Không được phép confirm draft BLOCKED
        tx_id = body["draft"]["tx_id"]
        c_res = client.post(
            f"/api/drafts/{tx_id}/confirm",
            json={"user_id": DEMO_USER_ID, "pin_code": DEMO_PIN},
        )
        assert c_res.status_code == 409
        assert c_res.json()["error_code"] == "DRAFT_NOT_CONFIRMABLE"

    def test_blacklisted_between_draft_and_confirm_returns_409(self, client, db):
        """STK bị đưa vào blacklist sau khi tạo draft, trước khi confirm -> 409 BLOCKED_BY_BLACKLIST."""
        res = client.post(
            "/api/chat",
            json={
                "user_id": DEMO_USER_ID,
                "intent_id": str(uuid4()),
                "message": "chuyển 500k tới STK 8888777766 MB",
            },
        )
        assert res.status_code == 200
        tx_id = res.json()["draft"]["tx_id"]

        with transaction(db):
            db.execute(
                "INSERT INTO scam_blacklist (account_number, bank_code, scam_type) VALUES ('8888777766', 'MB', 'SCAM_MOI')"
            )

        c_res = client.post(
            f"/api/drafts/{tx_id}/confirm",
            json={"user_id": DEMO_USER_ID, "pin_code": DEMO_PIN},
        )
        assert c_res.status_code == 409
        assert c_res.json()["error_code"] == "BLOCKED_BY_BLACKLIST"

        status = db.execute(
            "SELECT status FROM draft_transactions WHERE tx_id = ?", (tx_id,)
        ).fetchone()[0]
        assert status == "BLOCKED"


# ===========================================================================
# (4) LUẬT CỨNG OCR_ALWAYS_HITL
# ===========================================================================
class TestOcrAlwaysHitl:
    def test_ocr_receipt_small_amount_trusted_contact_never_pass_auto(self, db):
        """
        Dù amount <= auto_approve_limit (50k <= 100k) và contact là TRUSTED (C_001 Nam),
        nguồn OCR_RECEIPT bắt buộc REQUIRE_HITL, không bao giờ PASS_AUTO.
        """
        resp = create_draft_transfer(
            db,
            intent_id=str(uuid4()),
            sender_id=DEMO_USER_ID,
            amount=50_000,
            source_type="OCR_RECEIPT",
            contact_id="C_001",
            note="Hóa đơn OCR",
        )
        assert resp.decision == "REQUIRE_HITL"
        assert resp.draft is not None
        assert resp.draft.status == "DRAFT"
        assert resp.new_balance is None
        assert any("OCR_ALWAYS_HITL" in r for r in resp.xai_reasons)

        balance = db.execute(
            "SELECT balance FROM users WHERE user_id = ?", (DEMO_USER_ID,)
        ).fetchone()[0]
        assert balance == 10_000_000

    def test_chat_with_image_base64_forces_ocr_hitl(self, client):
        res = client.post(
            "/api/chat",
            json={
                "user_id": DEMO_USER_ID,
                "intent_id": str(uuid4()),
                "message": "chuyển 50k cho thằng Nam",
                "image_base64": "data:image/png;base64,iVBORw0KGgo=",
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["decision"] == "REQUIRE_HITL"
        assert body["draft"]["source_type"] == "OCR_RECEIPT"
        assert body["draft"]["suggested_alias"] is None


# ===========================================================================
# (5) NGƯỠNG SINH TRẮC HỌC / PIN NHNN (>10TR/LẦN, >20TR/NGÀY)
# ===========================================================================
class TestNhnnThreshold:
    @pytest.mark.parametrize(("daily_total", "amount", "expected", "reason"), [
        (0, NHNN_SINGLE_TX_LIMIT - 1, "PASS_AUTO", None),
        (0, NHNN_SINGLE_TX_LIMIT, "PASS_AUTO", None),
        (0, NHNN_SINGLE_TX_LIMIT + 1, "REQUIRE_HITL", "NHNN_SINGLE_TX_LIMIT"),
        (15_000_000, NHNN_DAILY_TX_LIMIT - 15_000_001, "PASS_AUTO", None),
        (15_000_000, NHNN_DAILY_TX_LIMIT - 15_000_000, "PASS_AUTO", None),
        (15_000_000, NHNN_DAILY_TX_LIMIT - 14_999_999, "REQUIRE_HITL", "NHNN_DAILY_TX_LIMIT"),
    ])
    def test_excel_threshold_boundaries(self, db, daily_total, amount, expected, reason):
        with transaction(db):
            db.execute(
                """UPDATE users
                   SET balance = 50000000, auto_approve_limit = 30000000, daily_transferred = ?
                   WHERE user_id = ?""",
                (daily_total, DEMO_USER_ID),
            )

        resp = create_draft_transfer(
            db,
            intent_id=str(uuid4()),
            sender_id=DEMO_USER_ID,
            amount=amount,
            source_type="CHAT_CONTACT",
            contact_id="C_001",
        )
        assert resp.decision == expected
        assert resp.draft is not None
        assert resp.draft.status == ("DRAFT" if reason else "COMPLETED")
        if reason:
            assert any(reason in item for item in resp.xai_reasons)

    def test_below_all_limits_trusted_contact_is_pass_auto(self, db):
        resp = create_draft_transfer(
            db,
            intent_id=str(uuid4()),
            sender_id=DEMO_USER_ID,
            amount=100_000,
            source_type="CHAT_CONTACT",
            contact_id="C_001",
        )
        assert resp.decision == "PASS_AUTO"
        assert resp.draft is not None and resp.draft.status == "COMPLETED"
        assert resp.new_balance == 9_900_000


# ===========================================================================
# (6) PRIVACY PROXY (RAM TTL) & ZERO-PII LOOKUP_CONTACT
# ===========================================================================
class TestPrivacyProxyAndZeroPii:
    def test_mask_and_demask_acct_phone_cccd(self):
        proxy = PrivacyProxy(ttl_seconds=300)
        raw = "Chuyển 500k tới STK 19031234567890 VCB, SĐT 0987654321, CCCD 079203012345"
        res = proxy.mask("sess_1", raw)

        assert "19031234567890" not in res.masked_text
        assert "0987654321" not in res.masked_text
        assert "079203012345" not in res.masked_text
        assert "<ACCT_1>" in res.masked_text
        assert "<PHONE_1>" in res.masked_text
        assert "<CCCD_1>" in res.masked_text

        restored = proxy.demask("sess_1", res.masked_text)
        assert restored == raw

    def test_phone_used_as_labeled_stk_masked_as_acct(self):
        """'STK 0987654321 MB' phải được mask là <ACCT_1>, không phải <PHONE_1>."""
        proxy = PrivacyProxy()
        res = proxy.mask("sess_2", "Chuyển 500k tới STK 0987654321 MB")
        assert "<ACCT_1>" in res.masked_text
        assert "0987654321" not in res.masked_text
        assert proxy.resolve_token("sess_2", "<ACCT_1>") == "0987654321"

    def test_ram_ttl_expiration_purges_mapping(self):
        now = [1000.0]
        proxy = PrivacyProxy(ttl_seconds=60.0, clock=lambda: now[0])
        res = proxy.mask("sess_ttl", "STK 9001234567")
        assert proxy.demask("sess_ttl", res.masked_text) == "STK 9001234567"

        # Quá TTL 60 giây -> mapping tự xóa khỏi RAM
        now[0] = 1061.0
        assert not proxy.has_session("sess_ttl")
        assert proxy.resolve_token("sess_ttl", "<ACCT_1>") is None
        assert proxy.demask("sess_ttl", "<ACCT_1>") == "<ACCT_1>"

    def test_lookup_contact_locked_zero_pii(self, db):
        """lookup_contact khóa đầu ra Zero-PII: chỉ đúng whitelist key, masked_acct dạng ****xxx."""
        out = lookup_contact(db, DEMO_USER_ID, "thằng Nam")
        assert len(out["matches"]) == 1
        m = out["matches"][0]
        assert set(m.keys()) == ALLOWED_ZERO_PII_KEYS
        assert m["contact_id"] == "C_001"
        assert m["masked_acct"] == "****567"
        assert re.match(r"^\*{4}\d{3}$", m["masked_acct"])
        assert "9001234567" not in str(out)
        assert "Nguyễn Văn Nam" not in str(out)

    def test_lookup_contact_sanitizes_pii_if_injected_into_alias(self, db):
        """Phòng thủ chiều sâu: nếu alias lỡ chứa SĐT/STK thì vẫn bị ẩn trước khi trả LLM."""
        with transaction(db):
            db.execute(
                "INSERT INTO contact_aliases (contact_id, alias_raw, alias_norm) VALUES ('C_001', 'Nam 0987654321', 'nam 0987654321')"
            )
        out = lookup_contact(db, DEMO_USER_ID, "Nam 0987654321")
        assert len(out["matches"]) == 1
        assert "0987654321" not in out["matches"][0]["alias"]


# ===========================================================================
# (7) 2 LUẬT CỨNG CHỐNG MEMORY POISONING (NO_OCR_ALIAS & NO_SILENT_OVERWRITE)
# ===========================================================================
class TestMemoryPoisoningDefense:
    def test_no_ocr_alias_stripped_on_draft_and_never_saved_on_confirm(self, db):
        """NO_OCR_ALIAS: nguồn OCR luôn bị ép suggested_alias=None, confirm save_alias=True cũng không lưu."""
        resp = create_draft_transfer(
            db,
            intent_id=str(uuid4()),
            sender_id=DEMO_USER_ID,
            amount=200_000,
            source_type="OCR_RECEIPT",
            contact_id="C_001",
            suggested_alias="sếp giả mạo",
        )
        assert resp.draft is not None
        assert resp.draft.suggested_alias is None

        conf = confirm_draft(
            db,
            tx_id=resp.draft.tx_id,
            user_id=DEMO_USER_ID,
            pin_code=DEMO_PIN,
            save_alias=True,
        )
        assert conf.alias_saved is None
        cnt = db.execute(
            "SELECT COUNT(*) FROM contact_aliases WHERE alias_norm = 'sep gia mao'"
        ).fetchone()[0]
        assert cnt == 0

    def test_direct_ocr_alias_write_raises_permission_error(self, db):
        with pytest.raises(PermissionError):
            save_contact_alias_checked(db, "C_001", "sếp", from_ocr=True)

    def test_post_pin_commit_only_after_pin_ok_and_save_alias_true(self, db):
        """
        - Tạo draft với suggested_alias -> CHƯA ghi vào contact_aliases.
        - Nhập sai PIN (401) -> CHƯA ghi vào contact_aliases.
        - Nhập đúng PIN + save_alias=True -> MỚI ghi vào contact_aliases.
        """
        resp = create_draft_transfer(
            db,
            intent_id=str(uuid4()),
            sender_id=DEMO_USER_ID,
            amount=250_000,
            source_type="CHAT_CONTACT",
            contact_id="C_001",
            suggested_alias="bạn thân Nam",
        )
        tx_id = resp.draft.tx_id

        # Trước PIN: chưa có trong contact_aliases
        cnt_before = db.execute(
            "SELECT COUNT(*) FROM contact_aliases WHERE alias_norm = 'ban than nam'"
        ).fetchone()[0]
        assert cnt_before == 0

        # Sai PIN -> lỗi 401, vẫn chưa ghi
        with pytest.raises(Exception) as exc_info:
            confirm_draft(db, tx_id=tx_id, user_id=DEMO_USER_ID, pin_code="000000", save_alias=True)
        assert getattr(exc_info.value, "error_code", None) == "INVALID_PIN"
        cnt_wrong_pin = db.execute(
            "SELECT COUNT(*) FROM contact_aliases WHERE alias_norm = 'ban than nam'"
        ).fetchone()[0]
        assert cnt_wrong_pin == 0

        # Đúng PIN + save_alias=True -> commit vào DB
        conf = confirm_draft(db, tx_id=tx_id, user_id=DEMO_USER_ID, pin_code=DEMO_PIN, save_alias=True)
        assert conf.alias_saved == "bạn thân Nam"
        cnt_after = db.execute(
            "SELECT COUNT(*) FROM contact_aliases WHERE contact_id = 'C_001' AND alias_norm = 'ban than nam'"
        ).fetchone()[0]
        assert cnt_after == 1

    def test_pass_auto_never_commits_alias(self, db):
        """PASS_AUTO không yêu cầu PIN -> tuyệt đối không tự lưu alias."""
        resp = create_draft_transfer(
            db,
            intent_id=str(uuid4()),
            sender_id=DEMO_USER_ID,
            amount=50_000,
            source_type="CHAT_CONTACT",
            contact_id="C_001",
            suggested_alias="auto alias",
        )
        assert resp.decision == "PASS_AUTO"
        cnt = db.execute(
            "SELECT COUNT(*) FROM contact_aliases WHERE alias_norm = 'auto alias'"
        ).fetchone()[0]
        assert cnt == 0

    def test_selecting_contact_only_proposes_alias_until_pin(self, client, db):
        original_intent = str(uuid4())
        response = client.post("/api/chat", json={
            "user_id": DEMO_USER_ID,
            "intent_id": original_intent,
            "message": "Chuyển 300k cho Minh",
        })
        assert response.json()["decision"] == "CLARIFY"

        def aliases():
            return db.execute(
                "SELECT contact_id, alias_norm, usage_count FROM contact_aliases ORDER BY contact_id, alias_norm"
            ).fetchall()

        before = aliases()
        choice = client.post("/api/chat", json={
            "user_id": DEMO_USER_ID,
            "intent_id": str(uuid4()),
            "message": "Chọn Minh",
            "clarify_choice": {"ref_intent_id": original_intent, "contact_id": "C_003"},
        })
        assert choice.status_code == 200
        draft = choice.json()["draft"]
        assert draft["suggested_alias"] == "Minh"
        assert aliases() == before

        confirmation = client.post(f"/api/drafts/{draft['tx_id']}/confirm", json={
            "user_id": DEMO_USER_ID, "pin_code": DEMO_PIN, "save_alias": True,
        })
        assert confirmation.status_code == 200
        assert confirmation.json()["alias_saved"] == "Minh"
        assert aliases() != before

    def test_no_silent_overwrite_duplicate_alias_triggers_clarify(self, client, db):
        """
        NO_SILENT_OVERWRITE:
        'sếp' đang thuộc C_002 (Trần Quốc Hùng).
        Khi gắn thêm 'sếp' cho C_001 (sau PIN OK), KHÔNG ghi đè C_002 ->
        lần sau lookup_contact('sếp') trả 2 người và POST /api/chat ép CLARIFY.
        """
        resp = create_draft_transfer(
            db,
            intent_id=str(uuid4()),
            sender_id=DEMO_USER_ID,
            amount=300_000,
            source_type="CHAT_CONTACT",
            contact_id="C_001",
            suggested_alias="sếp",
        )
        confirm_draft(
            db,
            tx_id=resp.draft.tx_id,
            user_id=DEMO_USER_ID,
            pin_code=DEMO_PIN,
            save_alias=True,
        )

        # Cả C_001 và C_002 đều có alias 'sep'
        lookup_res = lookup_contact(db, DEMO_USER_ID, "sếp")
        assert len(lookup_res["matches"]) == 2
        assert {m["contact_id"] for m in lookup_res["matches"]} == {"C_001", "C_002"}

        # Gọi /api/chat với 'sếp' -> bắt buộc CLARIFY (không đoán)
        chat_res = client.post(
            "/api/chat",
            json={
                "user_id": DEMO_USER_ID,
                "intent_id": str(uuid4()),
                "message": "chuyển 200k cho sếp",
            },
        )
        assert chat_res.status_code == 200
        body = chat_res.json()
        assert body["decision"] == "CLARIFY"
        assert body["clarify_options"]["kind"] == "CONTACT"
        assert len(body["clarify_options"]["options"]) == 2


# ===========================================================================
# ENDPOINT SMOKE TESTS (STATE, CANCEL, RESET, HEALTH)
# ===========================================================================
class TestWalletEndpoints:
    def test_health_state_cancel_and_reset(self, client):
        # /health
        h = client.get("/health")
        assert h.status_code == 200
        assert h.json() == {"status": "ok", "gatekeeper_loaded": False}

        # /api/wallet/state
        st = client.get(f"/api/wallet/state?user_id={DEMO_USER_ID}")
        assert st.status_code == 200
        assert st.json()["balance"] == 10_000_000
        assert len(st.json()["contacts"]) == 4

        # /api/drafts/{tx_id}/cancel
        draft_res = client.post(
            "/api/chat",
            json={
                "user_id": DEMO_USER_ID,
                "intent_id": str(uuid4()),
                "message": "chuyển 500k cho thằng Nam",
            },
        )
        tx_id = draft_res.json()["draft"]["tx_id"]
        cancel_res = client.post(
            f"/api/drafts/{tx_id}/cancel",
            json={"user_id": DEMO_USER_ID},
        )
        assert cancel_res.status_code == 200
        assert cancel_res.json() == {"status": "CANCELLED", "tx_id": tx_id}

        # /api/demo/reset
        reset_res = client.post("/api/demo/reset", json={"user_id": DEMO_USER_ID})
        assert reset_res.status_code == 200
        assert reset_res.json() == {"status": "RESET_OK", "balance": 10_000_000}
