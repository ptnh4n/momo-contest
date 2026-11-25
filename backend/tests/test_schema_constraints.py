"""
tests/test_schema_constraints.py

Kiểm tra toàn bộ checklist tutorial.md mục 5:
  [x] 6 bảng đúng mục 1
  [x] OCR_RECEIPT + suggested_alias -> lỗi CHECK
  [x] 2 draft cùng intent_id -> lỗi UNIQUE
  [x] ledger không có draft -> lỗi FK
  [x] lookup: "Minh" -> 2; "sếp" -> 1; "thằng Nam" và "Nam" -> cùng 1 người; "an" -> 0
  [x] output đúng 5 key và không chứa STK đầy đủ
  [x] upsert_alias: lần 2 chỉ tăng usage_count (đúng 1 dòng)
  [x] alias trùng ở contact khác -> lookup trả 2
  [x] from_ocr=True -> PermissionError
  [x] reset_demo chạy 2 lần cho cùng kết quả
  [x] lỗi giữa chừng thì dữ liệu cũ còn nguyên (rollback)
"""

import sqlite3
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.connection import get_conn, init_db, transaction
from app.db.repo.aliases import upsert_alias
from app.db.repo.contacts import lookup_contact_zero_pii
from app.db.seed import seed_demo, reset_demo, DEMO_USER_ID


# ---------------------------------------------------------------------------
# Fixture: DB file tạm đã chạy schema
# ---------------------------------------------------------------------------
@pytest.fixture
def db(tmp_path):
    """Trả về connection tới DB tạm đã chạy schema."""
    db_file = str(tmp_path / "test.db")
    init_db(db_file)
    conn = get_conn(db_file)
    yield conn
    conn.close()


@pytest.fixture
def seeded_db(tmp_path):
    """DB đã seed demo data sẵn."""
    db_file = str(tmp_path / "seeded.db")
    init_db(db_file)
    conn = get_conn(db_file)
    seed_demo(conn, DEMO_USER_ID)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Helpers seed tối thiểu (không phụ thuộc schema cũ)
# ---------------------------------------------------------------------------
def _insert_user(conn, user_id="u1", balance=10_000_000):
    conn.execute(
        """INSERT INTO users (user_id, full_name, balance, pin_hash)
           VALUES (?, ?, ?, ?)""",
        (user_id, "Test User", balance, "$argon2id$v=19$m=65536,t=3,p=4$fakehash"),
    )


def _insert_contact(conn, contact_id="C_001", owner_id="u1",
                    account_number="123456789", bank_code="MB"):
    conn.execute(
        """INSERT INTO contacts
               (contact_id, owner_id, full_name, account_number, bank_code)
           VALUES (?, ?, ?, ?, ?)""",
        (contact_id, owner_id, "Nguyễn Văn Nam", account_number, bank_code),
    )


def _insert_draft(conn, tx_id="tx_001", intent_id="intent_001",
                  source_type="CHAT_CONTACT", suggested_alias=None,
                  sender_id="u1", status="DRAFT"):
    conn.execute(
        """INSERT INTO draft_transactions
               (tx_id, intent_id, sender_id, recipient_name, recipient_account,
                bank_code, amount, source_type, suggested_alias, decision, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (tx_id, intent_id, sender_id, "Nguyễn Văn Nam", "123456789",
         "MB", 100_000, source_type, suggested_alias, "REQUIRE_HITL", status),
    )


# ===========================================================================
# MỐC 0: Schema idempotent
# ===========================================================================
class TestSchemaIdempotent:
    def test_init_twice(self, tmp_path):
        db_file = str(tmp_path / "idem.db")
        init_db(db_file)
        init_db(db_file)  # lần 2 phải không lỗi

    def test_six_tables_exist(self, db):
        tables = {
            row[0] for row in
            db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        required = {"users", "contacts", "contact_aliases",
                    "draft_transactions", "transactions", "scam_blacklist"}
        assert required.issubset(tables), f"Thiếu bảng: {required - tables}"


# ===========================================================================
# MỐC 1: PRAGMA
# ===========================================================================
class TestPragmas:
    def test_foreign_keys_on(self, db):
        row = db.execute("PRAGMA foreign_keys;").fetchone()
        assert row[0] == 1

    def test_journal_mode_wal(self, db):
        row = db.execute("PRAGMA journal_mode;").fetchone()
        assert row[0].upper() == "WAL"

    def test_foreign_key_check_clean(self, db):
        rows = db.execute("PRAGMA foreign_key_check;").fetchall()
        assert rows == []


# ===========================================================================
# MỐC 2: CHECK constraints (tutorial mục 5)
# ===========================================================================
class TestCheckConstraints:
    def test_ocr_receipt_with_suggested_alias_fails(self, db):
        """OCR_RECEIPT + suggested_alias -> lỗi CHECK (tutorial mục 5)."""
        with transaction(db):
            _insert_user(db)
            _insert_contact(db)

        with pytest.raises(sqlite3.IntegrityError):
            with transaction(db):
                _insert_draft(db, source_type="OCR_RECEIPT", suggested_alias="thằng Nam")

    def test_ocr_receipt_without_alias_ok(self, db):
        with transaction(db):
            _insert_user(db)
            _insert_contact(db)
            _insert_draft(db, source_type="OCR_RECEIPT", suggested_alias=None)
        row = db.execute("SELECT tx_id FROM draft_transactions").fetchone()
        assert row is not None

    def test_non_ocr_with_alias_ok(self, db):
        with transaction(db):
            _insert_user(db)
            _insert_contact(db)
            _insert_draft(db, source_type="CHAT_CONTACT", suggested_alias="thằng Nam")
        row = db.execute("SELECT suggested_alias FROM draft_transactions").fetchone()
        assert row["suggested_alias"] == "thằng Nam"

    def test_draft_amount_zero_fails(self, db):
        with transaction(db):
            _insert_user(db)
            _insert_contact(db)

        with pytest.raises(sqlite3.IntegrityError):
            with transaction(db):
                db.execute(
                    """INSERT INTO draft_transactions
                           (tx_id, intent_id, sender_id, recipient_name, recipient_account,
                            bank_code, amount, source_type, decision)
                       VALUES ('tx_bad', 'int_bad', 'u1', 'X', '999', 'MB', 0, 'CHAT_CONTACT', 'REQUIRE_HITL')"""
                )

    def test_user_balance_negative_fails(self, db):
        with pytest.raises(sqlite3.IntegrityError):
            with transaction(db):
                _insert_user(db, balance=-1)

    def test_invalid_trust_level_fails(self, db):
        with transaction(db):
            _insert_user(db)

        with pytest.raises(sqlite3.IntegrityError):
            with transaction(db):
                db.execute(
                    """INSERT INTO contacts
                           (contact_id, owner_id, full_name, account_number, bank_code, trust_level)
                       VALUES ('C_BAD', 'u1', 'X', '000', 'MB', 'VIP')"""
                )

    def test_invalid_source_type_fails(self, db):
        with transaction(db):
            _insert_user(db)
            _insert_contact(db)

        with pytest.raises(sqlite3.IntegrityError):
            with transaction(db):
                db.execute(
                    """INSERT INTO draft_transactions
                           (tx_id, intent_id, sender_id, recipient_name, recipient_account,
                            bank_code, amount, source_type, decision)
                       VALUES ('tx_st', 'int_st', 'u1', 'X', '999', 'MB', 100, 'UNKNOWN', 'REQUIRE_HITL')"""
                )


# ===========================================================================
# MỐC 3: UNIQUE constraints
# ===========================================================================
class TestUniqueConstraints:
    def test_duplicate_intent_id_fails(self, db):
        """2 draft cùng intent_id -> lỗi UNIQUE (tutorial mục 5)."""
        with transaction(db):
            _insert_user(db)
            _insert_contact(db)
            _insert_draft(db, tx_id="tx_001", intent_id="same_intent")

        with pytest.raises(sqlite3.IntegrityError):
            with transaction(db):
                _insert_draft(db, tx_id="tx_002", intent_id="same_intent")

    def test_different_intent_ids_ok(self, db):
        with transaction(db):
            _insert_user(db)
            _insert_contact(db)
            _insert_draft(db, tx_id="tx_001", intent_id="intent_A")
            _insert_draft(db, tx_id="tx_002", intent_id="intent_B")
        count = db.execute("SELECT COUNT(*) FROM draft_transactions").fetchone()[0]
        assert count == 2

    def test_duplicate_alias_norm_per_contact_fails(self, db):
        """UNIQUE(contact_id, alias_norm) trên contact_aliases."""
        with transaction(db):
            _insert_user(db)
            _insert_contact(db)
            db.execute(
                "INSERT INTO contact_aliases (contact_id, alias_raw, alias_norm) VALUES ('C_001', 'thằng Nam', 'thang nam')"
            )

        with pytest.raises(sqlite3.IntegrityError):
            with transaction(db):
                db.execute(
                    "INSERT INTO contact_aliases (contact_id, alias_raw, alias_norm) VALUES ('C_001', 'thằng Nam 2', 'thang nam')"
                )


# ===========================================================================
# MỐC 4: Foreign Keys
# ===========================================================================
class TestForeignKeys:
    def test_contact_requires_valid_user(self, db):
        with pytest.raises(sqlite3.IntegrityError):
            with transaction(db):
                _insert_contact(db, owner_id="nonexistent_user")

    def test_draft_requires_valid_sender(self, db):
        with pytest.raises(sqlite3.IntegrityError):
            with transaction(db):
                _insert_draft(db, sender_id="ghost_user")

    def test_transaction_requires_valid_draft(self, db):
        """ledger không có draft -> lỗi FK (tutorial mục 5)."""
        with transaction(db):
            _insert_user(db)

        with pytest.raises(sqlite3.IntegrityError):
            with transaction(db):
                db.execute(
                    """INSERT INTO transactions
                           (tx_id, sender_id, recipient_name, recipient_account,
                            bank_code, amount)
                       VALUES ('ghost_tx', 'u1', 'X', '999', 'MB', 100)"""
                )

    def test_cascade_delete_aliases_with_contact(self, db):
        with transaction(db):
            _insert_user(db)
            _insert_contact(db)
            db.execute(
                "INSERT INTO contact_aliases (contact_id, alias_raw, alias_norm) VALUES ('C_001', 'Nam', 'nam')"
            )

        with transaction(db):
            db.execute("DELETE FROM contacts WHERE contact_id = 'C_001'")

        count = db.execute("SELECT COUNT(*) FROM contact_aliases").fetchone()[0]
        assert count == 0


# ===========================================================================
# MỐC 5: Lookup (tutorial mục 5)
# ===========================================================================
class TestLookup:
    def test_lookup_minh_returns_2(self, seeded_db):
        """lookup('Minh') -> 2 kết quả (2 người tên Minh)."""
        result = lookup_contact_zero_pii(seeded_db, DEMO_USER_ID, "Minh")
        assert len(result["matches"]) == 2, f"Expected 2, got {result['matches']}"

    def test_lookup_sep_returns_1(self, seeded_db):
        """lookup('sếp') -> 1 kết quả."""
        result = lookup_contact_zero_pii(seeded_db, DEMO_USER_ID, "sếp")
        assert len(result["matches"]) == 1

    def test_lookup_thang_nam_returns_1(self, seeded_db):
        """lookup('thằng Nam') -> 1 người (Nam)."""
        result = lookup_contact_zero_pii(seeded_db, DEMO_USER_ID, "thằng Nam")
        assert len(result["matches"]) == 1
        assert result["matches"][0]["contact_id"] == "C_001"

    def test_lookup_nam_returns_same_person(self, seeded_db):
        """lookup('Nam') -> cùng 1 người với lookup('thằng Nam')."""
        r1 = lookup_contact_zero_pii(seeded_db, DEMO_USER_ID, "Nam")
        r2 = lookup_contact_zero_pii(seeded_db, DEMO_USER_ID, "thằng Nam")
        assert len(r1["matches"]) == 1
        assert r1["matches"][0]["contact_id"] == r2["matches"][0]["contact_id"] == "C_001"

    def test_lookup_an_returns_0(self, seeded_db):
        """lookup('an') -> 0 kết quả (không khớp nhầm 'van', 'nam')."""
        result = lookup_contact_zero_pii(seeded_db, DEMO_USER_ID, "an")
        assert len(result["matches"]) == 0

    def test_lookup_output_has_exactly_5_keys(self, seeded_db):
        """Output có đúng 5 key: contact_id, alias, bank_code, masked_acct, trust_level."""
        result = lookup_contact_zero_pii(seeded_db, DEMO_USER_ID, "Nam")
        assert len(result["matches"]) == 1
        match = result["matches"][0]
        assert set(match.keys()) == {"contact_id", "alias", "bank_code", "masked_acct", "trust_level"}

    def test_lookup_no_full_account_number(self, seeded_db):
        """masked_acct không được là STK đầy đủ (phải dạng ****xxx)."""
        result = lookup_contact_zero_pii(seeded_db, DEMO_USER_ID, "Nam")
        assert len(result["matches"]) == 1
        masked = result["matches"][0]["masked_acct"]
        # STK thật là '9001234567' (10 ký tự), masked phải ngắn hơn và bắt đầu bằng '****'
        assert masked.startswith("****"), f"masked_acct không bắt đầu bằng '****': {masked}"
        assert masked != "9001234567", "masked_acct không được là STK đầy đủ"
        # Phải có đúng 4 dấu * + 3 ký tự cuối
        assert len(masked) == 7, f"masked_acct phải dài 7 ký tự (****xxx): {masked}"

    def test_lookup_sep_hung_matches_sep_contact(self, seeded_db):
        """lookup('sếp Hùng') -> contact Hùng."""
        result = lookup_contact_zero_pii(seeded_db, DEMO_USER_ID, "sếp Hùng")
        assert len(result["matches"]) == 1
        assert result["matches"][0]["contact_id"] == "C_002"


# ===========================================================================
# MỐC 6: upsert_alias (tutorial mục 5)
# ===========================================================================
class TestUpsertAlias:
    def _setup(self, db):
        with transaction(db):
            _insert_user(db)
            _insert_contact(db, contact_id="C_001")

    def test_first_insert_creates_one_row(self, db):
        self._setup(db)
        with transaction(db):
            upsert_alias(db, "C_001", "thằng Nam")
        count = db.execute("SELECT COUNT(*) FROM contact_aliases WHERE contact_id='C_001'").fetchone()[0]
        assert count == 1

    def test_second_upsert_only_increments_usage_count(self, db):
        """Lần 2 chỉ tăng usage_count, đúng 1 dòng (tutorial mục 5)."""
        self._setup(db)
        with transaction(db):
            upsert_alias(db, "C_001", "thằng Nam")
        with transaction(db):
            upsert_alias(db, "C_001", "thằng Nam")

        rows = db.execute(
            "SELECT usage_count, alias_raw FROM contact_aliases WHERE contact_id='C_001'"
        ).fetchall()
        assert len(rows) == 1, "Phải chỉ có 1 dòng sau 2 lần upsert cùng alias"
        assert rows[0]["usage_count"] == 2, f"usage_count phải = 2, got {rows[0]['usage_count']}"
        # alias_raw giữ nguyên giá trị cũ
        assert rows[0]["alias_raw"] == "thằng Nam"

    def test_same_alias_norm_different_contact_adds_new_row(self, db):
        """Cùng alias_norm khác contact -> thêm dòng mới (tutorial mục 5)."""
        with transaction(db):
            _insert_user(db)
            _insert_contact(db, contact_id="C_001", account_number="111")
            _insert_contact(db, contact_id="C_002", account_number="222")

        with transaction(db):
            upsert_alias(db, "C_001", "Nam")
            upsert_alias(db, "C_002", "Nam")

        count = db.execute("SELECT COUNT(*) FROM contact_aliases WHERE alias_norm='nam'").fetchone()[0]
        assert count == 2, "Hai contact khác nhau phải có 2 alias row"

    def test_same_alias_different_contacts_lookup_returns_2(self, db):
        """alias trùng ở contact khác -> lookup trả 2 (tutorial mục 5)."""
        with transaction(db):
            _insert_user(db)
            _insert_contact(db, contact_id="C_001", account_number="111")
            _insert_contact(db, contact_id="C_002", account_number="222")
            upsert_alias(db, "C_001", "Nam")
            upsert_alias(db, "C_002", "Nam")

        result = lookup_contact_zero_pii(db, "u1", "Nam")
        assert len(result["matches"]) == 2

    def test_from_ocr_raises_permission_error(self, db):
        """from_ocr=True -> PermissionError, không ghi gì (tutorial mục 5)."""
        self._setup(db)
        with pytest.raises(PermissionError):
            with transaction(db):
                upsert_alias(db, "C_001", "thằng Nam", from_ocr=True)

        count = db.execute("SELECT COUNT(*) FROM contact_aliases").fetchone()[0]
        assert count == 0, "Không được ghi alias khi from_ocr=True"


# ===========================================================================
# MỐC 7: seed_demo / reset_demo (tutorial mục 5)
# ===========================================================================
class TestSeedReset:
    def test_seed_creates_expected_data(self, db):
        seed_demo(db, DEMO_USER_ID)
        users = db.execute("SELECT COUNT(*) FROM users WHERE user_id=?", (DEMO_USER_ID,)).fetchone()[0]
        contacts = db.execute("SELECT COUNT(*) FROM contacts WHERE owner_id=?", (DEMO_USER_ID,)).fetchone()[0]
        blacklist = db.execute("SELECT COUNT(*) FROM scam_blacklist").fetchone()[0]
        assert users == 1
        assert contacts == 4
        assert blacklist >= 3

    def test_seed_pin_is_hashed(self, db):
        seed_demo(db, DEMO_USER_ID)
        row = db.execute("SELECT pin_hash FROM users WHERE user_id=?", (DEMO_USER_ID,)).fetchone()
        pin_hash = row["pin_hash"]
        assert pin_hash != "123456", "PIN không được lưu thô"
        assert len(pin_hash) > 10, "pin_hash quá ngắn"

    def test_reset_demo_idempotent(self, db):
        """reset_demo chạy 2 lần cho cùng kết quả (tutorial mục 5)."""
        reset_demo(db, DEMO_USER_ID)
        cnt1_users = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        cnt1_contacts = db.execute("SELECT COUNT(*) FROM contacts").fetchone()[0]
        cnt1_blacklist = db.execute("SELECT COUNT(*) FROM scam_blacklist").fetchone()[0]

        reset_demo(db, DEMO_USER_ID)
        cnt2_users = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        cnt2_contacts = db.execute("SELECT COUNT(*) FROM contacts").fetchone()[0]
        cnt2_blacklist = db.execute("SELECT COUNT(*) FROM scam_blacklist").fetchone()[0]

        assert cnt1_users == cnt2_users, "Số user phải giống sau 2 lần reset"
        assert cnt1_contacts == cnt2_contacts, "Số contact phải giống sau 2 lần reset"
        assert cnt1_blacklist == cnt2_blacklist, "Blacklist phải giống sau 2 lần reset"

    def test_rollback_on_error_preserves_old_data(self, db):
        """Lỗi giữa chừng thì dữ liệu cũ còn nguyên (tutorial mục 5)."""
        # Seed ban đầu
        seed_demo(db, DEMO_USER_ID)
        old_balance = db.execute(
            "SELECT balance FROM users WHERE user_id=?", (DEMO_USER_ID,)
        ).fetchone()[0]

        # Thử rollback: transaction thất bại giữa chừng
        try:
            with transaction(db):
                db.execute(
                    "UPDATE users SET balance = 999 WHERE user_id=?", (DEMO_USER_ID,)
                )
                raise RuntimeError("Giả lỗi")
        except RuntimeError:
            pass

        new_balance = db.execute(
            "SELECT balance FROM users WHERE user_id=?", (DEMO_USER_ID,)
        ).fetchone()[0]
        assert new_balance == old_balance, "Balance phải rollback về giá trị cũ"

    def test_history_transaction_in_draft_first(self, db):
        """Lịch sử giao dịch: draft (COMPLETED) phải INSERT trước transactions (FK)."""
        seed_demo(db, DEMO_USER_ID)
        tx_in_draft = db.execute(
            "SELECT status FROM draft_transactions WHERE status='COMPLETED'"
        ).fetchall()
        tx_in_ledger = db.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        assert len(tx_in_draft) > 0, "Phải có draft COMPLETED"
        assert tx_in_ledger > 0, "Phải có transaction trong sổ cái"

    def test_reset_demo_rollback_preserves_all_user_data(self, db, monkeypatch):
        """reset_demo lỗi ở bước seed -> rollback toàn bộ (users, contacts, aliases, blacklist, transactions còn nguyên)."""
        import app.db.seed as seed_mod

        seed_demo(db, DEMO_USER_ID)
        # Thay đổi dữ liệu để phân biệt trạng thái trước reset với trạng thái mặc định của seed_demo
        with transaction(db):
            db.execute("UPDATE users SET balance = 5555555 WHERE user_id = ?", (DEMO_USER_ID,))
            db.execute(
                "INSERT INTO contact_aliases (contact_id, alias_raw, alias_norm) VALUES ('C_001', 'Nam bạn thân', 'nam ban than')"
            )
            db.execute(
                "INSERT INTO scam_blacklist (account_number, bank_code, scam_type) VALUES ('9999999999', 'BIDV', 'CUSTOM_SCAM')"
            )

        def _snapshot():
            users = [dict(r) for r in db.execute(
                "SELECT * FROM users WHERE user_id = ? ORDER BY user_id", (DEMO_USER_ID,)
            ).fetchall()]
            contacts = [dict(r) for r in db.execute(
                "SELECT * FROM contacts WHERE owner_id = ? ORDER BY contact_id", (DEMO_USER_ID,)
            ).fetchall()]
            aliases = [dict(r) for r in db.execute(
                """SELECT ca.* FROM contact_aliases ca
                   JOIN contacts c ON c.contact_id = ca.contact_id
                   WHERE c.owner_id = ? ORDER BY ca.id""",
                (DEMO_USER_ID,),
            ).fetchall()]
            blacklist = [dict(r) for r in db.execute(
                "SELECT * FROM scam_blacklist ORDER BY account_number, bank_code"
            ).fetchall()]
            txs = [dict(r) for r in db.execute(
                "SELECT * FROM transactions WHERE sender_id = ? ORDER BY tx_id", (DEMO_USER_ID,)
            ).fetchall()]
            return users, contacts, aliases, blacklist, txs

        before = _snapshot()

        def _failing_insert(conn, user_id):
            raise RuntimeError("Giả lập lỗi khi seed")

        monkeypatch.setattr(seed_mod, "_insert_demo_data", _failing_insert)

        with pytest.raises(RuntimeError, match="Giả lập lỗi khi seed"):
            reset_demo(db, DEMO_USER_ID)

        after = _snapshot()
        assert after == before, "Dữ liệu users/contacts/aliases/blacklist/transactions phải y như trước khi reset"

    def test_reset_demo_preserves_existing_blacklist_entries(self, db):
        """reset_demo không DELETE scam_blacklist, chỉ INSERT OR REPLACE các dòng mẫu."""
        seed_demo(db, DEMO_USER_ID)
        with transaction(db):
            db.execute(
                "INSERT INTO scam_blacklist (account_number, bank_code, scam_type) VALUES ('8888888888', 'ACB', 'EXTRA_SCAM')"
            )

        reset_demo(db, DEMO_USER_ID)

        extra = db.execute(
            "SELECT scam_type FROM scam_blacklist WHERE account_number = '8888888888' AND bank_code = 'ACB'"
        ).fetchone()
        assert extra is not None and extra["scam_type"] == "EXTRA_SCAM"
        total_bl = db.execute("SELECT COUNT(*) FROM scam_blacklist").fetchone()[0]
        assert total_bl == 4


# ===========================================================================
# MỐC 8: Rollback transaction context manager
# ===========================================================================
class TestTransactionRollback:
    def test_rollback_on_exception(self, db):
        with transaction(db):
            _insert_user(db)

        try:
            with transaction(db):
                _insert_contact(db, contact_id="C_TEMP")
                raise RuntimeError("Giả lỗi giữa chừng")
        except RuntimeError:
            pass

        count = db.execute("SELECT COUNT(*) FROM contacts").fetchone()[0]
        assert count == 0

    def test_commit_on_success(self, db):
        with transaction(db):
            _insert_user(db)
            _insert_contact(db)

        count = db.execute("SELECT COUNT(*) FROM contacts").fetchone()[0]
        assert count == 1
