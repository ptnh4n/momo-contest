"""
seed.py — Dữ liệu demo cho SafePay Agent.
Nguồn: tutorial.md mục 0, 1, 5.

PIN demo '123456' được băm bằng argon2.
"""

import sqlite3
from argon2 import PasswordHasher

from app.db.connection import transaction

_ph = PasswordHasher()

DEMO_PIN = "123456"
DEMO_USER_ID = "u_demo"


def _hash_pin(pin: str) -> str:
    return _ph.hash(pin)


def _insert_demo_data(conn: sqlite3.Connection, user_id: str) -> None:
    """
    Thực hiện các lệnh INSERT dữ liệu demo (không tự mở transaction).
    Hàm gọi phải đảm bảo đang nằm trong một transaction.
    """
    pin_hash = _hash_pin(DEMO_PIN)

    # --- User ---
    conn.execute(
        """
        INSERT INTO users (user_id, full_name, balance, pin_hash)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, "Người Dùng Demo", 10_000_000, pin_hash),
    )

    # --- Contacts ---
    # C1: Nam
    conn.execute(
        """INSERT INTO contacts (contact_id, owner_id, full_name, account_number, bank_code, trust_level)
           VALUES ('C_001', ?, 'Nguyễn Văn Nam', '9001234567', 'MB', 'TRUSTED')""",
        (user_id,),
    )
    # C2: Hùng (sếp)
    conn.execute(
        """INSERT INTO contacts (contact_id, owner_id, full_name, account_number, bank_code, trust_level)
           VALUES ('C_002', ?, 'Trần Quốc Hùng', '9007654321', 'VCB', 'TRUSTED')""",
        (user_id,),
    )
    # C3: Minh 1
    conn.execute(
        """INSERT INTO contacts (contact_id, owner_id, full_name, account_number, bank_code, trust_level)
           VALUES ('C_003', ?, 'Lê Quang Minh', '9001112222', 'TCB', 'NEW')""",
        (user_id,),
    )
    # C4: Minh 2
    conn.execute(
        """INSERT INTO contacts (contact_id, owner_id, full_name, account_number, bank_code, trust_level)
           VALUES ('C_004', ?, 'Phạm Thanh Minh', '9003334444', 'ACB', 'NEW')""",
        (user_id,),
    )

    # --- Aliases ---
    # Nam: 'thằng Nam', 'Nam'
    conn.execute(
        "INSERT INTO contact_aliases (contact_id, alias_raw, alias_norm) VALUES ('C_001', 'thằng Nam', 'thang nam')"
    )
    conn.execute(
        "INSERT INTO contact_aliases (contact_id, alias_raw, alias_norm) VALUES ('C_001', 'Nam', 'nam')"
    )
    # Hùng: 'sếp', 'sếp Hùng'
    conn.execute(
        "INSERT INTO contact_aliases (contact_id, alias_raw, alias_norm) VALUES ('C_002', 'sếp', 'sep')"
    )
    conn.execute(
        "INSERT INTO contact_aliases (contact_id, alias_raw, alias_norm) VALUES ('C_002', 'sếp Hùng', 'sep hung')"
    )
    # Minh1: 'Minh'
    conn.execute(
        "INSERT INTO contact_aliases (contact_id, alias_raw, alias_norm) VALUES ('C_003', 'Minh', 'minh')"
    )
    # Minh2: 'Minh' — alias_norm giống nhau nhưng contact khác → lookup trả 2
    conn.execute(
        "INSERT INTO contact_aliases (contact_id, alias_raw, alias_norm) VALUES ('C_004', 'Minh', 'minh')"
    )

    # --- Scam Blacklist (>= 3 STK) ---
    conn.execute(
        "INSERT OR REPLACE INTO scam_blacklist (account_number, bank_code, scam_type) VALUES ('0000000001', 'MB', 'GIA_DANH_CONG_AN')"
    )
    conn.execute(
        "INSERT OR REPLACE INTO scam_blacklist (account_number, bank_code, scam_type) VALUES ('0000000002', 'VCB', 'GIA_DANH_SHIPPER')"
    )
    conn.execute(
        "INSERT OR REPLACE INTO scam_blacklist (account_number, bank_code, scam_type) VALUES ('0000000003', 'TCB', 'RUA_TIEN')"
    )

    # --- Lịch sử giao dịch mẫu ---
    # Phải INSERT draft_transactions (status='COMPLETED') TRƯỚC transactions (FK)
    conn.execute(
        """
        INSERT INTO draft_transactions
            (tx_id, intent_id, sender_id, contact_id, recipient_name, recipient_account,
             bank_code, amount, source_type, decision, status)
        VALUES ('tx_seed_001', 'intent_seed_001', ?, 'C_001',
                'Nguyễn Văn Nam', '9001234567', 'MB', 200000,
                'CHAT_CONTACT', 'REQUIRE_HITL', 'COMPLETED')
        """,
        (user_id,),
    )
    conn.execute(
        """
        INSERT INTO transactions
            (tx_id, sender_id, contact_id, recipient_name, recipient_account, bank_code, amount)
        VALUES ('tx_seed_001', ?, 'C_001', 'Nguyễn Văn Nam', '9001234567', 'MB', 200000)
        """,
        (user_id,),
    )


def seed_demo(conn: sqlite3.Connection, user_id: str) -> None:
    """
    Seed dữ liệu demo vào DB trong một transaction.

    Dữ liệu:
    - 1 user (user_id) balance=10_000_000, pin='123456' (băm argon2)
    - 4 contacts: Nam, Hùng, Minh1, Minh2
    - Aliases theo tutorial mục 5
    - >= 3 STK trong scam_blacklist
    - 1 lịch sử giao dịch mẫu (COMPLETED): INSERT draft trước, rồi INSERT transaction
    """
    with transaction(conn):
        _insert_demo_data(conn, user_id)


def reset_demo(conn: sqlite3.Connection, user_id: str) -> None:
    """
    Xóa toàn bộ dữ liệu demo của user_id rồi seed lại trong CÙNG MỘT transaction.
    Thứ tự xóa: transactions -> draft_transactions -> contacts -> users (do FK).
    Không DELETE scam_blacklist; dùng INSERT OR REPLACE khi seed blacklist.
    """
    with transaction(conn):
        conn.execute("DELETE FROM transactions WHERE sender_id = ?", (user_id,))
        conn.execute("DELETE FROM draft_transactions WHERE sender_id = ?", (user_id,))
        conn.execute("DELETE FROM contacts WHERE owner_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        _insert_demo_data(conn, user_id)
