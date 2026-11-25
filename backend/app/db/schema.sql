PRAGMA foreign_keys = ON;   -- bật cho MỖI kết nối

CREATE TABLE IF NOT EXISTS users (
    user_id             TEXT PRIMARY KEY,
    full_name           TEXT NOT NULL,
    balance             INTEGER NOT NULL DEFAULT 10000000 CHECK (balance >= 0),
    auto_approve_limit  INTEGER NOT NULL DEFAULT 100000   CHECK (auto_approve_limit >= 0),
    daily_transferred   INTEGER NOT NULL DEFAULT 0        CHECK (daily_transferred >= 0),
    pin_hash            TEXT NOT NULL                     -- chỉ lưu hash
);

CREATE TABLE IF NOT EXISTS contacts (
    contact_id        TEXT PRIMARY KEY,                   -- 'C_042'
    owner_id          TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    full_name         TEXT NOT NULL,
    account_number    TEXT NOT NULL,
    bank_code         TEXT NOT NULL,
    trust_level       TEXT NOT NULL DEFAULT 'NEW' CHECK (trust_level IN ('NEW','TRUSTED')),
    transfer_count    INTEGER NOT NULL DEFAULT 0,
    avg_amount        INTEGER NOT NULL DEFAULT 0,
    last_transfer_at  TIMESTAMP
);

CREATE TABLE IF NOT EXISTS contact_aliases (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    contact_id   TEXT NOT NULL REFERENCES contacts(contact_id) ON DELETE CASCADE,
    alias_raw    TEXT NOT NULL,                           -- 'thằng Nam', 'sếp Hùng'
    alias_norm   TEXT NOT NULL,                           -- 'thang nam', 'sep hung'
    usage_count  INTEGER NOT NULL DEFAULT 1,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (contact_id, alias_norm)
);
CREATE INDEX IF NOT EXISTS idx_alias_norm ON contact_aliases(alias_norm);

CREATE TABLE IF NOT EXISTS draft_transactions (
    tx_id              TEXT PRIMARY KEY,
    intent_id          TEXT NOT NULL UNIQUE,              -- khóa idempotency (UUIDv4 từ client)
    sender_id          TEXT NOT NULL REFERENCES users(user_id),
    contact_id         TEXT REFERENCES contacts(contact_id),   -- NULL nếu STK thô / OCR
    recipient_name     TEXT NOT NULL,
    recipient_account  TEXT NOT NULL,                     -- STK thật, không bao giờ gửi cho LLM
    bank_code          TEXT NOT NULL,
    amount             INTEGER NOT NULL CHECK (amount > 0),
    note               TEXT,
    source_type        TEXT NOT NULL CHECK (source_type IN ('CHAT_CONTACT','CHAT_RAW_ACCT','OCR_RECEIPT')),
    suggested_alias    TEXT,                              -- lưu tạm, chỉ commit sau khi PIN OK
    decision           TEXT NOT NULL CHECK (decision IN ('PASS_AUTO','REQUIRE_HITL','CLARIFY','BLOCK_ANOMALY')),
    risk_score         REAL NOT NULL DEFAULT 0.0 CHECK (risk_score BETWEEN 0.0 AND 1.0),
    xai_reasons        TEXT NOT NULL DEFAULT '[]',        -- JSON array
    status             TEXT NOT NULL DEFAULT 'DRAFT' CHECK (status IN ('DRAFT','COMPLETED','CANCELLED','BLOCKED')),
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (NOT (source_type = 'OCR_RECEIPT' AND suggested_alias IS NOT NULL))   -- chống Memory Poisoning
);

CREATE TABLE IF NOT EXISTS transactions (                 -- sổ cái: chỉ giao dịch đã hoàn tất
    tx_id              TEXT PRIMARY KEY REFERENCES draft_transactions(tx_id),
    sender_id          TEXT NOT NULL REFERENCES users(user_id),
    contact_id         TEXT REFERENCES contacts(contact_id),
    recipient_name     TEXT NOT NULL,
    recipient_account  TEXT NOT NULL,
    bank_code          TEXT NOT NULL,
    amount             INTEGER NOT NULL CHECK (amount > 0),
    note               TEXT,
    completed_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scam_blacklist (
    account_number  TEXT NOT NULL,
    bank_code       TEXT NOT NULL,
    scam_type       TEXT NOT NULL,
    report_count    INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (account_number, bank_code)
);
