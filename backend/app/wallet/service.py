"""
service.py — Sandbox Wallet Service (API ví giả lập) tích hợp:
(1) Idempotency Key (`intent_id` UNIQUE + idempotent confirm)
(2) Deterministic Currency Parser
(3) Hard Veto Blacklist
(4) OCR-Always-HITL
(5) Ngưỡng sinh trắc học/PIN NHNN (>=10tr/lần, >=20tr/ngày)
(6) Privacy Proxy (RAM TTL) + Zero-PII lookup_contact
(7) Chống Memory Poisoning (NO_OCR_ALIAS, NO_SILENT_OVERWRITE + Post-PIN Commit)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import sqlite3
import time
from typing import Callable
from uuid import uuid4

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from app.db.connection import transaction
from app.db.repo.aliases import upsert_alias
from app.db.seed import reset_demo
from app.wallet.currency_parser import format_vnd, parse_currency
from app.wallet.guardrails import (
    SourceType,
    check_scam_blacklist,
    commit_alias_post_pin,
    enforce_no_ocr_alias,
    evaluate_hard_guardrails,
)
from app.wallet.privacy_proxy import DEFAULT_TTL_SECONDS, PrivacyProxy, lookup_contact
from app.wallet.schemas import (
    CancelResponse,
    ChatRequest,
    ChatResponse,
    ClarifyOptionItem,
    ClarifyOptions,
    ConfirmResponse,
    ContactView,
    DraftView,
    ErrorCode,
    ResetResponse,
    TransactionView,
    WalletState,
)

_ph = PasswordHasher()

_KNOWN_BANKS = ("VCB", "TCB", "ACB", "BIDV", "VTB", "TPB", "VPB", "STB", "VIB", "MB", "MOMO")
_BANK_RE = re.compile(r"\b(" + "|".join(_KNOWN_BANKS) + r")\b", re.IGNORECASE)
_RECIPIENT_PHRASE_RE = re.compile(
    r"(?i)(?:cho|tới|toi|đến|den|trả|tra|gửi|gui)\s+([^\d,<]+?)(?=\s*(?:stk|số\s*tài\s*khoản|tài\s*khoản|<|ngân\s*hàng|bank|\b(?:"
    + "|".join(_KNOWN_BANKS)
    + r")\b|$))"
)


class WalletError(Exception):
    """Lỗi nghiệp vụ có mã HTTP & error_code chuẩn theo OpenAPI."""

    def __init__(self, status_code: int, error_code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message


@dataclass
class _PendingClarify:
    user_id: str
    masked_message: str
    source_type: SourceType
    amount: int | None
    recipient_query: str | None
    acct_token: str | None
    bank_code: str
    suggested_alias: str | None
    session_id: str
    expires_at: float


class ClarifyContextStore:
    """
    Lưu ngữ cảnh CLARIFY tạm trong RAM (TTL mặc định 5 phút).
    Theo tutorial mục 3: CLARIFY không ghi DB.
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._items: dict[str, _PendingClarify] = {}

    def _purge(self) -> None:
        now = self._clock()
        expired = [k for k, v in self._items.items() if v.expires_at <= now]
        for k in expired:
            del self._items[k]

    def put(self, intent_id: str, ctx: _PendingClarify) -> None:
        self._purge()
        ctx.expires_at = self._clock() + self._ttl_seconds
        self._items[intent_id] = ctx

    def get(self, intent_id: str) -> _PendingClarify | None:
        self._purge()
        return self._items.get(intent_id)

    def pop(self, intent_id: str) -> _PendingClarify | None:
        self._purge()
        return self._items.pop(intent_id, None)


def _row_to_draft_view(row: sqlite3.Row) -> DraftView:
    return DraftView(
        tx_id=row["tx_id"],
        recipient_name=row["recipient_name"],
        recipient_account=row["recipient_account"],
        bank_code=row["bank_code"],
        amount=int(row["amount"]),
        note=row["note"],
        source_type=row["source_type"],
        suggested_alias=row["suggested_alias"],
        status=row["status"],
    )


def _parse_xai_reasons(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        return [str(x) for x in parsed] if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def _extract_bank_code(text: str, default: str = "MB") -> str:
    m = _BANK_RE.search(text)
    return m.group(1).upper() if m else default


def _extract_recipient_query(masked_text: str) -> str | None:
    m = _RECIPIENT_PHRASE_RE.search(masked_text)
    if not m:
        return None
    phrase = m.group(1).strip(" ,.;:-")
    # Loại bỏ từ nối thừa ở cuối như "tiền", "nhé", "nha", "đi"
    phrase = re.sub(r"(?i)\s+(?:nhé|nhe|nha|đi|di|giúp|giup|với|voi)$", "", phrase).strip()
    return phrase if phrase else None


def _build_response_from_existing_draft(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    sender_id: str,
) -> ChatResponse:
    if row["sender_id"] != sender_id:
        raise WalletError(404, "NOT_FOUND", "Không tìm thấy bản nháp cho người dùng này.")

    draft_view = _row_to_draft_view(row)
    reasons = _parse_xai_reasons(row["xai_reasons"])
    new_balance: int | None = None
    if row["decision"] == "PASS_AUTO" and row["status"] == "COMPLETED":
        u = conn.execute(
            "SELECT balance FROM users WHERE user_id = ?", (sender_id,)
        ).fetchone()
        if u is not None:
            new_balance = int(u["balance"])

    reply_map = {
        "PASS_AUTO": f"Giao dịch {format_vnd(draft_view.amount)} đã tự động hoàn tất.",
        "REQUIRE_HITL": "Vui lòng kiểm tra thẻ xác nhận giao dịch và nhập mã PIN.",
        "BLOCK_ANOMALY": "Giao dịch đã bị chặn bởi hệ thống bảo vệ an toàn.",
        "CLARIFY": "Cần thêm thông tin để làm rõ giao dịch.",
    }
    return ChatResponse(
        reply_text=reply_map.get(row["decision"], ""),
        decision=row["decision"],
        risk_score=float(row["risk_score"]),
        xai_reasons=reasons,
        draft=draft_view,
        clarify_options=None,
        new_balance=new_balance,
    )


def _update_contact_stats(conn: sqlite3.Connection, contact_id: str, amount: int) -> None:
    c_row = conn.execute(
        "SELECT transfer_count, avg_amount FROM contacts WHERE contact_id = ?",
        (contact_id,),
    ).fetchone()
    if c_row is None:
        return
    old_count = int(c_row["transfer_count"])
    old_avg = int(c_row["avg_amount"])
    new_count = old_count + 1
    new_avg = (old_avg * old_count + amount) // new_count
    conn.execute(
        """
        UPDATE contacts
        SET transfer_count = ?, avg_amount = ?, last_transfer_at = CURRENT_TIMESTAMP
        WHERE contact_id = ?
        """,
        (new_count, new_avg, contact_id),
    )


def create_draft_transfer(
    conn: sqlite3.Connection,
    *,
    intent_id: str,
    sender_id: str,
    amount: int,
    source_type: SourceType,
    bank_code: str = "MB",
    recipient_account: str | None = None,
    contact_id: str | None = None,
    recipient_name: str | None = None,
    note: str | None = None,
    suggested_alias: str | None = None,
    proxy: PrivacyProxy | None = None,
    session_id: str | None = None,
) -> ChatResponse:
    """
    Tạo bản nháp giao dịch và chạy toàn bộ lớp bảo vệ cứng:
    - (1) Idempotency Key (`intent_id` UNIQUE): gửi lại cùng intent_id -> trả draft cũ, không trừ tiền lần 2.
    - De-mask token (<ACCT_1>) cục bộ trong RAM nếu truyền qua Privacy Proxy.
    - (3) Hard Veto Blacklist -> BLOCK_ANOMALY (status='BLOCKED').
    - (4) OCR_ALWAYS_HITL -> nguồn OCR không bao giờ PASS_AUTO.
    - (5) Ngưỡng NHNN (>=10tr/lần, >=20tr/ngày) -> bắt buộc REQUIRE_HITL.
    - NO_OCR_ALIAS -> ép suggested_alias=None khi source_type='OCR_RECEIPT'.
    - Nếu PASS_AUTO: thực thi ngay trong 1 transaction SQLite (không commit alias).
    """
    # (1) Idempotency check trước khi làm bất kỳ thao tác ghi nào
    existing = conn.execute(
        "SELECT * FROM draft_transactions WHERE intent_id = ?",
        (intent_id,),
    ).fetchone()
    if existing is not None:
        return _build_response_from_existing_draft(conn, existing, sender_id)

    if amount <= 0:
        raise WalletError(422, "VALIDATION_ERROR", "Số tiền giao dịch phải lớn hơn 0 VNĐ.")

    user = conn.execute(
        "SELECT * FROM users WHERE user_id = ?", (sender_id,)
    ).fetchone()
    if user is None:
        raise WalletError(404, "NOT_FOUND", f"Không tìm thấy người dùng '{sender_id}'.")

    # De-mask STK cục bộ trong RAM nếu là token ẩn danh từ Privacy Proxy
    resolved_acct = recipient_account
    if resolved_acct and proxy is not None and session_id is not None:
        resolved_acct = proxy.demask(session_id, resolved_acct)

    resolved_contact_id = contact_id
    resolved_name = recipient_name
    resolved_bank = bank_code.upper()
    trust_level: str | None = None

    if resolved_contact_id is not None:
        contact = conn.execute(
            "SELECT * FROM contacts WHERE contact_id = ? AND owner_id = ?",
            (resolved_contact_id, sender_id),
        ).fetchone()
        if contact is None:
            raise WalletError(404, "NOT_FOUND", f"Không tìm thấy danh bạ '{resolved_contact_id}'.")
        resolved_acct = contact["account_number"]
        resolved_name = contact["full_name"]
        resolved_bank = contact["bank_code"]
        trust_level = contact["trust_level"]
    else:
        if not resolved_acct or resolved_acct.startswith("<"):
            raise WalletError(
                422,
                "VALIDATION_ERROR",
                "Thiếu số tài khoản người nhận hoặc token Privacy Proxy đã hết hạn.",
            )
        # Tìm xem STK thô có sẵn trong danh bạ hay chưa
        matched_contact = conn.execute(
            """
            SELECT * FROM contacts
            WHERE owner_id = ? AND account_number = ? AND UPPER(bank_code) = UPPER(?)
            """,
            (sender_id, resolved_acct, resolved_bank),
        ).fetchone()
        if matched_contact is not None:
            resolved_contact_id = matched_contact["contact_id"]
            resolved_name = resolved_name or matched_contact["full_name"]
            resolved_bank = matched_contact["bank_code"]
            trust_level = matched_contact["trust_level"]
        else:
            resolved_name = resolved_name or f"Tài khoản {resolved_acct}"
            trust_level = "NEW"

    # Luật cứng NO_OCR_ALIAS ở tầng ứng dụng (lớp DB CHECK là hàng rào thứ hai)
    safe_alias = enforce_no_ocr_alias(source_type, suggested_alias)

    # Đánh giá lớp bảo vệ cứng (Blacklist Veto -> Balance -> OCR_ALWAYS_HITL -> NHNN -> Auto-approve)
    eval_res = evaluate_hard_guardrails(
        conn,
        balance=int(user["balance"]),
        daily_transferred=int(user["daily_transferred"]),
        auto_approve_limit=int(user["auto_approve_limit"]),
        amount=amount,
        recipient_account=resolved_acct,
        bank_code=resolved_bank,
        source_type=source_type,
        trust_level=trust_level,
    )

    if eval_res.insufficient_balance:
        raise WalletError(
            422,
            "INSUFFICIENT_BALANCE",
            "Số dư ví không đủ để thực hiện giao dịch.",
        )

    tx_id = f"tx_{uuid4().hex[:12]}"
    if eval_res.decision == "BLOCK_ANOMALY":
        initial_status = "BLOCKED"
    elif eval_res.decision == "PASS_AUTO":
        initial_status = "COMPLETED"
    else:
        initial_status = "DRAFT"

    xai_json = json.dumps(eval_res.xai_reasons, ensure_ascii=False)
    new_balance: int | None = None

    try:
        with transaction(conn):
            conn.execute(
                """
                INSERT INTO draft_transactions
                    (tx_id, intent_id, sender_id, contact_id, recipient_name, recipient_account,
                     bank_code, amount, note, source_type, suggested_alias, decision,
                     risk_score, xai_reasons, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tx_id,
                    intent_id,
                    sender_id,
                    resolved_contact_id,
                    resolved_name,
                    resolved_acct,
                    resolved_bank,
                    amount,
                    note,
                    source_type,
                    safe_alias,
                    eval_res.decision,
                    eval_res.risk_score,
                    xai_json,
                    initial_status,
                ),
            )

            if eval_res.decision == "PASS_AUTO":
                cur = conn.execute(
                    """
                    UPDATE users
                    SET balance = balance - ?, daily_transferred = daily_transferred + ?
                    WHERE user_id = ? AND balance >= ?
                    """,
                    (amount, amount, sender_id, amount),
                )
                if cur.rowcount == 0:
                    raise WalletError(
                        422,
                        "INSUFFICIENT_BALANCE",
                        "Số dư ví không đủ để thực hiện giao dịch.",
                    )

                conn.execute(
                    """
                    INSERT INTO transactions
                        (tx_id, sender_id, contact_id, recipient_name, recipient_account,
                         bank_code, amount, note)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tx_id,
                        sender_id,
                        resolved_contact_id,
                        resolved_name,
                        resolved_acct,
                        resolved_bank,
                        amount,
                        note,
                    ),
                )
                if resolved_contact_id is not None:
                    _update_contact_stats(conn, resolved_contact_id, amount)

                bal_row = conn.execute(
                    "SELECT balance FROM users WHERE user_id = ?", (sender_id,)
                ).fetchone()
                new_balance = int(bal_row["balance"])
    except sqlite3.IntegrityError as exc:
        # Xử lý race condition trùng intent_id (Idempotency Key)
        if "intent_id" in str(exc):
            existing_row = conn.execute(
                "SELECT * FROM draft_transactions WHERE intent_id = ?",
                (intent_id,),
            ).fetchone()
            if existing_row is not None:
                return _build_response_from_existing_draft(conn, existing_row, sender_id)
        raise

    created_row = conn.execute(
        "SELECT * FROM draft_transactions WHERE tx_id = ?", (tx_id,)
    ).fetchone()
    resp = _build_response_from_existing_draft(conn, created_row, sender_id)
    if new_balance is not None:
        resp.new_balance = new_balance
    return resp


def confirm_draft(
    conn: sqlite3.Connection,
    *,
    tx_id: str,
    user_id: str,
    pin_code: str,
    save_alias: bool = False,
) -> ConfirmResponse:
    """
    Xác nhận bản nháp giao dịch bằng mã PIN (Idempotent, 1 transaction SQLite):
    - Trả về kết quả cũ nếu draft đã COMPLETED (không trừ tiền lần 2).
    - Kiểm tra PIN bằng argon2 -> sai PIN trả 401 INVALID_PIN (draft giữ nguyên, không lưu alias).
    - Kiểm tra lại scam_blacklist lúc confirm -> nếu dính trả 409 BLOCKED_BY_BLACKLIST.
    - Trừ balance, cộng daily_transferred, ghi sổ cái `transactions`, đổi draft sang COMPLETED.
    - Post-PIN Commit: chỉ commit `suggested_alias` sau khi PIN OK + save_alias=True + không phải OCR.
    """
    draft = conn.execute(
        "SELECT * FROM draft_transactions WHERE tx_id = ?", (tx_id,)
    ).fetchone()
    if draft is None or draft["sender_id"] != user_id:
        raise WalletError(404, "NOT_FOUND", "Không tìm thấy bản nháp giao dịch.")

    user = conn.execute(
        "SELECT * FROM users WHERE user_id = ?", (user_id,)
    ).fetchone()
    if user is None:
        raise WalletError(404, "NOT_FOUND", "Không tìm thấy người dùng.")

    # Idempotency: confirm lần 2 trên draft đã COMPLETED -> trả kết quả cũ, không trừ tiền nữa
    if draft["status"] == "COMPLETED":
        return ConfirmResponse(
            status="COMPLETED",
            new_balance=int(user["balance"]),
            tx_id=tx_id,
            alias_saved=None,
        )

    if draft["status"] in ("CANCELLED", "BLOCKED"):
        raise WalletError(
            409,
            "DRAFT_NOT_CONFIRMABLE",
            f"Bản nháp ở trạng thái '{draft['status']}' không thể xác nhận.",
        )

    # Xác thực PIN bằng argon2
    try:
        _ph.verify(user["pin_hash"], pin_code)
    except (VerifyMismatchError, InvalidHashError, VerificationError):
        raise WalletError(401, "INVALID_PIN", "Mã PIN không chính xác.")

    # Kiểm tra lại blacklist tại thời điểm confirm (phòng trường hợp STK mới bị đưa vào blacklist)
    bl = check_scam_blacklist(conn, draft["recipient_account"], draft["bank_code"])
    if bl.is_blacklisted:
        with transaction(conn):
            conn.execute(
                "UPDATE draft_transactions SET status = 'BLOCKED' WHERE tx_id = ? AND status = 'DRAFT'",
                (tx_id,),
            )
        raise WalletError(
            409,
            "BLOCKED_BY_BLACKLIST",
            f"Tài khoản nằm trong danh sách đen lừa đảo ({bl.scam_type}).",
        )

    amount = int(draft["amount"])
    alias_saved: str | None = None

    with transaction(conn):
        # Đọc lại status trong transaction để chống race condition confirm song song
        fresh = conn.execute(
            "SELECT status FROM draft_transactions WHERE tx_id = ?", (tx_id,)
        ).fetchone()
        if fresh["status"] == "COMPLETED":
            bal_row = conn.execute(
                "SELECT balance FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
            return ConfirmResponse(
                status="COMPLETED",
                new_balance=int(bal_row["balance"]),
                tx_id=tx_id,
                alias_saved=None,
            )
        if fresh["status"] != "DRAFT":
            raise WalletError(
                409,
                "DRAFT_NOT_CONFIRMABLE",
                f"Bản nháp ở trạng thái '{fresh['status']}' không thể xác nhận.",
            )

        cur = conn.execute(
            """
            UPDATE users
            SET balance = balance - ?, daily_transferred = daily_transferred + ?
            WHERE user_id = ? AND balance >= ?
            """,
            (amount, amount, user_id, amount),
        )
        if cur.rowcount == 0:
            raise WalletError(
                422,
                "INSUFFICIENT_BALANCE",
                "Số dư ví không đủ để thực hiện giao dịch.",
            )

        conn.execute(
            "UPDATE draft_transactions SET status = 'COMPLETED' WHERE tx_id = ?",
            (tx_id,),
        )

        conn.execute(
            """
            INSERT INTO transactions
                (tx_id, sender_id, contact_id, recipient_name, recipient_account,
                 bank_code, amount, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tx_id,
                user_id,
                draft["contact_id"],
                draft["recipient_name"],
                draft["recipient_account"],
                draft["bank_code"],
                amount,
                draft["note"],
            ),
        )

        contact_id = draft["contact_id"]
        # Nếu STK thô và cần lưu alias sau PIN -> tìm theo (owner_id, account_number, bank_code) trước khi INSERT
        if (
            contact_id is None
            and save_alias
            and draft["suggested_alias"]
            and draft["source_type"] != "OCR_RECEIPT"
        ):
            found = conn.execute(
                """
                SELECT contact_id FROM contacts
                WHERE owner_id = ? AND account_number = ? AND UPPER(bank_code) = UPPER(?)
                """,
                (user_id, draft["recipient_account"], draft["bank_code"]),
            ).fetchone()
            if found is not None:
                contact_id = found["contact_id"]
            else:
                contact_id = f"C_{uuid4().hex[:8]}"
                conn.execute(
                    """
                    INSERT INTO contacts
                        (contact_id, owner_id, full_name, account_number, bank_code, trust_level)
                    VALUES (?, ?, ?, ?, ?, 'NEW')
                    """,
                    (
                        contact_id,
                        user_id,
                        draft["recipient_name"],
                        draft["recipient_account"],
                        draft["bank_code"],
                    ),
                )

        if contact_id is not None:
            _update_contact_stats(conn, contact_id, amount)

        # Post-PIN Commit (NO_OCR_ALIAS + NO_SILENT_OVERWRITE)
        alias_saved = commit_alias_post_pin(
            conn,
            contact_id=contact_id,
            suggested_alias=draft["suggested_alias"],
            source_type=draft["source_type"],
            save_alias=save_alias,
            pin_verified=True,
        )

        bal_row = conn.execute(
            "SELECT balance FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        new_balance = int(bal_row["balance"])

    return ConfirmResponse(
        status="COMPLETED",
        new_balance=new_balance,
        tx_id=tx_id,
        alias_saved=alias_saved,
    )


def cancel_draft(
    conn: sqlite3.Connection,
    *,
    tx_id: str,
    user_id: str,
) -> CancelResponse:
    """Hủy bản nháp giao dịch đang ở trạng thái DRAFT."""
    draft = conn.execute(
        "SELECT * FROM draft_transactions WHERE tx_id = ?", (tx_id,)
    ).fetchone()
    if draft is None or draft["sender_id"] != user_id:
        raise WalletError(404, "NOT_FOUND", "Không tìm thấy bản nháp giao dịch.")

    if draft["status"] == "CANCELLED":
        return CancelResponse(status="CANCELLED", tx_id=tx_id)

    if draft["status"] != "DRAFT":
        raise WalletError(
            409,
            "DRAFT_NOT_CONFIRMABLE",
            f"Không thể hủy bản nháp ở trạng thái '{draft['status']}'.",
        )

    with transaction(conn):
        conn.execute(
            "UPDATE draft_transactions SET status = 'CANCELLED' WHERE tx_id = ?",
            (tx_id,),
        )

    return CancelResponse(status="CANCELLED", tx_id=tx_id)


def get_wallet_state(conn: sqlite3.Connection, user_id: str) -> WalletState:
    """Lấy trạng thái ví đầy đủ cho UI (không dành cho LLM)."""
    user = conn.execute(
        "SELECT * FROM users WHERE user_id = ?", (user_id,)
    ).fetchone()
    if user is None:
        raise WalletError(404, "NOT_FOUND", f"Không tìm thấy người dùng '{user_id}'.")

    contact_rows = conn.execute(
        "SELECT * FROM contacts WHERE owner_id = ? ORDER BY contact_id",
        (user_id,),
    ).fetchall()

    contacts: list[ContactView] = []
    for c in contact_rows:
        alias_rows = conn.execute(
            """
            SELECT alias_raw FROM contact_aliases
            WHERE contact_id = ?
            ORDER BY usage_count DESC, id ASC
            """,
            (c["contact_id"],),
        ).fetchall()
        contacts.append(
            ContactView(
                contact_id=c["contact_id"],
                full_name=c["full_name"],
                bank_code=c["bank_code"],
                account_number=c["account_number"],
                trust_level=c["trust_level"],
                aliases=[r["alias_raw"] for r in alias_rows],
            )
        )

    tx_rows = conn.execute(
        """
        SELECT tx_id, recipient_name, bank_code, amount, note, completed_at
        FROM transactions
        WHERE sender_id = ?
        ORDER BY completed_at DESC, tx_id DESC
        LIMIT 20
        """,
        (user_id,),
    ).fetchall()

    recent_transactions = [
        TransactionView(
            tx_id=r["tx_id"],
            recipient_name=r["recipient_name"],
            bank_code=r["bank_code"],
            amount=int(r["amount"]),
            note=r["note"],
            completed_at=str(r["completed_at"]),
        )
        for r in tx_rows
    ]

    return WalletState(
        balance=int(user["balance"]),
        daily_transferred=int(user["daily_transferred"]),
        auto_approve_limit=int(user["auto_approve_limit"]),
        contacts=contacts,
        recent_transactions=recent_transactions,
    )


def reset_wallet_demo(conn: sqlite3.Connection, user_id: str) -> ResetResponse:
    """Reset dữ liệu demo của user về mặc định (balance 10.000.000 VNĐ)."""
    reset_demo(conn, user_id)
    user = conn.execute(
        "SELECT balance FROM users WHERE user_id = ?", (user_id,)
    ).fetchone()
    return ResetResponse(status="RESET_OK", balance=int(user["balance"]))


def handle_chat_message(
    conn: sqlite3.Connection,
    req: ChatRequest,
    *,
    proxy: PrivacyProxy,
    clarify_store: ClarifyContextStore,
) -> ChatResponse:
    """
    Xử lý POST /api/chat hoàn toàn xác định qua Privacy Proxy + Currency Parser +
    Zero-PII lookup_contact + Hard Guardrails + Idempotency Key.
    """
    intent_str = str(req.intent_id)

    # (1) Idempotency Key check
    existing = conn.execute(
        "SELECT * FROM draft_transactions WHERE intent_id = ?",
        (intent_str,),
    ).fetchone()
    if existing is not None:
        return _build_response_from_existing_draft(conn, existing, req.user_id)

    user = conn.execute(
        "SELECT * FROM users WHERE user_id = ?", (req.user_id,)
    ).fetchone()
    if user is None:
        raise WalletError(404, "NOT_FOUND", f"Không tìm thấy người dùng '{req.user_id}'.")

    prior_ctx: _PendingClarify | None = None
    if req.clarify_choice and req.clarify_choice.ref_intent_id is not None:
        prior_ctx = clarify_store.pop(str(req.clarify_choice.ref_intent_id))

    session_id = prior_ctx.session_id if prior_ctx else f"{req.user_id}:{intent_str}"
    mask_res = proxy.mask(session_id, req.message)
    masked_text = mask_res.masked_text

    # Tìm token STK trong tin nhắn hiện tại hoặc ngữ cảnh CLARIFY trước
    acct_token: str | None = next(
        (tok for tok, kind in mask_res.tokens.items() if kind == "ACCT"),
        prior_ctx.acct_token if prior_ctx else None,
    )

    if req.image_base64:
        source_type: SourceType = "OCR_RECEIPT"
    elif prior_ctx is not None:
        source_type = prior_ctx.source_type
    elif acct_token is not None:
        source_type = "CHAT_RAW_ACCT"
    else:
        source_type = "CHAT_CONTACT"

    bank_code = _extract_bank_code(
        masked_text, default=prior_ctx.bank_code if prior_ctx else "MB"
    )
    recipient_query = _extract_recipient_query(masked_text) or (
        prior_ctx.recipient_query if prior_ctx else None
    )
    suggested_alias = enforce_no_ocr_alias(
        source_type,
        recipient_query if acct_token is not None else (prior_ctx.suggested_alias if prior_ctx else None),
    )

    # (2) Deterministic Currency Parser
    amount: int | None = None
    if req.clarify_choice and req.clarify_choice.amount is not None:
        amount = req.clarify_choice.amount
    elif prior_ctx and prior_ctx.amount is not None:
        amount = prior_ctx.amount
    else:
        cur_res = parse_currency(masked_text)
        if cur_res.status == "INVALID":
            raise WalletError(422, "VALIDATION_ERROR", cur_res.reason or "Số tiền không hợp lệ.")
        if cur_res.status == "AMBIGUOUS":
            clarify_store.put(
                intent_str,
                _PendingClarify(
                    user_id=req.user_id,
                    masked_message=masked_text,
                    source_type=source_type,
                    amount=None,
                    recipient_query=recipient_query,
                    acct_token=acct_token,
                    bank_code=bank_code,
                    suggested_alias=suggested_alias,
                    session_id=session_id,
                    expires_at=0.0,
                ),
            )
            return ChatResponse(
                reply_text=cur_res.reason or "Vui lòng xác nhận đơn vị tiền tệ.",
                decision="CLARIFY",
                risk_score=0.0,
                xai_reasons=["AMBIGUOUS_CURRENCY_UNIT: Số tiền không kèm đơn vị rõ ràng."],
                draft=None,
                clarify_options=ClarifyOptions(
                    kind="CURRENCY_UNIT",
                    options=[
                        ClarifyOptionItem(label=opt.label, amount=opt.amount)
                        for opt in cur_res.options
                    ],
                ),
                new_balance=None,
            )
        if cur_res.status == "NOT_FOUND":
            raise WalletError(422, "VALIDATION_ERROR", "Không tìm thấy số tiền trong yêu cầu.")
        amount = cur_res.amount

    assert amount is not None

    # Phân giải người nhận
    contact_id: str | None = None
    recipient_account: str | None = None

    if req.clarify_choice and req.clarify_choice.contact_id:
        contact_id = req.clarify_choice.contact_id
        # Alias Expansion (tutorial mục 2): bổ sung cụm user vừa gọi (nếu không phải OCR)
        if source_type != "OCR_RECEIPT" and prior_ctx and prior_ctx.recipient_query:
            with transaction(conn):
                upsert_alias(conn, contact_id, prior_ctx.recipient_query, from_ocr=False)
    elif acct_token is not None:
        recipient_account = acct_token
    else:
        query_to_use = recipient_query or masked_text
        lookup_res = lookup_contact(conn, req.user_id, query_to_use)
        matches = lookup_res["matches"]
        if len(matches) == 0:
            return ChatResponse(
                reply_text="Không tìm thấy người nhận phù hợp trong danh bạ.",
                decision="CLARIFY",
                risk_score=0.0,
                xai_reasons=["CONTACT_NOT_FOUND: Không khớp danh bạ nào."],
                draft=None,
                clarify_options=None,
                new_balance=None,
            )
        if len(matches) >= 2:
            clarify_store.put(
                intent_str,
                _PendingClarify(
                    user_id=req.user_id,
                    masked_message=masked_text,
                    source_type=source_type,
                    amount=amount,
                    recipient_query=query_to_use,
                    acct_token=None,
                    bank_code=bank_code,
                    suggested_alias=None,
                    session_id=session_id,
                    expires_at=0.0,
                ),
            )
            options: list[ClarifyOptionItem] = []
            for m in matches:
                # Nhãn CLARIFY chỉ dùng alias + bank_code + masked_acct (không lộ STK đầy đủ)
                label = f"{m['alias']} — {m['bank_code']} {m['masked_acct']}"
                options.append(
                    ClarifyOptionItem(label=label, contact_id=m["contact_id"])
                )
            return ChatResponse(
                reply_text="Tìm thấy nhiều người nhận trùng tên. Vui lòng chọn đúng người.",
                decision="CLARIFY",
                risk_score=0.0,
                xai_reasons=["AMBIGUOUS_CONTACT: Có từ 2 danh bạ khớp từ khóa."],
                draft=None,
                clarify_options=ClarifyOptions(kind="CONTACT", options=options),
                new_balance=None,
            )
        contact_id = matches[0]["contact_id"]

    return create_draft_transfer(
        conn,
        intent_id=intent_str,
        sender_id=req.user_id,
        amount=amount,
        source_type=source_type,
        bank_code=bank_code,
        recipient_account=recipient_account,
        contact_id=contact_id,
        note=masked_text,
        suggested_alias=suggested_alias,
        proxy=proxy,
        session_id=session_id,
    )
