"""
guardrails.py — Lớp bảo vệ cứng (Hard Guardrails) & 2 luật chống Memory Poisoning.

Bao gồm:
(3) Hard Veto Blacklist: khớp scam_blacklist -> lập tức BLOCK_ANOMALY.
(4) Luật cứng OCR_ALWAYS_HITL: nguồn OCR_RECEIPT không bao giờ PASS_AUTO, tối thiểu REQUIRE_HITL.
(5) Ngưỡng sinh trắc học/PIN NHNN: > 10.000.000 VNĐ/lần HOẶC tổng trong ngày > 20.000.000 VNĐ -> REQUIRE_HITL.
(6) Chống Memory Poisoning:
    - NO_OCR_ALIAS: cấm học alias từ nguồn OCR_RECEIPT (ép suggested_alias=None, từ chối ghi DB).
    - NO_SILENT_OVERWRITE + Post-PIN Commit: không ghi đè ngầm alias trùng, chỉ commit vào contact_aliases
      sau khi xác nhận PIN thành công và user đồng ý (save_alias=True).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import sqlite3
from typing import Literal

from app.db.repo.aliases import upsert_alias

NHNN_SINGLE_TX_LIMIT = 10_000_000   # > 10 triệu VNĐ/lần bắt buộc HITL
NHNN_DAILY_TX_LIMIT = 20_000_000    # > 20 triệu VNĐ/ngày bắt buộc HITL

Decision = Literal["PASS_AUTO", "REQUIRE_HITL", "CLARIFY", "BLOCK_ANOMALY"]
SourceType = Literal["CHAT_CONTACT", "CHAT_RAW_ACCT", "OCR_RECEIPT"]


@dataclass(frozen=True)
class BlacklistCheckResult:
    is_blacklisted: bool
    scam_type: str | None = None
    report_count: int = 0


@dataclass(frozen=True)
class GuardrailEvaluation:
    decision: Decision
    risk_score: float
    xai_reasons: list[str] = field(default_factory=list)
    blocked_by_blacklist: bool = False
    insufficient_balance: bool = False


def check_scam_blacklist(
    conn: sqlite3.Connection,
    account_number: str,
    bank_code: str,
) -> BlacklistCheckResult:
    """Kiểm tra (account_number, bank_code) trong bảng scam_blacklist."""
    row = conn.execute(
        """
        SELECT scam_type, report_count
        FROM scam_blacklist
        WHERE account_number = ? AND UPPER(bank_code) = UPPER(?)
        """,
        (account_number, bank_code),
    ).fetchone()
    if row is None:
        return BlacklistCheckResult(is_blacklisted=False)
    return BlacklistCheckResult(
        is_blacklisted=True,
        scam_type=row["scam_type"],
        report_count=int(row["report_count"]),
    )


def enforce_no_ocr_alias(
    source_type: SourceType,
    suggested_alias: str | None,
) -> str | None:
    """
    Luật cứng NO_OCR_ALIAS:
    Nguồn OCR_RECEIPT (hoặc bên thứ ba) tuyệt đối không được mang theo suggested_alias.
    """
    if source_type == "OCR_RECEIPT":
        return None
    if suggested_alias is None:
        return None
    cleaned = suggested_alias.strip()
    return cleaned if cleaned else None


def commit_alias_post_pin(
    conn: sqlite3.Connection,
    *,
    contact_id: str | None,
    suggested_alias: str | None,
    source_type: SourceType,
    save_alias: bool,
    pin_verified: bool,
) -> str | None:
    """
    Post-PIN Commit + NO_OCR_ALIAS + NO_SILENT_OVERWRITE:
    - Chỉ commit vào `contact_aliases` khi:
      1. `pin_verified` là True (PASS_AUTO không PIN -> không lưu alias).
      2. `save_alias` là True.
      3. `source_type != 'OCR_RECEIPT'` (NO_OCR_ALIAS).
      4. `contact_id` và `suggested_alias` không rỗng.
    - Ghi thông qua `upsert_alias` (ON CONFLICT chỉ tăng usage_count, không ghi đè alias_raw cũ;
      cùng alias_norm khác contact thì thêm dòng mới để lần sau lookup trả >=2 -> CLARIFY).
    """
    if not pin_verified or not save_alias:
        return None
    if source_type == "OCR_RECEIPT":
        return None
    if not contact_id or not suggested_alias or not suggested_alias.strip():
        return None

    alias_clean = suggested_alias.strip()
    upsert_alias(conn, contact_id, alias_clean, from_ocr=False)
    return alias_clean


def save_contact_alias_checked(
    conn: sqlite3.Connection,
    contact_id: str,
    alias_raw: str,
    *,
    from_ocr: bool = False,
) -> None:
    """
    Wrapper trực tiếp bảo vệ 2 luật Memory Poisoning:
    - from_ocr=True -> raise PermissionError (NO_OCR_ALIAS).
    - Delegate tới upsert_alias để đảm bảo NO_SILENT_OVERWRITE.
    """
    upsert_alias(conn, contact_id, alias_raw, from_ocr=from_ocr)


def evaluate_hard_guardrails(
    conn: sqlite3.Connection,
    *,
    balance: int,
    daily_transferred: int,
    auto_approve_limit: int,
    amount: int,
    recipient_account: str,
    bank_code: str,
    source_type: SourceType,
    trust_level: str | None,
) -> GuardrailEvaluation:
    """
    Đánh giá các lớp bảo vệ cứng theo thứ tự ưu tiên tuyệt đối:
    1. Hard Veto Blacklist -> BLOCK_ANOMALY (phủ quyết mọi luật khác).
    2. Kiểm tra số dư (amount > balance -> đánh dấu insufficient_balance).
    3. Luật OCR_ALWAYS_HITL -> tối thiểu REQUIRE_HITL.
    4. Ngưỡng NHNN (> 10tr/lần hoặc daily + amount > 20tr/ngày) -> bắt buộc REQUIRE_HITL.
    5. Hạn mức ủy quyền tự động (<= auto_approve_limit & TRUSTED & non-OCR) -> PASS_AUTO,
       ngược lại -> REQUIRE_HITL.
    """
    if amount <= 0:
        raise ValueError("Số tiền giao dịch phải lớn hơn 0 VNĐ.")

    # (3) Hard Veto Blacklist — phủ quyết tuyệt đối
    bl = check_scam_blacklist(conn, recipient_account, bank_code)
    if bl.is_blacklisted:
        return GuardrailEvaluation(
            decision="BLOCK_ANOMALY",
            risk_score=1.0,
            xai_reasons=[
                f"BLACKLIST_HARD_VETO: Tài khoản nằm trong danh sách đen ({bl.scam_type})."
            ],
            blocked_by_blacklist=True,
        )

    # Kiểm tra số dư (CONTEXT.md 4.5: Chặn mọi giá trị <= 0 hoặc vượt số dư)
    if amount > balance:
        return GuardrailEvaluation(
            decision="BLOCK_ANOMALY",
            risk_score=0.0,
            xai_reasons=["INSUFFICIENT_BALANCE: Số dư ví không đủ để thực hiện giao dịch."],
            insufficient_balance=True,
        )

    reasons: list[str] = []
    must_hitl = False

    # (4) Luật cứng OCR_ALWAYS_HITL
    if source_type == "OCR_RECEIPT":
        must_hitl = True
        reasons.append(
            "OCR_ALWAYS_HITL: Giao dịch có nguồn từ ảnh/OCR bắt buộc xác thực PIN/sinh trắc học."
        )

    # (5) Ngưỡng sinh trắc học/PIN NHNN
    if amount > NHNN_SINGLE_TX_LIMIT:
        must_hitl = True
        reasons.append(
            "NHNN_SINGLE_TX_LIMIT: Giao dịch > 10.000.000 VNĐ/lần bắt buộc xác thực PIN/sinh trắc học."
        )

    if daily_transferred + amount > NHNN_DAILY_TX_LIMIT:
        must_hitl = True
        reasons.append(
            "NHNN_DAILY_TX_LIMIT: Tổng giao dịch trong ngày > 20.000.000 VNĐ bắt buộc xác thực PIN/sinh trắc học."
        )

    if must_hitl:
        return GuardrailEvaluation(
            decision="REQUIRE_HITL",
            risk_score=0.4,
            xai_reasons=reasons,
        )

    # Kiểm tra điều kiện PASS_AUTO: danh bạ TRUSTED + số tiền <= auto_approve_limit
    if trust_level == "TRUSTED" and amount <= auto_approve_limit:
        return GuardrailEvaluation(
            decision="PASS_AUTO",
            risk_score=0.05,
            xai_reasons=[],
        )

    if trust_level != "TRUSTED":
        reasons.append("RECIPIENT_NOT_TRUSTED: Người nhận chưa ở mức tin cậy TRUSTED, yêu cầu xác thực PIN.")
    if amount > auto_approve_limit:
        reasons.append("EXCEEDS_AUTO_APPROVE_LIMIT: Số tiền vượt hạn mức tự duyệt, yêu cầu xác thực PIN.")

    return GuardrailEvaluation(
        decision="REQUIRE_HITL",
        risk_score=0.25,
        xai_reasons=reasons,
    )
