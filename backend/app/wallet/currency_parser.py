"""
currency_parser.py — Deterministic Currency Parser (Lớp bảo vệ cứng #2).

Mục tiêu:
- Phân tích số tiền tiếng Việt hoàn toàn xác định (deterministic), KHÔNG để LLM tự đoán.
- Bắt các số mơ hồ không kèm đơn vị (VD: "500", "200" trong "chuyển 500 cho Nam")
  và ép trạng thái CLARIFY kèm lựa chọn rõ ràng ("500.000 VNĐ" vs "500 VNĐ").
- Từ chối số tiền <= 0.
- Bỏ qua các placeholder đã được Privacy Proxy mask (<ACCT_1>, <PHONE_1>, <CCCD_1>).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Literal

CurrencyStatus = Literal["OK", "AMBIGUOUS", "INVALID", "NOT_FOUND"]

# Ngưỡng số nguyên trần không có đơn vị được xem là mơ hồ đơn vị VNĐ vs nghìn VNĐ
# VD: "chuyển 500 cho Nam" (500 VNĐ hay 500.000 VNĐ?), "chuyển 200", "chuyển 1000"
AMBIGUOUS_BARE_MAX = 9_999

_PLACEHOLDER_RE = re.compile(r"<(?:ACCT|PHONE|CCCD)_\d+>", re.IGNORECASE)

# Số âm rõ ràng gắn với đơn vị hoặc đứng sau động từ chuyển tiền
_NEGATIVE_AMOUNT_RE = re.compile(
    r"(?<!\w)-\s*\d+(?:[.,]\d+)?\s*(?:tr(?:iệu|ieu)?|k|ngh[iì]n|ng[aà]n|vn[dđ]|đ|d)?\b",
    re.IGNORECASE,
)

# Dạng viết tắt triệu kiểu Việt Nam: "1tr5" -> 1_500_000, "2tr25" -> 2_250_000
_MILLION_SHORTHAND_RE = re.compile(
    r"(?<!\d)(\d+)\s*(?:tr|triệu|trieu)\s*(\d{1,3})\b",
    re.IGNORECASE,
)

# Triệu chuẩn: "1.5tr", "1,5 triệu", "10 tr", "20 triệu"
_MILLION_RE = re.compile(
    r"(?<!\d)(\d+(?:[.,]\d+)?)\s*(?:tr|triệu|trieu)\b",
    re.IGNORECASE,
)

# Nghìn chuẩn: "500k", "200 nghìn", "50 ngàn", "1.5k"
_THOUSAND_RE = re.compile(
    r"(?<!\d)(\d+(?:[.,]\d+)?)\s*(?:k|nghìn|nghin|ngàn|ngan)\b",
    re.IGNORECASE,
)

# Đơn vị VNĐ tường minh: "500.000đ", "500,000 vnđ", "500000 vnd", "500đ", "0đ"
_EXPLICIT_VND_RE = re.compile(
    r"(?<!\d)(\d{1,3}(?:[.,]\d{3})+|\d+)\s*(?:vnđ|vnd|đồng|dong|đ|d)\b",
    re.IGNORECASE,
)

# Số có dấu phân cách hàng nghìn rõ ràng (không cần hậu tố): "500.000", "10.000.000", "1,500,000"
_FORMATTED_THOUSANDS_RE = re.compile(
    r"(?<!\d)(\d{1,3}(?:\.\d{3})+|\d{1,3}(?:,\d{3})+)(?!\d)"
)

# Số nguyên trần không đơn vị (bỏ qua số 8+ chữ số vì có thể là STK chưa mask)
_BARE_INT_RE = re.compile(r"(?<!\d)(\d{1,7})(?!\d)")


@dataclass(frozen=True)
class CurrencyClarifyOption:
    label: str
    amount: int


@dataclass(frozen=True)
class CurrencyParseResult:
    status: CurrencyStatus
    amount: int | None = None
    raw_token: str | None = None
    reason: str | None = None
    options: list[CurrencyClarifyOption] = field(default_factory=list)


def format_vnd(amount: int) -> str:
    """Định dạng số nguyên VNĐ theo kiểu Việt Nam: 500000 -> '500.000 VNĐ'."""
    return f"{amount:,}".replace(",", ".") + " VNĐ"


def _parse_decimal(num_str: str) -> float:
    return float(num_str.replace(",", "."))


def parse_currency(text: str) -> CurrencyParseResult:
    """
    Phân tích số tiền từ chuỗi văn bản (nên đưa chuỗi đã mask PII vào).

    Trả về:
    - status='OK' với amount (int > 0) nếu đơn vị rõ ràng.
    - status='AMBIGUOUS' với 2 phương án ClarifyOptions nếu gặp số trần mơ hồ (VD: '500').
    - status='INVALID' nếu số tiền <= 0.
    - status='NOT_FOUND' nếu không tìm thấy số tiền.
    """
    cleaned = _PLACEHOLDER_RE.sub(" ", text or "")

    # 1. Chặn số âm tường minh
    neg_match = _NEGATIVE_AMOUNT_RE.search(cleaned)
    if neg_match:
        return CurrencyParseResult(
            status="INVALID",
            raw_token=neg_match.group(0).strip(),
            reason="Số tiền giao dịch phải lớn hơn 0 VNĐ.",
        )

    # 2. Dạng triệu viết tắt kiểu Việt Nam: "1tr5" -> 1_500_000
    m = _MILLION_SHORTHAND_RE.search(cleaned)
    if m:
        whole = int(m.group(1))
        frac_str = m.group(2)
        frac_value = int(frac_str) * (10 ** (6 - len(frac_str)))
        amount = whole * 1_000_000 + frac_value
        if amount <= 0:
            return CurrencyParseResult(
                status="INVALID",
                raw_token=m.group(0),
                reason="Số tiền giao dịch phải lớn hơn 0 VNĐ.",
            )
        return CurrencyParseResult(status="OK", amount=amount, raw_token=m.group(0))

    # 3. Dạng triệu chuẩn: "10tr", "1.5 triệu"
    m = _MILLION_RE.search(cleaned)
    if m:
        amount = round(_parse_decimal(m.group(1)) * 1_000_000)
        if amount <= 0:
            return CurrencyParseResult(
                status="INVALID",
                raw_token=m.group(0),
                reason="Số tiền giao dịch phải lớn hơn 0 VNĐ.",
            )
        return CurrencyParseResult(status="OK", amount=amount, raw_token=m.group(0))

    # 4. Dạng nghìn: "500k", "200 nghìn"
    m = _THOUSAND_RE.search(cleaned)
    if m:
        amount = round(_parse_decimal(m.group(1)) * 1_000)
        if amount <= 0:
            return CurrencyParseResult(
                status="INVALID",
                raw_token=m.group(0),
                reason="Số tiền giao dịch phải lớn hơn 0 VNĐ.",
            )
        return CurrencyParseResult(status="OK", amount=amount, raw_token=m.group(0))

    # 5. Đơn vị VNĐ tường minh: "500.000đ", "500000 vnd", "500đ", "0đ"
    m = _EXPLICIT_VND_RE.search(cleaned)
    if m:
        digits = m.group(1).replace(".", "").replace(",", "")
        amount = int(digits)
        if amount <= 0:
            return CurrencyParseResult(
                status="INVALID",
                raw_token=m.group(0),
                reason="Số tiền giao dịch phải lớn hơn 0 VNĐ.",
            )
        return CurrencyParseResult(status="OK", amount=amount, raw_token=m.group(0))

    # 6. Số định dạng phân cách hàng nghìn: "500.000", "10.000.000"
    m = _FORMATTED_THOUSANDS_RE.search(cleaned)
    if m:
        digits = m.group(1).replace(".", "").replace(",", "")
        amount = int(digits)
        if amount <= 0:
            return CurrencyParseResult(
                status="INVALID",
                raw_token=m.group(0),
                reason="Số tiền giao dịch phải lớn hơn 0 VNĐ.",
            )
        return CurrencyParseResult(status="OK", amount=amount, raw_token=m.group(0))

    # 7. Số nguyên trần không kèm đơn vị
    m = _BARE_INT_RE.search(cleaned)
    if m:
        raw = m.group(1)
        val = int(raw)
        if val <= 0:
            return CurrencyParseResult(
                status="INVALID",
                raw_token=raw,
                reason="Số tiền giao dịch phải lớn hơn 0 VNĐ.",
            )
        if val <= AMBIGUOUS_BARE_MAX:
            return CurrencyParseResult(
                status="AMBIGUOUS",
                raw_token=raw,
                reason=(
                    f"Số tiền '{raw}' không có đơn vị rõ ràng. "
                    "Vui lòng xác nhận đơn vị tiền tệ."
                ),
                options=[
                    CurrencyClarifyOption(label=format_vnd(val * 1_000), amount=val * 1_000),
                    CurrencyClarifyOption(label=format_vnd(val), amount=val),
                ],
            )
        # Số nguyên >= 10.000 (VD: 500000) -> VNĐ tường minh
        return CurrencyParseResult(status="OK", amount=val, raw_token=raw)

    return CurrencyParseResult(
        status="NOT_FOUND",
        reason="Không tìm thấy số tiền trong yêu cầu.",
    )
