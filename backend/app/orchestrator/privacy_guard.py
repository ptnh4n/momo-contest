"""
privacy_guard.py — Mở rộng PrivacyProxy cho T05 + Kiểm chứng 0% PII Outbound (E07).

Nhiệm vụ:
1. Mask cả STK/SĐT/CCCD có dấu cách, dấu chấm hoặc gạch ngang (E07) trước khi đưa vào PrivacyProxy.
2. Không mask nhầm số tiền VNĐ có dấu phân cách hàng nghìn (VD: `500.000`, `10.000.000`).
3. Kiểm chứng fail-closed (`assert_zero_pii_outbound`) 100% payload gửi ra Gemini không chứa PII thô.
4. Lưu ring-buffer OutboundAuditLog (chỉ lưu bản đã che PII) phục vụ kiểm thử & QA.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
import threading
from typing import Any

from app.wallet.privacy_proxy import MaskResult, PrivacyProxy

# Nhận diện chuỗi số tiền phân cách 3 chữ số (VD: 500.000, 10.000.000, 1,500,000) để KHÔNG mask nhầm thành STK
_CURRENCY_GROUPED_RE = re.compile(r"^\d{1,3}(?:[.,]\d{3})+$")

# Labeled PII cho phép có khoảng trắng, dấu chấm hoặc gạch ngang (E07)
_SEP_LABELED_ACCT_RE = re.compile(
    r"(?i)(\b(?:stk|số\s*tài\s*khoản|so\s*tai\s*khoan|tài\s*khoản|tai\s*khoan|số\s*tk|so\s*tk)\s*[:#-]?\s*)(\d[\d\s.\-]{6,24}\d)(?!\d)"
)
_SEP_LABELED_CCCD_RE = re.compile(
    r"(?i)(\b(?:cccd|cmnd|căn\s*cước(?:\s*công\s*dân)?|can\s*cuoc(?:\s*cong\s*dan)?)\s*[:#-]?\s*)(\d[\d\s.\-]{10,18}\d)(?!\d)"
)
_SEP_LABELED_PHONE_RE = re.compile(
    r"(?i)(\b(?:sđt|sdt|điện\s*thoại|dien\s*thoai|số\s*điện\s*thoại|phone)\s*[:#-]?\s*)((?:\+?84|0)[\d\s.\-]{8,16}\d)(?!\d)"
)

# Bare separated phone: 0987 654 321, 0987.654.321, +84 987 654 321
_SEP_BARE_PHONE_RE = re.compile(
    r"(?<!\d)((?:\+84[\s.\-]?|0)[35789](?:[\s.\-]?\d){8})(?!\d)"
)

# Bare 4-digit grouped bank account numbers separated by spaces/dashes: 1903 1234 5678 90
_SEP_BARE_ACCT_RE = re.compile(
    r"(?<!\d)(\d{4}(?:[\s\-]\d{4}){1,3}|\d{4}[\s\-]\d{4}[\s\-]\d{2,4})(?!\d)"
)

_VN_PHONE_10_RE = re.compile(r"^0[35789]\d{8}$")
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_INTERNAL_REF_RE = re.compile(r"\b(?:img|tx)_[0-9a-fA-F]{8,16}\b")
_UNMASKED_CONTIGUOUS_DIGITS_RE = re.compile(r"(?<!\d)(\d{8,16})(?!\d)")
_CURRENCY_SUFFIX_AFTER_DIGITS_RE = re.compile(
    r"^\s*(?:vn[dđ]|đồng|dong|đ|d)\b", re.IGNORECASE
)
_SKIP_AUDIT_KEYS = frozenset({"amount", "detected_total_vnd"})


class PiiLeakBlockedError(RuntimeError):
    """Ném lỗi fail-closed nếu phát hiện PII thô trong payload chuẩn bị gửi tới LLM."""


@dataclass(frozen=True)
class OutboundAuditEntry:
    timestamp: str
    session_id: str
    model: str
    payload_preview: str
    pii_clean: bool


class OutboundAuditLog:
    """Ring buffer lưu lịch sử payload đã kiểm duyệt sạch PII trước khi gửi Gemini."""

    def __init__(self, maxlen: int = 100) -> None:
        self._lock = threading.Lock()
        self._entries: deque[OutboundAuditEntry] = deque(maxlen=maxlen)

    def record(self, session_id: str, model: str, payload_preview: str) -> None:
        entry = OutboundAuditEntry(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            session_id=session_id,
            model=model,
            payload_preview=payload_preview[:2000],
            pii_clean=True,
        )
        with self._lock:
            self._entries.append(entry)

    def entries(self) -> list[OutboundAuditEntry]:
        with self._lock:
            return list(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


def _normalize_vn_phone(raw: str) -> str | None:
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("84") and len(digits) == 11:
        digits = "0" + digits[2:]
    if _VN_PHONE_10_RE.match(digits):
        return digits
    return None


class PrivacyGuard:
    """Bọc PrivacyProxy (T04) để xử lý separated-digit PII và kiểm tra fail-closed outbound."""

    def __init__(self, proxy: PrivacyProxy | None = None) -> None:
        self.proxy = proxy or PrivacyProxy()

    def _register_token(self, session_id: str, kind: str, raw_digits: str) -> str:
        sess = self.proxy._get_or_create_session(session_id)
        return self.proxy._register(sess, kind, raw_digits)

    def mask_text(self, session_id: str, text: str) -> MaskResult:
        """
        Ẩn danh toàn bộ STK/SĐT/CCCD (cả dạng liền lẫn tách bằng dấu cách/chấm/gạch ngang).
        """
        if not text:
            return MaskResult(masked_text="", tokens={})

        tokens_found: dict[str, str] = {}
        out = text

        def _repl_labeled_acct(m: re.Match[str]) -> str:
            prefix, candidate = m.group(1), m.group(2).strip()
            if _CURRENCY_GROUPED_RE.match(candidate):
                return m.group(0)
            digits = re.sub(r"\D", "", candidate)
            if 8 <= len(digits) <= 16:
                tok = self._register_token(session_id, "ACCT", digits)
                tokens_found[tok] = "ACCT"
                return f"{prefix}{tok}"
            return m.group(0)

        def _repl_labeled_cccd(m: re.Match[str]) -> str:
            prefix, candidate = m.group(1), m.group(2).strip()
            digits = re.sub(r"\D", "", candidate)
            if len(digits) == 12:
                tok = self._register_token(session_id, "CCCD", digits)
                tokens_found[tok] = "CCCD"
                return f"{prefix}{tok}"
            return m.group(0)

        def _repl_labeled_phone(m: re.Match[str]) -> str:
            prefix, candidate = m.group(1), m.group(2).strip()
            norm_phone = _normalize_vn_phone(candidate)
            if norm_phone is not None:
                tok = self._register_token(session_id, "PHONE", norm_phone)
                tokens_found[tok] = "PHONE"
                return f"{prefix}{tok}"
            return m.group(0)

        def _repl_bare_phone(m: re.Match[str]) -> str:
            candidate = m.group(1).strip()
            if _CURRENCY_GROUPED_RE.match(candidate):
                return m.group(0)
            norm_phone = _normalize_vn_phone(candidate)
            if norm_phone is not None:
                tok = self._register_token(session_id, "PHONE", norm_phone)
                tokens_found[tok] = "PHONE"
                return tok
            return m.group(0)

        def _repl_bare_acct(m: re.Match[str]) -> str:
            candidate = m.group(1).strip()
            digits = re.sub(r"\D", "", candidate)
            if 8 <= len(digits) <= 16:
                tok = self._register_token(session_id, "ACCT", digits)
                tokens_found[tok] = "ACCT"
                return tok
            return m.group(0)

        out = _SEP_LABELED_ACCT_RE.sub(_repl_labeled_acct, out)
        out = _SEP_LABELED_CCCD_RE.sub(_repl_labeled_cccd, out)
        out = _SEP_LABELED_PHONE_RE.sub(_repl_labeled_phone, out)
        out = _SEP_BARE_PHONE_RE.sub(_repl_bare_phone, out)
        out = _SEP_BARE_ACCT_RE.sub(_repl_bare_acct, out)

        base_res = self.proxy.mask(session_id, out)
        merged_tokens = {**tokens_found, **base_res.tokens}
        return MaskResult(masked_text=base_res.masked_text, tokens=merged_tokens)

    def demask(self, session_id: str, text: str) -> str:
        return self.proxy.demask(session_id, text)

    def resolve_token(self, session_id: str, token: str) -> str | None:
        return self.proxy.resolve_token(session_id, token)

    def get_session_raw_pii(self, session_id: str) -> set[str]:
        self.proxy._purge_expired()
        sess = self.proxy._sessions.get(session_id)
        if sess is None:
            return set()
        return set(sess.token_to_raw.values())

    def assert_zero_pii_outbound(
        self,
        session_id: str,
        payload: Any,
        extra_forbidden: set[str] | None = None,
    ) -> str:
        """
        Kiểm tra fail-closed mọi chuỗi trong payload trước khi gửi tới LLM.
        Trả về JSON preview sạch (đã xác nhận 0% PII) để ghi vào OutboundAuditLog.
        """
        forbidden = {v for v in self.get_session_raw_pii(session_id) if len(v) >= 6}
        if extra_forbidden:
            forbidden |= {v for v in extra_forbidden if v and len(v) >= 6}

        texts: list[str] = []

        def _collect(node: Any, key_name: str | None = None) -> None:
            if node is None or isinstance(node, (bool, int, float)):
                return
            if isinstance(node, str):
                if key_name in _SKIP_AUDIT_KEYS:
                    return
                texts.append(node)
                return
            if isinstance(node, dict):
                for k, v in node.items():
                    _collect(v, str(k))
                return
            if isinstance(node, (list, tuple)):
                for item in node:
                    _collect(item, key_name)
                return
            texts.append(str(node))

        _collect(payload)

        for txt in texts:
            for secret in forbidden:
                if secret in txt:
                    raise PiiLeakBlockedError(
                        "Phát hiện PII thô (đã được đăng ký trong phiên) chuẩn bị gửi ra LLM."
                    )

            for pm in _SEP_BARE_PHONE_RE.finditer(txt):
                if _normalize_vn_phone(pm.group(1)) is not None:
                    raise PiiLeakBlockedError(
                        "Phát hiện số điện thoại chưa ẩn danh chuẩn bị gửi ra LLM."
                    )

            txt_no_uuid = _INTERNAL_REF_RE.sub("<REF_ID>", _UUID_RE.sub("<UUID>", txt))
            for dm in _UNMASKED_CONTIGUOUS_DIGITS_RE.finditer(txt_no_uuid):
                suffix = txt_no_uuid[dm.end() :]
                if _CURRENCY_SUFFIX_AFTER_DIGITS_RE.match(suffix):
                    continue
                raise PiiLeakBlockedError(
                    "Phát hiện chuỗi 8-16 chữ số chưa ẩn danh chuẩn bị gửi ra LLM."
                )

        return json.dumps(payload, ensure_ascii=False, default=str)
