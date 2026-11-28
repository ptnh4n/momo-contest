"""
privacy_proxy.py — Regex Privacy Proxy (Masking/De-masking tạm trong RAM)
và khóa đầu ra Zero-PII cho tool `lookup_contact`.

Đặc tả (CONTEXT.md mục 3.2 & tutorial.md mục 2):
- Quét chuỗi đầu vào bằng Regex nhận diện:
  * STK ngân hàng (8–16 chữ số) -> <ACCT_k>
  * SĐT Việt Nam (0[3|5|7|8|9]\\d{8}) -> <PHONE_k>
  * Số CCCD (12 chữ số) -> <CCCD_k>
- Bảng ánh xạ (mapping table) chỉ lưu tạm trong RAM theo phiên với TTL mặc định 5 phút (300s).
- Tool `lookup_contact` khóa cứng đầu ra Zero-PII: chỉ trả các trường được phép
  (`contact_id`, `alias`, `bank_code`, `masked_acct` dạng `****789`, `trust_level`),
  tuyệt đối không để lọt STK/SĐT/CCCD thật.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import sqlite3
import time
from typing import Callable

from app.db.repo.contacts import lookup_contact_zero_pii

DEFAULT_TTL_SECONDS = 300.0  # 5 phút theo CONTEXT.md mục 3.1

# Regex nhận diện PII theo thứ tự ưu tiên ngữ cảnh
_LABELED_ACCT_RE = re.compile(
    r"(?i)(\b(?:stk|số\s*tài\s*khoản|so\s*tai\s*khoan|tài\s*khoản|tai\s*khoan|số\s*tk|so\s*tk)\s*[:#-]?\s*)(\d{8,16})\b"
)
_LABELED_CCCD_RE = re.compile(
    r"(?i)(\b(?:cccd|cmnd|căn\s*cước(?:\s*công\s*dân)?|can\s*cuoc(?:\s*cong\s*dan)?)\s*[:#-]?\s*)(\d{12})\b"
)
_LABELED_PHONE_RE = re.compile(
    r"(?i)(\b(?:sđt|sdt|điện\s*thoại|dien\s*thoai|số\s*điện\s*thoại|phone)\s*[:#-]?\s*)(0[35789]\d{8})\b"
)
_BARE_PHONE_RE = re.compile(r"(?<!\d)(0[35789]\d{8})(?!\d)")
_BARE_ACCT_RE = re.compile(r"(?<!\d)(\d{8,16})(?!\d)")

_TOKEN_RE = re.compile(r"<(?:ACCT|PHONE|CCCD)_\d+>")
_MASKED_ACCT_RE = re.compile(r"^\*{4}.{1,3}$")
_RAW_PII_LEAK_RE = re.compile(r"(?<!\d)(?:0[35789]\d{8}|\d{8,16})(?!\d)")

ALLOWED_ZERO_PII_KEYS = frozenset(
    {"contact_id", "alias", "bank_code", "masked_acct", "trust_level"}
)


@dataclass
class _SessionMapping:
    token_to_raw: dict[str, str] = field(default_factory=dict)
    raw_to_token: dict[str, str] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=lambda: {"ACCT": 0, "PHONE": 0, "CCCD": 0})
    expires_at: float = 0.0


@dataclass(frozen=True)
class MaskResult:
    masked_text: str
    tokens: dict[str, str]  # token -> kind ('ACCT' | 'PHONE' | 'CCCD'), KHÔNG chứa raw PII


class PrivacyProxy:
    """
    Privacy Proxy lưu bảng ánh xạ tạm trong RAM theo `session_id` với TTL.
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._sessions: dict[str, _SessionMapping] = {}

    def _purge_expired(self) -> None:
        now = self._clock()
        expired = [sid for sid, sess in self._sessions.items() if sess.expires_at <= now]
        for sid in expired:
            del self._sessions[sid]

    def _get_or_create_session(self, session_id: str) -> _SessionMapping:
        self._purge_expired()
        now = self._clock()
        sess = self._sessions.get(session_id)
        if sess is None:
            sess = _SessionMapping(expires_at=now + self._ttl_seconds)
            self._sessions[session_id] = sess
        else:
            sess.expires_at = now + self._ttl_seconds
        return sess

    def _register(self, sess: _SessionMapping, kind: str, raw_value: str) -> str:
        existing = sess.raw_to_token.get(raw_value)
        if existing is not None:
            return existing
        sess.counters[kind] += 1
        token = f"<{kind}_{sess.counters[kind]}>"
        sess.raw_to_token[raw_value] = token
        sess.token_to_raw[token] = raw_value
        return token

    def mask(self, session_id: str, text: str) -> MaskResult:
        """
        Thay thế STK (8-16 số), SĐT VN, CCCD (12 số) thành token ẩn danh
        <ACCT_k>, <PHONE_k>, <CCCD_k> và lưu mapping trong RAM theo session_id.
        """
        sess = self._get_or_create_session(session_id)
        tokens_found: dict[str, str] = {}

        def _sub_labeled(kind: str) -> Callable[[re.Match[str]], str]:
            def _repl(match: re.Match[str]) -> str:
                prefix, raw = match.group(1), match.group(2)
                tok = self._register(sess, kind, raw)
                tokens_found[tok] = kind
                return f"{prefix}{tok}"

            return _repl

        def _sub_bare(kind: str) -> Callable[[re.Match[str]], str]:
            def _repl(match: re.Match[str]) -> str:
                raw = match.group(1)
                tok = self._register(sess, kind, raw)
                tokens_found[tok] = kind
                return tok

            return _repl

        out = text
        # 1. Ngữ cảnh tường minh trước (VD: "STK 0987654321" phải là ACCT, không phải PHONE)
        out = _LABELED_ACCT_RE.sub(_sub_labeled("ACCT"), out)
        out = _LABELED_CCCD_RE.sub(_sub_labeled("CCCD"), out)
        out = _LABELED_PHONE_RE.sub(_sub_labeled("PHONE"), out)
        # 2. Số trần không nhãn: SĐT VN (10 số) trước, rồi STK (8-16 số)
        out = _BARE_PHONE_RE.sub(_sub_bare("PHONE"), out)
        out = _BARE_ACCT_RE.sub(_sub_bare("ACCT"), out)

        return MaskResult(masked_text=out, tokens=tokens_found)

    def demask(self, session_id: str, text: str) -> str:
        """
        Ánh xạ ngược token (<ACCT_1>, <PHONE_1>, <CCCD_1>) về giá trị thật trong RAM.
        Nếu phiên đã hết hạn hoặc không tồn tại, không giải mã (trả nguyên chuỗi).
        """
        self._purge_expired()
        sess = self._sessions.get(session_id)
        if sess is None:
            return text

        def _repl(match: re.Match[str]) -> str:
            tok = match.group(0)
            return sess.token_to_raw.get(tok, tok)

        return _TOKEN_RE.sub(_repl, text)

    def resolve_token(self, session_id: str, token: str) -> str | None:
        """Tra cứu 1 token cụ thể trong RAM (dùng cục bộ ở Backend)."""
        self._purge_expired()
        sess = self._sessions.get(session_id)
        if sess is None:
            return None
        return sess.token_to_raw.get(token)

    def clear_session(self, session_id: str) -> None:
        """Xóa ngay bảng ánh xạ của phiên khỏi RAM."""
        self._sessions.pop(session_id, None)

    def has_session(self, session_id: str) -> bool:
        self._purge_expired()
        return session_id in self._sessions


def _mask_account_tail(account_or_masked: str) -> str:
    """Đảm bảo định dạng masked_acct luôn là '****' + 3 ký tự cuối (VD: '****789')."""
    if _MASKED_ACCT_RE.match(account_or_masked):
        return account_or_masked
    tail = account_or_masked[-3:] if len(account_or_masked) >= 3 else account_or_masked
    return f"****{tail}"


def _sanitize_alias_no_pii(alias: str) -> str:
    """Chặn tuyệt đối mọi chuỗi STK/SĐT/CCCD vô tình lọt vào trường alias trả cho LLM."""
    return _RAW_PII_LEAK_RE.sub("****", alias)


def lookup_contact(conn: sqlite3.Connection, owner_id: str, query: str) -> dict:
    """
    Tra cứu danh bạ nội bộ. Tool sẽ lọc các trường trước khi trả cho LLM.

    Khóa cứng đầu ra theo chuẩn Zero-PII:
    - Chỉ giữ đúng whitelist key: contact_id, alias, bank_code, masked_acct, trust_level.
    - `masked_acct` bắt buộc dạng `****xxx` (3 số cuối).
    - Chặn/ẩn mọi chuỗi số PII (8-16 số, SĐT VN) nếu có trong alias.
    - Tuyệt đối KHÔNG trả account_number đầy đủ, full_name, SĐT, số dư.
    """
    raw_result = lookup_contact_zero_pii(conn, owner_id, query)
    locked_matches: list[dict[str, str]] = []

    for item in raw_result.get("matches", []):
        locked_item = {
            "contact_id": str(item["contact_id"]),
            "alias": _sanitize_alias_no_pii(str(item["alias"])),
            "bank_code": str(item["bank_code"]),
            "masked_acct": _mask_account_tail(str(item["masked_acct"])),
            "trust_level": str(item["trust_level"]),
        }
        # Double-check khóa cứng: chỉ đúng các key được phép và 0% PII thô
        assert set(locked_item.keys()) == ALLOWED_ZERO_PII_KEYS
        assert _MASKED_ACCT_RE.match(locked_item["masked_acct"])
        locked_matches.append(locked_item)

    return {"matches": locked_matches}
