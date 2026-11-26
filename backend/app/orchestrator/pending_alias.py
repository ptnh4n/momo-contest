"""
pending_alias.py — Bộ nhớ tạm trong RAM cho quy trình xác nhận 2 bước (Phương án B)
khi tool `upsert_contact_alias` gặp STK thô MỚI chưa có trong bảng `contacts`.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import threading
import time
from typing import Callable

from app.core.text_norm import normalize_text
from app.wallet.privacy_proxy import DEFAULT_TTL_SECONDS

_AFFIRMATIVE_EXACT = frozenset(
    {
        "dong y",
        "ok",
        "oke",
        "okay",
        "xac nhan",
        "co",
        "luu",
        "luu di",
        "luu nhe",
        "dong y luu",
        "yes",
        "dung",
        "dung roi",
        "chuan",
        "duyet",
    }
)
_NEGATIVE_EXACT = frozenset(
    {
        "huy",
        "khong",
        "ko",
        "k",
        "thoi",
        "cancel",
        "no",
        "bo qua",
        "khong luu",
        "dung luu",
    }
)


@dataclass
class PendingAliasItem:
    user_id: str
    intent_id: str
    raw_account: str
    masked_acct: str
    bank_code: str
    alias_raw: str
    session_id: str
    expires_at: float


class PendingAliasStore:
    """Lưu tạm yêu cầu gắn alias cho STK mới trong RAM theo user_id (TTL 300s)."""

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._items: dict[str, PendingAliasItem] = {}

    def _purge(self) -> None:
        now = self._clock()
        expired = [k for k, v in self._items.items() if v.expires_at <= now]
        for k in expired:
            del self._items[k]

    def put(self, item: PendingAliasItem) -> None:
        with self._lock:
            self._purge()
            item.expires_at = self._clock() + self._ttl_seconds
            self._items[item.user_id] = item

    def get(self, user_id: str) -> PendingAliasItem | None:
        with self._lock:
            self._purge()
            return self._items.get(user_id)

    def pop(self, user_id: str) -> PendingAliasItem | None:
        with self._lock:
            self._purge()
            return self._items.pop(user_id, None)


def _clean_reply(text: str) -> str:
    norm = normalize_text(text)
    norm = re.sub(r"[.!?,:;]+", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()


def is_affirmative_reply(text: str) -> bool:
    return _clean_reply(text) in _AFFIRMATIVE_EXACT


def is_negative_reply(text: str) -> bool:
    return _clean_reply(text) in _NEGATIVE_EXACT
