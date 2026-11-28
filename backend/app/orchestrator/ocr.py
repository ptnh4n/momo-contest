"""
ocr.py — Dịch vụ OCR hóa đơn 100% cục bộ (RapidOCR + PrivacyGuard).

Nguyên tắc bảo mật:
- Ảnh base64 chỉ lưu tạm trong RAM (`ImageStagingStore`, TTL 300s), KHÔNG lưu đĩa.
- Trích xuất chữ trên Backend bằng RapidOCR -> lập tức mask qua `PrivacyGuard`
  -> chỉ trả văn bản đã mask (`<ACCT_k>`, `<PHONE_k>`, `<CCCD_k>`) cho Gemini.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import re
import threading
import time
from typing import Any, Callable
from uuid import uuid4

from app.orchestrator.privacy_guard import PrivacyGuard
from app.wallet.currency_parser import parse_currency
from app.wallet.privacy_proxy import DEFAULT_TTL_SECONDS

_DATA_URL_PREFIX_RE = re.compile(r"^data:image/[^;]+;base64,", re.IGNORECASE)
_ACCT_TOKEN_RE = re.compile(r"<ACCT_\d+>")
_TOTAL_LINE_RE = re.compile(
    r"(?i)(?:tổng\s*cộng|tong\s*cong|tổng\s*tiền|tong\s*tien|thành\s*tiền|thanh\s*tien|"
    r"tổng\s*thanh\s*toán|tong\s*thanh\s*toan|phải\s*trả|phai\s*tra|total)"
)

_BANK_ALIASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:vietcombank|vcb)\b", re.IGNORECASE), "VCB"),
    (re.compile(r"\b(?:techcombank|tcb)\b", re.IGNORECASE), "TCB"),
    (re.compile(r"\b(?:mbbank|mb\s*bank|mb)\b", re.IGNORECASE), "MB"),
    (re.compile(r"\b(?:acb)\b", re.IGNORECASE), "ACB"),
    (re.compile(r"\b(?:bidv)\b", re.IGNORECASE), "BIDV"),
    (re.compile(r"\b(?:vietinbank|vtb)\b", re.IGNORECASE), "VTB"),
    (re.compile(r"\b(?:tpbank|tpb)\b", re.IGNORECASE), "TPB"),
    (re.compile(r"\b(?:vpbank|vpb)\b", re.IGNORECASE), "VPB"),
    (re.compile(r"\b(?:sacombank|stb)\b", re.IGNORECASE), "STB"),
    (re.compile(r"\b(?:vib)\b", re.IGNORECASE), "VIB"),
    (re.compile(r"\b(?:momo)\b", re.IGNORECASE), "MOMO"),
)


@dataclass
class _StagedImage:
    user_id: str
    image_bytes: bytes
    expires_at: float


class ImageStagingStore:
    """Lưu tạm bytes ảnh hóa đơn trong RAM theo TTL (không ghi xuống đĩa)."""

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._items: dict[str, _StagedImage] = {}

    def _purge(self) -> None:
        now = self._clock()
        expired = [k for k, v in self._items.items() if v.expires_at <= now]
        for k in expired:
            del self._items[k]

    def put(self, user_id: str, image_bytes: bytes) -> str:
        image_id = f"img_{uuid4().hex[:12]}"
        with self._lock:
            self._purge()
            self._items[image_id] = _StagedImage(
                user_id=user_id,
                image_bytes=image_bytes,
                expires_at=self._clock() + self._ttl_seconds,
            )
        return image_id

    def get(self, image_id: str, user_id: str) -> bytes | None:
        with self._lock:
            self._purge()
            item = self._items.get(image_id)
            if item is None or item.user_id != user_id:
                return None
            return item.image_bytes


def decode_base64_image(image_base64: str) -> bytes:
    """Giải mã chuỗi base64 (hỗ trợ cả `data:image/...;base64,...`)."""
    raw = (image_base64 or "").strip()
    if not raw:
        raise ValueError("Chuỗi image_base64 rỗng.")
    raw = _DATA_URL_PREFIX_RE.sub("", raw)
    try:
        data = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise ValueError("Chuỗi image_base64 không hợp lệ.") from exc
    if not data:
        raise ValueError("Dữ liệu ảnh sau giải mã rỗng.")
    return data


_ocr_lock = threading.Lock()
_ocr_engine: Any = None


def _get_rapidocr_engine() -> Any:
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine
    with _ocr_lock:
        if _ocr_engine is None:
            from rapidocr import RapidOCR

            _ocr_engine = RapidOCR()
    return _ocr_engine


def run_rapidocr_bytes(image_bytes: bytes) -> tuple[str, list[str], float]:
    """Chạy RapidOCR cục bộ trên bytes ảnh, trả về (text_gộp, các_dòng, latency_ms)."""
    engine = _get_rapidocr_engine()
    t0 = time.perf_counter()
    out = engine(image_bytes)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    txts = getattr(out, "txts", None)
    lines = [str(line).strip() for line in txts if str(line).strip()] if txts else []
    return "\n".join(lines), lines, elapsed_ms


def parse_receipt_hints(masked_text: str, *, require_total_label: bool = False) -> dict[str, Any]:
    """Trích gợi ý có cấu trúc từ văn bản hóa đơn ĐÃ ĐƯỢC MASK PII."""
    account_tokens = list(dict.fromkeys(_ACCT_TOKEN_RE.findall(masked_text)))

    bank_codes: list[str] = []
    for pat, code in _BANK_ALIASES:
        if pat.search(masked_text) and code not in bank_codes:
            bank_codes.append(code)

    detected_total_vnd: int | None = None
    for line in masked_text.splitlines():
        if _TOTAL_LINE_RE.search(line):
            cur = parse_currency(line)
            if cur.status == "OK" and cur.amount is not None:
                detected_total_vnd = cur.amount
                break

    if detected_total_vnd is None and not require_total_label:
        # Fallback: lấy số tiền hợp lệ lớn nhất tìm được trong hóa đơn
        candidates: list[int] = []
        for line in masked_text.splitlines():
            cur = parse_currency(line)
            if cur.status == "OK" and cur.amount is not None:
                candidates.append(cur.amount)
        if candidates:
            detected_total_vnd = max(candidates)

    return {
        "account_tokens": account_tokens,
        "bank_codes": bank_codes,
        "detected_total_vnd": detected_total_vnd,
    }


OcrCallable = Callable[[bytes], tuple[str, list[str], float]]


class OcrReceiptService:
    """Quản lý staging ảnh và chạy OCR cục bộ kèm mask Zero-PII trước khi trả cho LLM."""

    def __init__(
        self,
        privacy_guard: PrivacyGuard,
        image_store: ImageStagingStore | None = None,
        ocr_fn: OcrCallable | None = None,
    ) -> None:
        self.privacy_guard = privacy_guard
        self.image_store = image_store or ImageStagingStore()
        self._ocr_fn = ocr_fn or run_rapidocr_bytes

    def stage_image(self, user_id: str, image_base64: str) -> str:
        image_bytes = decode_base64_image(image_base64)
        return self.image_store.put(user_id, image_bytes)

    def parse_receipt(
        self, user_id: str, session_id: str, image_id: str, *, require_total_label: bool = False
    ) -> dict[str, Any]:
        image_bytes = self.image_store.get(image_id, user_id)
        if image_bytes is None:
            return {
                "image_id": image_id,
                "source_type": "OCR_RECEIPT",
                "error": "Không tìm thấy ảnh hóa đơn hoặc phiên ảnh đã hết hạn.",
                "masked_ocr_text": "",
                "account_tokens": [],
                "bank_codes": [],
                "detected_total_vnd": None,
            }

        try:
            raw_text, _lines, elapsed_ms = self._ocr_fn(image_bytes)
        except Exception as exc:
            return {
                "image_id": image_id,
                "source_type": "OCR_RECEIPT",
                "error": f"Không thể đọc nội dung ảnh hóa đơn ({type(exc).__name__}).",
                "masked_ocr_text": "",
                "account_tokens": [],
                "bank_codes": [],
                "detected_total_vnd": None,
            }

        mask_res = self.privacy_guard.mask_text(session_id, raw_text)
        hints = parse_receipt_hints(mask_res.masked_text, require_total_label=require_total_label)
        return {
            "image_id": image_id,
            "source_type": "OCR_RECEIPT",
            "masked_ocr_text": mask_res.masked_text,
            "account_tokens": hints["account_tokens"],
            "bank_codes": hints["bank_codes"],
            "detected_total_vnd": hints["detected_total_vnd"],
            "ocr_latency_ms": round(elapsed_ms, 2),
        }
