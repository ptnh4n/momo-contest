"""
preprocessing.py — Single Source of Truth cho tiền xử lý Text & Tabular của SafePay Gatekeeper.

Dùng chung cho:
- T08 (Sinh dữ liệu huấn luyện)
- T09 (Huấn luyện & xuất ONNX Gatekeeper)
- T12 (Tích hợp ONNX Runtime vào Backend Middleware)
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm

_GATEKEEPER_DIR = Path(__file__).resolve().parent
DEFAULT_SCHEMA_PATH = _GATEKEEPER_DIR / "feature_schema.json"
DEFAULT_TOKENIZER_PATH = _GATEKEEPER_DIR / "tokenizer" / "safepay_spm.model"

VN_TZ = timezone(timedelta(hours=7))

_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad\u2060]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_PRIVACY_OVERFLOW_RE = re.compile(r"<(ACCT|PHONE|CCCD)_(?:[6-9]|\d{2,})>")
_DELIMITER_SPOOF_RE = re.compile(r"\[(CLS|SEP|PAD|MASK|UNK)\]", re.IGNORECASE)

# Phát hiện số tiền trần không kèm đơn vị (ví dụ: "chuyển 500 cho Nam", "gửi 200 cho Minh")
# Không khớp:
# - Số có dấu chấm/phẩy phân cách hàng nghìn theo sau (500.000, 1,500,000)
# - Số có hậu tố đơn vị (k, tr, triệu, nghìn, ngàn, đ, d, đồng, vnd, củ, lít, xị, %, người, phút, giờ, tháng...)
# - Số đứng sau ngữ cảnh không phải tiền (tháng 11, ngày 20, bàn 4, mức 2, lần 1, quận 1, lớp 5...)
# - Chuỗi >= 6 chữ số liên tiếp (STK, SĐT, mã đơn, năm...) hoặc bên trong placeholder <ACCT_1>
_AMBIGUOUS_AMOUNT_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9_<.,])(\d{2,4})(?![A-Za-z0-9_>.,])"
)
_PRE_NON_MONEY_STRICT_RE = re.compile(
    r"\b(?:tháng|thang|năm|nam|bàn|ngày|lần|lan|mức|muc|quận|quan|phòng|phong|"
    r"tầng|tang|lớp|lop|nhóm|nhom|chia|gồm|gom|top|mã|ma|số|so|stk|tk|id|ver|v)\s*$",
    re.IGNORECASE,
)
_PRE_NGAY_RE = re.compile(r"\bngay\s*$", re.IGNORECASE)
_PRE_BAN_RE = re.compile(r"\bban\s*$", re.IGNORECASE)
_TRAILING_TRANSFER_VERB_RE = re.compile(
    r"\b(?:chuyển|chuyen|ck|gửi|gui|bắn|ban|trả|tra|nộp|nop|đóng|dong)\s+$",
    re.IGNORECASE,
)
_POST_UNIT_RE = re.compile(
    r"^\s*(?:k\b|tr\b|triệu|trieu|nghìn|nghin|ngàn|ngan|đ\b|d\b|đồng|dong|vnd|vnđ|"
    r"củ|cu\b|lít|lit\b|xị|xi\b|%|người|nguoi|bạn|ban|đứa|dua|phút|phut|giờ|gio|h\b|"
    r"ngày|ngay|tháng|thang|tuần|tuan|năm|nam|món|mon|ly\b|cốc|coc|phần|phan|suất|suat)",
    re.IGNORECASE,
)
_TRANSFER_INTENT_HINT_RE = re.compile(
    r"\b(?:chuyển|chuyen|ck|gửi|gui|bắn|ban|trả|tra|nộp|nop|đóng|dong|thanh\s*toán|thanh\s*toan)\b",
    re.IGNORECASE,
)


def load_feature_schema(schema_path: str | Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    path = Path(schema_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str | None) -> str:
    """
    Chuẩn hóa Unicode NFC, giữ nguyên HOA/thường và dấu tiếng Việt,
    xóa ký tự zero-width chống né bộ lọc, và gộp placeholder vượt ngưỡng (>5) về dạng chuẩn.
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = _ZERO_WIDTH_RE.sub("", s)
    s = _PRIVACY_OVERFLOW_RE.sub(r"<\1>", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def sanitize_delimiters(text: str | None) -> str:
    """
    Chống Delimiter Spoofing: vô hiệu hóa các token điều khiển giả mạo như
    [SEP], [CLS], [MASK], [PAD], [UNK] nằm trong nội dung người dùng hoặc ghi chú.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    return _DELIMITER_SPOOF_RE.sub(r"(\1)", cleaned)


def detect_ambiguous_amount(message: str | None) -> bool:
    """
    Phát hiện số tiền trần mơ hồ đơn vị tiền tệ tiếng Việt (VD: 'chuyển 500 cho Nam').
    Trả về False nếu số đã có đơn vị rõ ràng (500k, 500.000đ, 2tr) hoặc chỉ ngữ cảnh phi tiền tệ.
    """
    cleaned = clean_text(message)
    if not cleaned or not _TRANSFER_INTENT_HINT_RE.search(cleaned):
        return False

    for match in _AMBIGUOUS_AMOUNT_NUM_RE.finditer(cleaned):
        val = int(match.group(1))
        if val < 10:
            continue
        prefix = cleaned[: match.start()]
        suffix = cleaned[match.end() :]
        if _PRE_NON_MONEY_STRICT_RE.search(prefix):
            continue
        ngay_match = _PRE_NGAY_RE.search(prefix)
        if ngay_match and 1 <= val <= 31:
            prefix_before_ngay = prefix[: ngay_match.start()]
            if not _TRAILING_TRANSFER_VERB_RE.search(prefix_before_ngay):
                continue
        ban_match = _PRE_BAN_RE.search(prefix)
        if ban_match:
            prefix_before_ban = prefix[: ban_match.start()]
            if _TRANSFER_INTENT_HINT_RE.search(prefix_before_ban) is not None:
                continue
        if _POST_UNIT_RE.search(suffix):
            continue
        return True
    return False


class GatekeeperTokenizer:
    """Wrapper SentencePiece cho SafePay Gatekeeper với cơ chế bảo toàn Segment B."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_TOKENIZER_PATH,
        schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    ) -> None:
        self.schema = load_feature_schema(schema_path)
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy tokenizer model: {self.model_path}")

        self.sp = spm.SentencePieceProcessor(model_file=str(self.model_path))
        text_cfg = self.schema["text_config"]
        self.max_length: int = int(text_cfg["max_length"])
        self.max_seg_b_tokens: int = int(text_cfg["max_seg_b_tokens"])
        self.valid_source_types = set(self.schema["source_types"])
        self.valid_tool_names = set(self.schema["tool_names"])

        ctrl = text_cfg["control_tokens"]
        self.pad_id: int = self.sp.piece_to_id(ctrl["pad"])
        self.unk_id: int = self.sp.piece_to_id(ctrl["unk"])
        self.cls_id: int = self.sp.piece_to_id(ctrl["cls"])
        self.sep_id: int = self.sp.piece_to_id(ctrl["sep"])
        self.mask_id: int = self.sp.piece_to_id(ctrl["mask"])

        for name, tid in [
            ("pad", self.pad_id),
            ("unk", self.unk_id),
            ("cls", self.cls_id),
            ("sep", self.sep_id),
            ("mask", self.mask_id),
        ]:
            if tid < 0 or (name != "unk" and tid == self.unk_id):
                raise ValueError(f"Control token '{name}' chưa được đăng ký đúng trong SentencePiece.")

    @property
    def vocab_size(self) -> int:
        return int(self.sp.get_piece_size())

    def _encode_raw(self, text: str) -> list[int]:
        if not text:
            return []
        return list(self.sp.encode(text, out_type=int))

    def encode_pair(
        self,
        user_message: str,
        tool_name: str,
        note: str | None,
        source_type: str,
        max_length: int | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Mã hóa cặp văn bản theo mẫu:
            [CLS] <User_Message> [SEP] <Tool_Name> | <Note> | <Source_Type> [SEP]

        Chiến lược cắt độ dài:
        - Luôn bảo toàn `tool_name` và `source_type` ở Segment B (chỉ cắt bớt `note` nếu Seg B > max_seg_b_tokens).
        - Nếu tổng chuỗi > max_length: cắt Segment A theo head+tail giữ cả đầu và đuôi câu người dùng.
        """
        if source_type not in self.valid_source_types:
            raise ValueError(f"source_type không hợp lệ: {source_type!r}")
        if tool_name not in self.valid_tool_names:
            raise ValueError(f"tool_name không hợp lệ: {tool_name!r}")

        seq_max = int(max_length or self.max_length)
        safe_msg = sanitize_delimiters(user_message)
        safe_note = sanitize_delimiters(note)

        # Build Segment B while strictly preserving tool_name and source_type
        tool_ids = self._encode_raw(f"{tool_name} |")
        src_ids = self._encode_raw(f"| {source_type}")
        note_ids = self._encode_raw(safe_note)

        max_note_budget = max(0, self.max_seg_b_tokens - len(tool_ids) - len(src_ids))
        if len(note_ids) > max_note_budget:
            note_ids = note_ids[:max_note_budget]
        seg_b_ids = tool_ids + note_ids + src_ids

        # Reserve 3 slots for [CLS], middle [SEP], trailing [SEP]
        seg_a_budget = max(1, seq_max - 3 - len(seg_b_ids))
        seg_a_ids = self._encode_raw(safe_msg)
        if len(seg_a_ids) > seg_a_budget:
            head_len = seg_a_budget // 2
            tail_len = seg_a_budget - head_len
            seg_a_ids = (seg_a_ids[:head_len] if head_len > 0 else []) + (
                seg_a_ids[-tail_len:] if tail_len > 0 else []
            )

        seg_zero = [self.cls_id] + seg_a_ids + [self.sep_id]
        seg_one = seg_b_ids + [self.sep_id]
        full_ids = seg_zero + seg_one
        type_ids = [0] * len(seg_zero) + [1] * len(seg_one)
        attn_mask = [1] * len(full_ids)

        pad_len = seq_max - len(full_ids)
        if pad_len > 0:
            full_ids.extend([self.pad_id] * pad_len)
            type_ids.extend([0] * pad_len)
            attn_mask.extend([0] * pad_len)

        return {
            "input_ids": np.asarray(full_ids[:seq_max], dtype=np.int64),
            "attention_mask": np.asarray(attn_mask[:seq_max], dtype=np.int64),
            "token_type_ids": np.asarray(type_ids[:seq_max], dtype=np.int64),
        }


# ---------------------------------------------------------------------------
# Tabular Feature Extraction (14 Features — Deterministic Fixed Scaling)
# ---------------------------------------------------------------------------
def _clip(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(val)))


def _parse_utc_timestamp(ts: str | datetime | None) -> datetime | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    s = str(ts).strip()
    if not s:
        return None
    # SQLite CURRENT_TIMESTAMP format: 'YYYY-MM-DD HH:MM:SS'
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def extract_raw_features(
    payload: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    Tính vector 14 đặc trưng Tabular chuẩn hóa cố định (0% train-serve skew).

    Các khóa đầu vào hỗ trợ trong `payload`:
      - amount (int/float, bắt buộc)
      - balance (int/float, mặc định 0)
      - auto_approve_limit (int/float, mặc định 100_000)
      - daily_transferred (int/float, mặc định 0)
      - user_avg_amount (float | None, nếu None/<=0 dùng cold-start max(auto_approve_limit, 200_000))
      - contact_id (str | None)
      - trust_level (str, 'TRUSTED' | 'NEW' | None)
      - recipient_transfer_count (int, mặc định 0)
      - matched_contact_count (int, mặc định 0)
      - source_type (str, 'CHAT_CONTACT' | 'CHAT_RAW_ACCT' | 'OCR_RECEIPT')
      - vn_local_hour (int | None) hoặc now_utc (datetime | str | None)
      - time_since_last_tx_seconds (float | None, mặc định 2_592_000.0)
      - amount_ambiguous (bool | None) + tùy chọn message (str) để suy ra nếu amount_ambiguous is None
    """
    cfg = schema or load_feature_schema()
    sbv = cfg["sbv_limits"]
    valid_sources = set(cfg["source_types"])

    source_type = str(payload.get("source_type", "CHAT_RAW_ACCT"))
    if source_type not in valid_sources:
        raise ValueError(f"source_type không hợp lệ: {source_type!r}")

    amount = max(0.0, float(payload.get("amount", 0.0)))
    balance = max(0.0, float(payload.get("balance", 0.0)))
    auto_limit = max(0.0, float(payload.get("auto_approve_limit", sbv["default_auto_approve_limit_vnd"])))
    daily_transferred = max(0.0, float(payload.get("daily_transferred", 0.0)))

    raw_avg = payload.get("user_avg_amount")
    if raw_avg is None or float(raw_avg) <= 0.0:
        user_avg_amount = max(auto_limit, float(sbv["cold_start_avg_amount_vnd"]))
    else:
        user_avg_amount = float(raw_avg)

    contact_id = payload.get("contact_id")
    trust_level = payload.get("trust_level")
    transfer_count = max(0, int(payload.get("recipient_transfer_count", 0) or 0))
    matched_count = max(0, int(payload.get("matched_contact_count", 0) or 0))

    if payload.get("vn_local_hour") is not None:
        vn_hour = int(payload["vn_local_hour"]) % 24
    else:
        now_dt = _parse_utc_timestamp(payload.get("now_utc")) or datetime.now(timezone.utc)
        vn_hour = now_dt.astimezone(VN_TZ).hour

    delta_sec = payload.get("time_since_last_tx_seconds")
    if delta_sec is None:
        delta_sec = float(sbv["no_prev_tx_seconds"])
    else:
        delta_sec = _clip(float(delta_sec), 0.0, float(sbv["no_prev_tx_seconds"]))

    if payload.get("amount_ambiguous") is not None:
        ambiguous = bool(payload["amount_ambiguous"])
    else:
        ambiguous = detect_ambiguous_amount(payload.get("message", ""))

    # 0..9: Blueprint 10 features + 10..13: Schema 4 extension features
    ratio_avg_amount = math.log1p(_clip(amount / (user_avg_amount + 1.0), 0.0, 1000.0)) / 7.0
    amount_log = _clip(math.log1p(amount) / 20.0, 0.0, 1.25)
    is_new_recipient = 1.0 if (contact_id is None or trust_level != "TRUSTED") else 0.0
    transfer_count_log = math.log1p(_clip(transfer_count, 0.0, 10000.0)) / 10.0
    matched_contact_count = _clip(matched_count, 0.0, 5.0) / 5.0
    exceeds_auto_limit = 1.0 if amount > auto_limit else 0.0
    exceeds_sbv_limit = (
        1.0
        if (
            amount >= float(sbv["single_tx_vnd"])
            or (daily_transferred + amount) >= float(sbv["daily_total_vnd"])
        )
        else 0.0
    )
    is_ocr_source = 1.0 if source_type == "OCR_RECEIPT" else 0.0
    is_late_night = 1.0 if vn_hour in set(sbv["late_night_hours"]) else 0.0
    time_since_last_tx_log = math.log1p(delta_sec) / 15.0

    daily_total_ratio = _clip((daily_transferred + amount) / float(sbv["daily_total_vnd"]), 0.0, 5.0) / 2.0
    balance_drain_ratio = _clip(amount / (balance + 1.0), 0.0, 2.0)
    is_raw_account = 1.0 if source_type == "CHAT_RAW_ACCT" else 0.0
    has_ambiguous_amount = 1.0 if ambiguous else 0.0

    features = {
        "ratio_avg_amount": float(ratio_avg_amount),
        "amount_log": float(amount_log),
        "is_new_recipient": float(is_new_recipient),
        "transfer_count_log": float(transfer_count_log),
        "matched_contact_count": float(matched_contact_count),
        "exceeds_auto_limit": float(exceeds_auto_limit),
        "exceeds_sbv_limit": float(exceeds_sbv_limit),
        "is_ocr_source": float(is_ocr_source),
        "is_late_night": float(is_late_night),
        "time_since_last_tx_log": float(time_since_last_tx_log),
        "daily_total_ratio": float(daily_total_ratio),
        "balance_drain_ratio": float(balance_drain_ratio),
        "is_raw_account": float(is_raw_account),
        "has_ambiguous_amount": float(has_ambiguous_amount),
    }

    for k, v in features.items():
        if not math.isfinite(v):
            raise ValueError(f"Đặc trưng '{k}' có giá trị không hữu hạn: {v}")
    return features


def vectorize_features(
    payload: dict[str, Any],
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    schema: dict[str, Any] | None = None,
) -> np.ndarray:
    cfg = schema or load_feature_schema(schema_path)
    raw = extract_raw_features(payload, schema=cfg)
    ordered_names = [f["name"] for f in cfg["tabular_config"]["features"]]
    vec = np.asarray([raw[name] for name in ordered_names], dtype=np.float32)
    if vec.shape != (int(cfg["tabular_config"]["num_features"]),) or not np.all(np.isfinite(vec)):
        raise ValueError(f"Vector tabular không hợp lệ: shape={vec.shape}, values={vec}")
    return vec


def extract_features_from_db(
    conn: sqlite3.Connection,
    user_id: str,
    amount: int,
    source_type: str,
    contact_id: str | None = None,
    matched_contact_count: int = 0,
    now_utc: datetime | None = None,
    amount_ambiguous: bool | None = None,
    message: str = "",
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
) -> tuple[dict[str, float], np.ndarray]:
    """
    Trích xuất trực tiếp 14 đặc trưng từ SQLite schema T01 (users, contacts, transactions).
    Xử lý chuẩn xác cold-start, UTC -> UTC+7, và tương thích dữ liệu seed.py.
    """
    cfg = load_feature_schema(schema_path)
    row_factory_orig = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        user_row = conn.execute(
            "SELECT balance, auto_approve_limit, daily_transferred FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if user_row is None:
            raise ValueError(f"Không tìm thấy user_id={user_id!r} trong DB.")

        tx_stats = conn.execute(
            """
            SELECT COUNT(*) AS tx_cnt,
                   AVG(amount) AS avg_amt,
                   MAX(completed_at) AS last_at
            FROM transactions
            WHERE sender_id = ?
            """,
            (user_id,),
        ).fetchone()

        trust_level: str | None = None
        effective_transfer_count = 0
        if contact_id is not None:
            c_row = conn.execute(
                """
                SELECT trust_level, transfer_count
                FROM contacts
                WHERE contact_id = ? AND owner_id = ?
                """,
                (contact_id, user_id),
            ).fetchone()
            if c_row is not None:
                trust_level = c_row["trust_level"]
                c_cnt = int(c_row["transfer_count"] or 0)
                ledger_cnt_row = conn.execute(
                    """
                    SELECT COUNT(*) AS cnt
                    FROM transactions
                    WHERE sender_id = ? AND contact_id = ?
                    """,
                    (user_id, contact_id),
                ).fetchone()
                ledger_cnt = int(ledger_cnt_row["cnt"] or 0) if ledger_cnt_row else 0
                effective_transfer_count = max(c_cnt, ledger_cnt)
    finally:
        conn.row_factory = row_factory_orig

    ref_utc = _parse_utc_timestamp(now_utc) or datetime.now(timezone.utc)
    last_dt = _parse_utc_timestamp(tx_stats["last_at"]) if tx_stats and tx_stats["last_at"] else None
    delta_sec: float | None = None
    if last_dt is not None:
        delta_sec = max(0.0, (ref_utc - last_dt).total_seconds())

    user_avg = float(tx_stats["avg_amt"]) if (tx_stats and int(tx_stats["tx_cnt"] or 0) > 0 and tx_stats["avg_amt"]) else None

    payload = {
        "amount": amount,
        "balance": int(user_row["balance"]),
        "auto_approve_limit": int(user_row["auto_approve_limit"]),
        "daily_transferred": int(user_row["daily_transferred"]),
        "user_avg_amount": user_avg,
        "contact_id": contact_id,
        "trust_level": trust_level,
        "recipient_transfer_count": effective_transfer_count,
        "matched_contact_count": matched_contact_count,
        "source_type": source_type,
        "now_utc": ref_utc,
        "time_since_last_tx_seconds": delta_sec,
        "amount_ambiguous": amount_ambiguous,
        "message": message,
    }
    raw = extract_raw_features(payload, schema=cfg)
    vec = vectorize_features(payload, schema=cfg)
    return raw, vec


def prepare_gatekeeper_inputs(
    tokenizer: GatekeeperTokenizer,
    user_message: str,
    tool_name: str,
    note: str | None,
    source_type: str,
    tabular_payload: dict[str, Any],
    add_batch_dim: bool = True,
) -> dict[str, np.ndarray]:
    """Đóng gói trọn bộ tensor đầu vào cho PyTorch (T09) và ONNX Runtime (T12)."""
    tok = tokenizer.encode_pair(
        user_message=user_message,
        tool_name=tool_name,
        note=note,
        source_type=source_type,
    )
    tab = vectorize_features(
        {**tabular_payload, "source_type": source_type, "message": tabular_payload.get("message", user_message)},
        schema=tokenizer.schema,
    )
    out = {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "token_type_ids": tok["token_type_ids"],
        "tabular_features": tab,
    }
    if add_batch_dim:
        return {k: v[None, ...] for k, v in out.items()}
    return out
