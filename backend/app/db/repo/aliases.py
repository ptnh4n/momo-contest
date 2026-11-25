"""
aliases.py — Repo thao tác bảng contact_aliases.
Nguồn: tutorial.md mục 0, 2.
"""

import sqlite3
from app.core.text_norm import normalize_text


def upsert_alias(conn: sqlite3.Connection, contact_id: str, alias_raw: str, from_ocr: bool = False) -> None:
    """
    Thêm hoặc cập nhật alias cho contact.

    Luật:
    - from_ocr=True  → raise PermissionError, không ghi gì (NO_OCR_ALIAS).
    - Trùng (contact_id, alias_norm) → chỉ usage_count + 1 (ON CONFLICT DO UPDATE), giữ alias_raw cũ.
    - Cùng alias_norm khác contact → thêm dòng mới (để lookup trả >=2).
    """
    if from_ocr:
        raise PermissionError("NO_OCR_ALIAS: OCR_RECEIPT không được phép ghi alias.")

    alias_norm = normalize_text(alias_raw)

    conn.execute(
        """
        INSERT INTO contact_aliases (contact_id, alias_raw, alias_norm, usage_count)
        VALUES (?, ?, ?, 1)
        ON CONFLICT (contact_id, alias_norm)
        DO UPDATE SET usage_count = usage_count + 1
        """,
        (contact_id, alias_raw, alias_norm),
    )
