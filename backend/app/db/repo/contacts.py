"""
contacts.py — Repo tra cứu contact (zero-PII cho LLM).
Nguồn: tutorial.md mục 2.
"""

import sqlite3
from app.core.text_norm import normalize_text, strip_honorific


def lookup_contact_zero_pii(conn: sqlite3.Connection, owner_id: str, query: str) -> dict:
    """
    Tra cứu contact theo alias query. Trả dữ liệu ẩn danh cho LLM.

    SQL từ tutorial.md mục 2 — không sửa.
    Trả về: {"matches": [{contact_id, alias, bank_code, masked_acct, trust_level}]}
    """
    q_norm = normalize_text(query)
    q_core = strip_honorific(q_norm)

    rows = conn.execute(
        """
        SELECT c.contact_id,
               COALESCE(MAX(CASE WHEN a.alias_norm = :q_norm THEN a.alias_raw END), MAX(a.alias_raw)) AS alias,
               c.bank_code,
               '****' || substr(c.account_number, -3) AS masked_acct,
               c.trust_level
        FROM contacts c JOIN contact_aliases a ON a.contact_id = c.contact_id
        WHERE c.owner_id = :owner_id
          AND (a.alias_norm IN (:q_norm, :q_core) OR a.alias_norm LIKE '% ' || :q_core)
        GROUP BY c.contact_id
        ORDER BY MAX(a.usage_count) DESC
        LIMIT 5
        """,
        {"owner_id": owner_id, "q_norm": q_norm, "q_core": q_core},
    ).fetchall()

    matches = [
        {
            "contact_id": row["contact_id"],
            "alias": row["alias"],
            "bank_code": row["bank_code"],
            "masked_acct": row["masked_acct"],
            "trust_level": row["trust_level"],
        }
        for row in rows
    ]

    return {"matches": matches}
