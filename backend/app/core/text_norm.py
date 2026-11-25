"""
text_norm.py — Chuẩn hóa văn bản tiếng Việt cho SafePay Agent.
Nguồn: tutorial.md mục 2.
"""

import re
import unicodedata

HONORIFICS = ("thang ", "con ", "anh ", "chi ", "em ", "ong ", "ba ", "chu ", "co ", "bac ", "sep ")


def normalize_text(s: str) -> str:
    """'thằng Nam' -> 'thang nam'"""
    s = unicodedata.normalize("NFD", s.strip().lower().replace("đ", "d"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s)


def strip_honorific(n: str) -> str:
    """'thang nam' -> 'nam'; 'sep' đứng một mình giữ nguyên"""
    for h in HONORIFICS:
        if n.startswith(h) and len(n) > len(h):
            return n[len(h):].strip()
    return n
