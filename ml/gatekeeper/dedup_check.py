"""
dedup_check.py — Kiểm tra trùng lặp nội bộ và chống Data Leakage (T08 Train/Val vs T07 Red-Team).

Hỗ trợ gọi như thư viện từ `generate_t08.py` / `test_t08_dataset.py` hoặc chạy CLI độc lập:
    python -m ml.gatekeeper.dedup_check --train ml/data/t08/train.jsonl --val ml/data/t08/val.jsonl
    python -m ml.gatekeeper.dedup_check --train ml/data/t08/train.jsonl --val ml/data/t08/val.jsonl --redteam ml/data/t07/test_redteam.jsonl --purge-leaks
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad\u2060]")
_PUNCT_TO_SPACE_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_for_dedup(text: str | None) -> str:
    """Chuẩn hóa về chữ thường, không dấu tiếng Việt, bỏ dấu câu và khoảng trắng thừa."""
    if not text:
        return ""
    s = _ZERO_WIDTH_RE.sub("", str(text))
    s = s.replace("đ", "d").replace("Đ", "d")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.lower()
    s = _PUNCT_TO_SPACE_RE.sub(" ", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def word_ngrams(text: str | None, n: int = 3) -> set[str]:
    """Tạo tập n-gram từ (mặc định 3-gram; fallback bigram/unigram cho câu ngắn)."""
    norm = normalize_for_dedup(text)
    if not norm:
        return set()
    tokens = norm.split()
    if len(tokens) >= n:
        return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}
    if len(tokens) >= 2:
        return {" ".join(tokens[i : i + 2]) for i in range(len(tokens) - 1)}
    return set(tokens)


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Tính hệ số Jaccard giữa 2 tập n-gram."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def _record_key(rec: dict[str, Any]) -> tuple[str, str, str, str]:
    msg = normalize_for_dedup(rec.get("user_message") or rec.get("message"))
    note = normalize_for_dedup(rec.get("note"))
    src = str(rec.get("source_type", ""))
    tool = str(rec.get("tool_name", ""))
    return (msg, note, src, tool)


def _count_internal_duplicates(records: list[dict[str, Any]]) -> int:
    seen: set[tuple[str, str, str, str]] = set()
    dup = 0
    for rec in records:
        key = _record_key(rec)
        if key in seen:
            dup += 1
        else:
            seen.add(key)
    return dup


def _get_msg(rec: dict[str, Any]) -> str:
    return str(rec.get("user_message") or rec.get("message") or rec.get("text") or "")


def check_dataset_leakage(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    external_records: list[dict[str, Any]] | None = None,
    jaccard_threshold: float = 0.80,
) -> dict[str, Any]:
    """
    Kiểm tra:
    1. Trùng group_id giữa train và val.
    2. Trùng chính xác normalize_for_dedup(user_message) giữa train và val.
    3. Trùng bản ghi nội bộ (msg, note, source_type, tool_name) trong train và trong val.
    4. Nếu có external_records (T07 Red-Team): trùng chính xác và near-duplicate (Jaccard >= ngưỡng).
    """
    train_groups = {str(r["group_id"]) for r in train_records if r.get("group_id")}
    val_groups = {str(r["group_id"]) for r in val_records if r.get("group_id")}
    group_overlap = sorted(train_groups & val_groups)

    train_msgs = {normalize_for_dedup(_get_msg(r)) for r in train_records if _get_msg(r)}
    val_msgs = {normalize_for_dedup(_get_msg(r)) for r in val_records if _get_msg(r)}
    exact_msg_overlap = sorted(train_msgs & val_msgs)

    train_internal_dups = _count_internal_duplicates(train_records)
    val_internal_dups = _count_internal_duplicates(val_records)

    external_exact_matches: list[dict[str, Any]] = []
    external_near_matches: list[dict[str, Any]] = []
    leaked_ids: set[str] = set()

    if external_records:
        ext_items: list[tuple[str, str, set[str]]] = []
        ext_norm_map: dict[str, list[str]] = defaultdict(list)
        ext_ngram_index: dict[str, list[int]] = defaultdict(list)

        for idx, ext in enumerate(external_records):
            ext_id = str(ext.get("sample_id") or ext.get("id") or f"ext_{idx}")
            norm_msg = normalize_for_dedup(_get_msg(ext))
            if not norm_msg:
                continue
            grams = word_ngrams(norm_msg, n=3)
            ext_items.append((ext_id, norm_msg, grams))
            ext_norm_map[norm_msg].append(ext_id)
            cur_idx = len(ext_items) - 1
            for g in grams:
                ext_ngram_index[g].append(cur_idx)

        for split_name, recs in (("train", train_records), ("val", val_records)):
            for r_idx, rec in enumerate(recs):
                sid = str(rec.get("sample_id") or f"{split_name}_{r_idx}")
                norm_msg = normalize_for_dedup(_get_msg(rec))
                if not norm_msg:
                    continue
                if norm_msg in ext_norm_map:
                    leaked_ids.add(sid)
                    external_exact_matches.append(
                        {
                            "split": split_name,
                            "sample_id": sid,
                            "external_ids": ext_norm_map[norm_msg],
                            "user_message": _get_msg(rec),
                        }
                    )
                    continue

                grams = word_ngrams(norm_msg, n=3)
                cand_indices: set[int] = set()
                for g in grams:
                    cand_indices.update(ext_ngram_index.get(g, ()))

                best_sim = 0.0
                best_ext_id: str | None = None
                for c_idx in cand_indices:
                    ext_id, _, ext_grams = ext_items[c_idx]
                    sim = jaccard_similarity(grams, ext_grams)
                    if sim >= jaccard_threshold and sim > best_sim:
                        best_sim = sim
                        best_ext_id = ext_id

                if best_ext_id is not None:
                    leaked_ids.add(sid)
                    external_near_matches.append(
                        {
                            "split": split_name,
                            "sample_id": sid,
                            "external_id": best_ext_id,
                            "jaccard": round(best_sim, 4),
                            "user_message": _get_msg(rec),
                        }
                    )

    passed = (
        len(group_overlap) == 0
        and len(exact_msg_overlap) == 0
        and train_internal_dups == 0
        and val_internal_dups == 0
        and len(leaked_ids) == 0
    )

    return {
        "passed": passed,
        "train_count": len(train_records),
        "val_count": len(val_records),
        "external_count": len(external_records) if external_records else 0,
        "group_overlap_count": len(group_overlap),
        "group_overlap_samples": group_overlap[:10],
        "exact_train_val_message_overlap_count": len(exact_msg_overlap),
        "exact_train_val_message_overlap_samples": exact_msg_overlap[:10],
        "train_internal_duplicates": train_internal_dups,
        "val_internal_duplicates": val_internal_dups,
        "external_exact_match_count": len(external_exact_matches),
        "external_near_match_count": len(external_near_matches),
        "leaked_ids": sorted(leaked_ids),
        "external_exact_matches": external_exact_matches[:20],
        "external_near_matches": external_near_matches[:20],
    }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra trùng lặp và data leakage cho T08.")
    parser.add_argument("--train", required=True, help="Đường dẫn tới train.jsonl")
    parser.add_argument("--val", required=True, help="Đường dẫn tới val.jsonl")
    parser.add_argument("--redteam", default=None, help="Đường dẫn tùy chọn tới T07 test_redteam.jsonl")
    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=0.80,
        help="Ngưỡng Jaccard 3-gram coi là near-duplicate với T07 (mặc định 0.80)",
    )
    parser.add_argument(
        "--purge-leaks",
        action="store_true",
        help="Nếu phát hiện trùng với T07, tự động lọc bỏ khỏi train/val",
    )
    parser.add_argument("--report-out", default=None, help="Tùy chọn lưu kết quả JSON")
    args = parser.parse_args(argv)

    train_records = load_jsonl(args.train)
    val_records = load_jsonl(args.val)
    ext_records = load_jsonl(args.redteam) if args.redteam else None

    report = check_dataset_leakage(
        train_records,
        val_records,
        external_records=ext_records,
        jaccard_threshold=args.jaccard_threshold,
    )

    if args.purge_leaks and report["leaked_ids"]:
        leak_set = set(report["leaked_ids"])
        train_records = [r for r in train_records if r.get("sample_id") not in leak_set]
        val_records = [r for r in val_records if r.get("sample_id") not in leak_set]
        save_jsonl(args.train, train_records)
        save_jsonl(args.val, val_records)
        report = check_dataset_leakage(
            train_records,
            val_records,
            external_records=ext_records,
            jaccard_threshold=args.jaccard_threshold,
        )
        report["purged_leak_count"] = len(leak_set)

    if args.report_out:
        Path(args.report_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps({k: v for k, v in report.items() if not k.endswith("_matches")}, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
