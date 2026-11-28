"""
test_t08_dataset.py — Kiểm thử toàn diện tập dữ liệu tổng hợp T08 và dedup_check.py.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.gatekeeper.dedup_check import check_dataset_leakage, load_jsonl, normalize_for_dedup
from ml.gatekeeper.preprocessing import (
    DEFAULT_SCHEMA_PATH,
    DEFAULT_TOKENIZER_PATH,
    GatekeeperTokenizer,
    detect_ambiguous_amount,
    load_feature_schema,
    vectorize_features,
)

T08_DIR = ROOT_DIR / "ml" / "data" / "t08"
PII_RE = re.compile(r"\b0[35789]\d{8}\b|\b\d{9,16}\b")


class TestT08Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train_path = T08_DIR / "train.jsonl"
        cls.val_path = T08_DIR / "val.jsonl"
        cls.report_path = T08_DIR / "dataset_report.json"
        if not (cls.train_path.exists() and cls.val_path.exists() and cls.report_path.exists()):
            raise RuntimeError("Thiếu artifact T08 trong ml/data/t08/. Hãy chạy generate_t08 trước.")

        cls.train_records = load_jsonl(cls.train_path)
        cls.val_records = load_jsonl(cls.val_path)
        cls.all_records = cls.train_records + cls.val_records
        cls.report = json.loads(cls.report_path.read_text(encoding="utf-8"))
        cls.schema = load_feature_schema(DEFAULT_SCHEMA_PATH)
        cls.tokenizer = GatekeeperTokenizer(DEFAULT_TOKENIZER_PATH, DEFAULT_SCHEMA_PATH)

    def test_counts_and_label_ratios(self) -> None:
        self.assertEqual(len(self.train_records), 13600)
        self.assertEqual(len(self.val_records), 2400)
        self.assertEqual(len(self.all_records), 16000)

        expected_train = {"PASS_AUTO": 5440, "REQUIRE_HITL": 3400, "CLARIFY": 2040, "BLOCK_ANOMALY": 2720}
        expected_val = {"PASS_AUTO": 960, "REQUIRE_HITL": 600, "CLARIFY": 360, "BLOCK_ANOMALY": 480}
        self.assertEqual(dict(Counter(r["label"] for r in self.train_records)), expected_train)
        self.assertEqual(dict(Counter(r["label"] for r in self.val_records)), expected_val)

    def test_tabular_zero_skew_and_no_pii(self) -> None:
        valid_tools = set(self.schema["tool_names"])
        valid_sources = set(self.schema["source_types"])
        seen_tools = set()
        seen_sources = set()

        for rec in self.all_records:
            self.assertIn(rec["tool_name"], valid_tools)
            self.assertIn(rec["source_type"], valid_sources)
            seen_tools.add(rec["tool_name"])
            seen_sources.add(rec["source_type"])

            self.assertIsNone(PII_RE.search(rec["user_message"]), f"PII leak in msg: {rec['sample_id']}")
            self.assertIsNone(PII_RE.search(rec["note"]), f"PII leak in note: {rec['sample_id']}")
            self.assertEqual(rec["label_id"], self.schema["labels"].index(rec["label"]))
            self.assertNotIn("message", rec["tabular_raw"])

            expected_vec = vectorize_features(
                {**rec["tabular_raw"], "message": rec["user_message"]},
                schema=self.schema,
            )
            stored_vec = np.asarray(rec["tabular_features"], dtype=np.float32)
            self.assertEqual(stored_vec.shape, (14,))
            self.assertTrue(np.all(np.isfinite(stored_vec)))
            self.assertTrue(
                np.allclose(stored_vec, expected_vec, atol=1e-6),
                f"Tabular mismatch at {rec['sample_id']}",
            )

        self.assertEqual(seen_tools, valid_tools)
        self.assertEqual(seen_sources, valid_sources)

    def test_tokenizer_full_audit(self) -> None:
        seq_lens: list[int] = []
        unk_count = 0
        total_tokens = 0
        at_max_len = 0

        for rec in self.all_records:
            enc = self.tokenizer.encode_pair(
                user_message=rec["user_message"],
                tool_name=rec["tool_name"],
                note=rec["note"],
                source_type=rec["source_type"],
            )
            mask = enc["attention_mask"]
            valid_len = int(mask.sum())
            seq_lens.append(valid_len)
            valid_ids = enc["input_ids"][mask == 1]
            total_tokens += len(valid_ids)
            unk_count += int((valid_ids == self.tokenizer.unk_id).sum())
            if valid_len >= self.tokenizer.max_length:
                at_max_len += 1

        seq_arr = np.asarray(seq_lens, dtype=np.int32)
        unk_rate = unk_count / max(1, total_tokens)
        trunc_rate = at_max_len / len(self.all_records)
        p95 = float(np.percentile(seq_arr, 95))

        self.assertEqual(unk_rate, 0.0)
        self.assertLessEqual(int(seq_arr.max()), 128)
        self.assertLessEqual(p95, 96.0)
        self.assertLess(trunc_rate, 0.01)

    def test_zero_leakage_between_train_and_val(self) -> None:
        train_groups = {r["group_id"] for r in self.train_records}
        val_groups = {r["group_id"] for r in self.val_records}
        self.assertEqual(train_groups & val_groups, set())

        leak_report = check_dataset_leakage(self.train_records, self.val_records)
        self.assertTrue(leak_report["passed"], f"Leakage report failed: {leak_report}")
        self.assertEqual(leak_report["group_overlap_count"], 0)
        self.assertEqual(leak_report["exact_train_val_message_overlap_count"], 0)
        self.assertEqual(leak_report["train_internal_duplicates"], 0)
        self.assertEqual(leak_report["val_internal_duplicates"], 0)

    def test_label_invariants_and_anti_shortcut_properties(self) -> None:
        hitl_late_night = 0
        hitl_high_drain = 0
        hitl_upsert_alias = 0
        hitl_rapid_tx = 0
        hitl_total = 0

        pass_rapid_tx = 0
        pass_total = 0

        clarify_late_night = 0
        clarify_total = 0

        cold_user_avg = 0
        cold_no_prev_tx = 0

        block_day_low_drain = 0
        block_total = 0

        for rec in self.all_records:
            lbl = rec["label"]
            raw = rec["tabular_raw"]
            feat = rec["tabular_features"]
            msg = rec["user_message"]
            risk = float(rec["risk_score"])

            if raw["user_avg_amount"] is None:
                cold_user_avg += 1
            if raw["time_since_last_tx_seconds"] is None:
                cold_no_prev_tx += 1

            if lbl == "PASS_AUTO":
                pass_total += 1
                self.assertLessEqual(risk, 0.15)
                self.assertFalse(detect_ambiguous_amount(msg))
                self.assertEqual(feat[13], 0.0)  # has_ambiguous_amount
                if raw["time_since_last_tx_seconds"] is not None and raw["time_since_last_tx_seconds"] <= 300:
                    pass_rapid_tx += 1
                if rec["tool_name"] == "create_draft_transfer":
                    self.assertEqual(rec["source_type"], "CHAT_CONTACT")
                    self.assertEqual(raw["trust_level"], "TRUSTED")
                    self.assertLessEqual(raw["amount"], raw["auto_approve_limit"])
                    self.assertLess(raw["amount"], 10_000_000)
                    self.assertLess(raw["daily_transferred"] + raw["amount"], 20_000_000)
                    self.assertLessEqual(raw["amount"], raw["balance"])

            elif lbl == "REQUIRE_HITL":
                hitl_total += 1
                self.assertGreaterEqual(risk, 0.18)
                self.assertLessEqual(risk, 0.50)
                self.assertLessEqual(raw["amount"], raw["balance"])
                self.assertFalse(detect_ambiguous_amount(msg))
                self.assertEqual(feat[13], 0.0)

                is_hitl_cond = (
                    rec["source_type"] == "OCR_RECEIPT"
                    or raw["trust_level"] != "TRUSTED"
                    or raw["amount"] > raw["auto_approve_limit"]
                    or raw["amount"] >= 10_000_000
                    or (raw["daily_transferred"] + raw["amount"]) >= 20_000_000
                    or rec["tool_name"] == "upsert_contact_alias"
                )
                self.assertTrue(is_hitl_cond, f"REQUIRE_HITL violated condition: {rec['sample_id']}")
                if feat[8] == 1.0:
                    hitl_late_night += 1
                if feat[11] >= 0.55:
                    hitl_high_drain += 1
                if raw["time_since_last_tx_seconds"] is not None and raw["time_since_last_tx_seconds"] <= 300:
                    hitl_rapid_tx += 1
                if rec["sub_group"] == "legit_upsert_alias":
                    hitl_upsert_alias += 1

            elif lbl == "CLARIFY":
                clarify_total += 1
                if feat[8] == 1.0:
                    clarify_late_night += 1
                self.assertTrue(
                    feat[13] == 1.0 or raw["matched_contact_count"] >= 2,
                    f"CLARIFY violated ambiguity condition: {rec['sample_id']}",
                )
                if feat[13] == 1.0:
                    self.assertTrue(detect_ambiguous_amount(msg))

            elif lbl == "BLOCK_ANOMALY":
                block_total += 1
                self.assertGreaterEqual(risk, 0.75)
                self.assertFalse(detect_ambiguous_amount(msg))
                self.assertEqual(feat[13], 0.0)
                if feat[8] == 0.0 and feat[11] <= 0.55:
                    block_day_low_drain += 1

        self.assertGreaterEqual(block_day_low_drain / block_total, 0.45)
        self.assertGreaterEqual(hitl_late_night / hitl_total, 0.12)
        self.assertGreaterEqual(hitl_high_drain / hitl_total, 0.10)
        self.assertGreater(hitl_upsert_alias, 50)
        self.assertGreaterEqual(pass_rapid_tx / pass_total, 0.10)
        self.assertGreaterEqual(hitl_rapid_tx / hitl_total, 0.10)
        self.assertGreaterEqual(clarify_late_night / clarify_total, 0.05)
        self.assertGreaterEqual(cold_user_avg / len(self.all_records), 0.08)
        self.assertGreaterEqual(cold_no_prev_tx / len(self.all_records), 0.05)

    def test_dedup_check_with_simulated_t07_redteam(self) -> None:
        sample_train = self.train_records[:50]
        sample_val = self.val_records[:20]

        # 1. Exact duplicate + near duplicate (Jaccard >= 0.80)
        exact_dup = {
            "sample_id": "rt_001",
            "user_message": sample_train[0]["user_message"],
        }
        # Giữ phần lớn cụm từ của sample_train[1] để test near-duplicate
        words = normalize_for_dedup(sample_train[1]["user_message"]).split()
        near_msg = " ".join(words)
        near_dup = {"sample_id": "rt_002", "user_message": near_msg}
        disjoint = {
            "sample_id": "rt_003",
            "user_message": "Kich ban hoan toan doc lap khong trung lap voi tap huan luyen safepay nao ca",
        }

        leak_hit = check_dataset_leakage(
            sample_train,
            sample_val,
            external_records=[exact_dup, near_dup, disjoint],
            jaccard_threshold=0.80,
        )
        self.assertFalse(leak_hit["passed"])
        self.assertGreaterEqual(leak_hit["external_exact_match_count"], 1)
        self.assertIn(sample_train[0]["sample_id"], leak_hit["leaked_ids"])

        # 2. Disjoint external set passes cleanly
        leak_clean = check_dataset_leakage(
            sample_train,
            sample_val,
            external_records=[disjoint],
            jaccard_threshold=0.80,
        )
        self.assertTrue(leak_clean["passed"])
        self.assertEqual(leak_clean["leaked_ids"], [])


if __name__ == "__main__":
    unittest.main()
