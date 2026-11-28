"""
test_t03_feature_schema.py — Kiểm thử Feature Schema 14 chiều, biên giá trị,
phát hiện số tiền mơ hồ và tích hợp SQLite schema T01 (seed.py).
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db.connection import get_conn, init_db
from app.db.seed import DEMO_USER_ID, seed_demo
from ml.gatekeeper.preprocessing import (
    DEFAULT_SCHEMA_PATH,
    detect_ambiguous_amount,
    extract_features_from_db,
    extract_raw_features,
    load_feature_schema,
    vectorize_features,
)


class TestFeatureSchemaContract(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = load_feature_schema(DEFAULT_SCHEMA_PATH)

    def test_schema_matches_openapi_and_sql(self) -> None:
        openapi_path = BACKEND_DIR / "openapi" / "safepay.openapi.json"
        schema_sql_path = BACKEND_DIR / "app" / "db" / "schema.sql"
        openapi = json.loads(openapi_path.read_text(encoding="utf-8"))
        sql_text = schema_sql_path.read_text(encoding="utf-8")

        schemas = openapi["components"]["schemas"]
        self.assertEqual(self.schema["labels"], schemas["Decision"]["enum"])
        self.assertEqual(self.schema["source_types"], schemas["SourceType"]["enum"])

        for label in self.schema["labels"]:
            self.assertIn(f"'{label}'", sql_text)
        for src in self.schema["source_types"]:
            self.assertIn(f"'{src}'", sql_text)

        features = self.schema["tabular_config"]["features"]
        self.assertEqual(self.schema["tabular_config"]["num_features"], 14)
        self.assertEqual(len(features), 14)
        self.assertEqual([f["index"] for f in features], list(range(14)))
        self.assertEqual(self.schema["model_budget_spec"]["tabular_input_dim"], 14)
        self.assertEqual(self.schema["model_budget_spec"]["num_classes"], 4)

    def test_boundary_cold_start_and_extreme_values(self) -> None:
        vec = vectorize_features({
            "amount": 100_000_000_000,
            "balance": 0,
            "auto_approve_limit": 100_000,
            "daily_transferred": 0,
            "user_avg_amount": None,
            "contact_id": None,
            "trust_level": None,
            "recipient_transfer_count": 0,
            "matched_contact_count": 0,
            "source_type": "CHAT_RAW_ACCT",
            "vn_local_hour": 14,
            "time_since_last_tx_seconds": None,
            "amount_ambiguous": False,
        })
        self.assertEqual(vec.dtype, np.float32)
        self.assertEqual(vec.shape, (14,))
        self.assertTrue(np.all(np.isfinite(vec)))
        # amount_log clipped at 1.25, balance_drain_ratio clipped at 2.0, is_raw_account == 1.0
        self.assertAlmostEqual(float(vec[1]), 1.25, places=5)
        self.assertAlmostEqual(float(vec[11]), 2.0, places=5)
        self.assertEqual(float(vec[12]), 1.0)

    def test_late_night_utc_to_vn_boundaries(self) -> None:
        cases = [
            (datetime(2026, 11, 25, 15, 59, tzinfo=timezone.utc), 0.0),  # 22:59 VN
            (datetime(2026, 11, 25, 16, 0, tzinfo=timezone.utc), 1.0),   # 23:00 VN
            (datetime(2026, 11, 25, 21, 59, tzinfo=timezone.utc), 1.0),  # 04:59 VN
            (datetime(2026, 11, 25, 22, 0, tzinfo=timezone.utc), 0.0),   # 05:00 VN
        ]
        for now_utc, expected in cases:
            raw = extract_raw_features({
                "amount": 50_000,
                "balance": 1_000_000,
                "source_type": "CHAT_CONTACT",
                "contact_id": "C_001",
                "trust_level": "TRUSTED",
                "now_utc": now_utc,
            })
            self.assertEqual(raw["is_late_night"], expected, f"Failed for {now_utc}")

    def test_sbv_limits_thresholds(self) -> None:
        base = {
            "balance": 50_000_000,
            "source_type": "CHAT_CONTACT",
            "contact_id": "C_001",
            "trust_level": "TRUSTED",
            "vn_local_hour": 10,
        }
        self.assertEqual(extract_raw_features({**base, "amount": 9_999_999, "daily_transferred": 0})["exceeds_sbv_limit"], 0.0)
        self.assertEqual(extract_raw_features({**base, "amount": 10_000_000, "daily_transferred": 0})["exceeds_sbv_limit"], 1.0)
        self.assertEqual(extract_raw_features({**base, "amount": 4_999_999, "daily_transferred": 15_000_000})["exceeds_sbv_limit"], 0.0)
        self.assertEqual(extract_raw_features({**base, "amount": 5_000_000, "daily_transferred": 15_000_000})["exceeds_sbv_limit"], 1.0)

    def test_ambiguous_amount_detector(self) -> None:
        self.assertTrue(detect_ambiguous_amount("chuyển 500 cho Nam"))
        self.assertTrue(detect_ambiguous_amount("Gửi 200 cho Minh nhé"))
        self.assertTrue(detect_ambiguous_amount("bắn 150 cho anh Minh"))
        self.assertTrue(detect_ambiguous_amount("ck 1000 cho sếp"))
        self.assertTrue(detect_ambiguous_amount("ban 350 cho a Tuan"))
        self.assertTrue(detect_ambiguous_amount("Gửi ngay 1500 cho mẹ"))
        self.assertTrue(detect_ambiguous_amount("Bắn ngay 120 cho Thảo"))
        self.assertTrue(detect_ambiguous_amount("chuyen ngay 25 cho Nam"))

        self.assertFalse(detect_ambiguous_amount("chuyển 500k cho Nam"))
        self.assertFalse(detect_ambiguous_amount("chuyển 500.000đ cho Nam"))
        self.assertFalse(detect_ambiguous_amount("chuyển 500.000 cho Nam"))
        self.assertFalse(detect_ambiguous_amount("chuyển 500 nghìn cho Nam"))
        self.assertFalse(detect_ambiguous_amount("chia bill 600k cho 3 người"))
        self.assertFalse(detect_ambiguous_amount("thanh toán cước tháng 11 hết 265k"))
        self.assertFalse(detect_ambiguous_amount("chuyen 50k tien com trua ban 12 cho Nam"))
        self.assertFalse(detect_ambiguous_amount("chuyen 50k ngay 20 cho Nam"))
        self.assertFalse(detect_ambiguous_amount("lưu <ACCT_1> MB là Nam"))
        self.assertFalse(detect_ambiguous_amount("chuyển 100k vào 9001234567"))

    def test_extract_features_from_seeded_sqlite_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = str(Path(tmpdir) / "test_safepay.db")
            init_db(db_file)
            conn = get_conn(db_file)
            try:
                seed_demo(conn, DEMO_USER_ID)
                day_utc = datetime(2026, 11, 25, 7, 0, tzinfo=timezone.utc)  # 14:00 VN

                # C_001: Nam (TRUSTED, 1 seeded completed tx)
                raw_nam, vec_nam = extract_features_from_db(
                    conn,
                    user_id=DEMO_USER_ID,
                    amount=50_000,
                    source_type="CHAT_CONTACT",
                    contact_id="C_001",
                    matched_contact_count=1,
                    now_utc=day_utc,
                    message="chuyển 50k cho thằng Nam",
                )
                self.assertEqual(vec_nam.shape, (14,))
                self.assertEqual(raw_nam["is_new_recipient"], 0.0)
                self.assertGreater(raw_nam["transfer_count_log"], 0.0)
                self.assertEqual(raw_nam["exceeds_auto_limit"], 0.0)
                self.assertEqual(raw_nam["is_raw_account"], 0.0)
                self.assertEqual(raw_nam["is_ocr_source"], 0.0)
                self.assertEqual(raw_nam["has_ambiguous_amount"], 0.0)

                # C_002: sếp Hùng (TRUSTED, 0 tx) -> must still be TRUSTED (is_new_recipient=0)
                raw_sep, _ = extract_features_from_db(
                    conn,
                    user_id=DEMO_USER_ID,
                    amount=500_000,
                    source_type="CHAT_CONTACT",
                    contact_id="C_002",
                    matched_contact_count=1,
                    now_utc=day_utc,
                    message="gửi sếp 500k",
                )
                self.assertEqual(raw_sep["is_new_recipient"], 0.0)
                self.assertEqual(raw_sep["exceeds_auto_limit"], 1.0)

                # C_003: Minh (NEW, ambiguous 2 matches)
                raw_minh, _ = extract_features_from_db(
                    conn,
                    user_id=DEMO_USER_ID,
                    amount=200,
                    source_type="CHAT_CONTACT",
                    contact_id="C_003",
                    matched_contact_count=2,
                    now_utc=day_utc,
                    message="chuyển 200 cho Minh",
                )
                self.assertEqual(raw_minh["is_new_recipient"], 1.0)
                self.assertAlmostEqual(raw_minh["matched_contact_count"], 0.4, places=5)
                self.assertEqual(raw_minh["has_ambiguous_amount"], 1.0)

                # Raw account & OCR receipt
                raw_acct, _ = extract_features_from_db(
                    conn,
                    user_id=DEMO_USER_ID,
                    amount=1_000_000,
                    source_type="CHAT_RAW_ACCT",
                    contact_id=None,
                    now_utc=day_utc,
                    message="chuyển 1tr vào <ACCT_1>",
                )
                self.assertEqual(raw_acct["is_new_recipient"], 1.0)
                self.assertEqual(raw_acct["is_raw_account"], 1.0)
                self.assertEqual(raw_acct["is_ocr_source"], 0.0)

                raw_ocr, _ = extract_features_from_db(
                    conn,
                    user_id=DEMO_USER_ID,
                    amount=450_000,
                    source_type="OCR_RECEIPT",
                    contact_id=None,
                    now_utc=day_utc,
                    message="chia bill hóa đơn 450k",
                )
                self.assertEqual(raw_ocr["is_ocr_source"], 1.0)
                self.assertEqual(raw_ocr["is_raw_account"], 0.0)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
