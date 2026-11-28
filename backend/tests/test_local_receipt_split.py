"""Regression tests for the explicit, offline receipt-split flow."""
import base64
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from app.wallet.api import create_app

RECEIPT = Path(__file__).resolve().parents[2] / "qa/receipts/ocr-006.png"


def test_local_ocr_split_uses_receipt_total_and_needs_confirmation(tmp_path):
    app = create_app(db_path=str(tmp_path / "wallet.sqlite"), orchestrator_mode="deterministic")
    with TestClient(app) as client:
        data = {
            "user_id": "u_demo",
            "intent_id": str(uuid4()),
            "message": "Chia hóa đơn cho 4 người rồi chuyển cho thằng Nam",
            "image_base64": base64.b64encode(RECEIPT.read_bytes()).decode(),
        }
        response = client.post("/api/chat", json=data)
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["decision"] == "REQUIRE_HITL"
        assert payload["draft"]["amount"] == 100_000
        assert payload["draft"]["source_type"] == "OCR_RECEIPT"
        assert payload["draft"]["suggested_alias"] is None

        repeated = client.post("/api/chat", json=data).json()
        assert repeated["draft"]["tx_id"] == payload["draft"]["tx_id"]
        state = client.get("/api/wallet/state", params={"user_id": "u_demo"}).json()
        assert state["balance"] == 10_000_000

        data["intent_id"] = str(uuid4())
        data["image_base64"] = "invalid base64"
        assert client.post("/api/chat", json=data).status_code == 422
