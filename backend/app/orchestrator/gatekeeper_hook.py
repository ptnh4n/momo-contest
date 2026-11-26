"""
gatekeeper_hook.py — Giao diện sẵn sàng gắn Gatekeeper ONNX (<20M params) ở T12.

Trong T05 dùng `NoOpGatekeeperHook` (`is_loaded=False`).
Khi T12 hoàn tất, chỉ cần truyền `OnnxGatekeeperHook` triển khai `GatekeeperHook`
mà không cần sửa luồng LangGraph hay API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import sqlite3
from typing import Protocol

from app.wallet.guardrails import Decision, SourceType
from app.wallet.schemas import ChatResponse


@dataclass(frozen=True)
class GatekeeperResult:
    decision: Decision
    risk_score: float
    xai_reasons: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


class GatekeeperHook(Protocol):
    @property
    def is_loaded(self) -> bool: ...

    def evaluate(
        self,
        conn: sqlite3.Connection,
        *,
        user_id: str,
        masked_message: str,
        tool_name: str,
        note: str | None,
        source_type: SourceType,
        amount: int,
        contact_id: str | None,
        matched_contact_count: int = 1,
    ) -> GatekeeperResult | None: ...


class NoOpGatekeeperHook:
    """Hook mặc định cho T05 (chưa gắn model T12)."""

    @property
    def is_loaded(self) -> bool:
        return False

    def evaluate(
        self,
        conn: sqlite3.Connection,
        *,
        user_id: str,
        masked_message: str,
        tool_name: str,
        note: str | None,
        source_type: SourceType,
        amount: int,
        contact_id: str | None,
        matched_contact_count: int = 1,
    ) -> GatekeeperResult | None:
        return None


_SEVERITY = {
    "PASS_AUTO": 0,
    "CLARIFY": 1,
    "REQUIRE_HITL": 2,
    "BLOCK_ANOMALY": 3,
}


def merge_guardrail_and_gatekeeper(
    hard_resp: ChatResponse,
    gk_res: GatekeeperResult | None,
    *,
    is_ocr: bool,
    blocked_by_blacklist: bool,
) -> ChatResponse:
    """
    Hợp nhất bảo thủ giữa Hard Guardrails (T04) và Gatekeeper (T12):
    - Gatekeeper KHÔNG BAO GIỜ được hạ cấp quyết định cứng (Blacklist, OCR_ALWAYS_HITL, NHNN).
    - Gatekeeper có quyền nâng cấp rủi ro lên REQUIRE_HITL hoặc BLOCK_ANOMALY kèm XAI.
    """
    if gk_res is None:
        return hard_resp
    if blocked_by_blacklist:
        return hard_resp

    merged_reasons = list(dict.fromkeys([*hard_resp.xai_reasons, *gk_res.xai_reasons]))
    merged_risk = max(float(hard_resp.risk_score), float(gk_res.risk_score))

    hard_sev = _SEVERITY.get(hard_resp.decision, 2)
    gk_sev = _SEVERITY.get(gk_res.decision, 0)
    final_decision = gk_res.decision if gk_sev > hard_sev else hard_resp.decision
    if is_ocr and final_decision == "PASS_AUTO":
        final_decision = "REQUIRE_HITL"

    draft = hard_resp.draft
    if draft is not None and final_decision == "BLOCK_ANOMALY":
        draft = draft.model_copy(update={"status": "BLOCKED"})

    return hard_resp.model_copy(
        update={
            "decision": final_decision,
            "risk_score": min(1.0, max(0.0, merged_risk)),
            "xai_reasons": merged_reasons,
            "draft": draft,
        }
    )
