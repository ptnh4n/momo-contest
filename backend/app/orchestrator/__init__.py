"""
orchestrator — T05: LLM Orchestrator (LangGraph + Gemini) với Privilege Separation,
Privacy Proxy (Zero-PII), Local RapidOCR và Gatekeeper hook readiness (T12).
"""

from app.orchestrator.config import OrchestratorConfig, load_backend_env
from app.orchestrator.gatekeeper_hook import GatekeeperHook, GatekeeperResult, NoOpGatekeeperHook
from app.orchestrator.graph import SafePayOrchestrator
from app.orchestrator.llm_client import (
    BasePlannerClient,
    GeminiPlannerClient,
    PlannerTurnResult,
    PlannerUnavailableError,
    ToolCall,
)
from app.orchestrator.ocr import ImageStagingStore, OcrReceiptService
from app.orchestrator.pending_alias import PendingAliasItem, PendingAliasStore
from app.orchestrator.privacy_guard import OutboundAuditLog, PiiLeakBlockedError, PrivacyGuard
from app.orchestrator.tools import ALLOWED_LLM_TOOLS, ToolExecutionContext, ToolRegistry

__all__ = [
    "ALLOWED_LLM_TOOLS",
    "BasePlannerClient",
    "GatekeeperHook",
    "GatekeeperResult",
    "GeminiPlannerClient",
    "ImageStagingStore",
    "NoOpGatekeeperHook",
    "OcrReceiptService",
    "OrchestratorConfig",
    "OutboundAuditLog",
    "PendingAliasItem",
    "PendingAliasStore",
    "PiiLeakBlockedError",
    "PlannerTurnResult",
    "PlannerUnavailableError",
    "PrivacyGuard",
    "SafePayOrchestrator",
    "ToolCall",
    "ToolExecutionContext",
    "ToolRegistry",
    "load_backend_env",
]
