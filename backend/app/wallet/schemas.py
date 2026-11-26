"""
schemas.py — Pydantic v2 schemas khớp hợp đồng backend/openapi/safepay.openapi.json.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID
from pydantic import BaseModel, Field

Decision = Literal["PASS_AUTO", "REQUIRE_HITL", "CLARIFY", "BLOCK_ANOMALY"]
SourceType = Literal["CHAT_CONTACT", "CHAT_RAW_ACCT", "OCR_RECEIPT"]
DraftStatus = Literal["DRAFT", "COMPLETED", "CANCELLED", "BLOCKED"]
TrustLevel = Literal["NEW", "TRUSTED"]
ErrorCode = Literal[
    "INVALID_PIN",
    "NOT_FOUND",
    "DRAFT_NOT_CONFIRMABLE",
    "BLOCKED_BY_BLACKLIST",
    "INSUFFICIENT_BALANCE",
    "VALIDATION_ERROR",
]


class ClarifyChoice(BaseModel):
    ref_intent_id: UUID | None = None
    contact_id: str | None = None
    amount: int | None = Field(default=None, ge=1)


class ChatRequest(BaseModel):
    user_id: str
    intent_id: UUID
    message: str
    image_base64: str | None = None
    clarify_choice: ClarifyChoice | None = None


class DraftView(BaseModel):
    tx_id: str
    recipient_name: str
    recipient_account: str
    bank_code: str
    amount: int = Field(ge=1)
    note: str | None = None
    source_type: SourceType
    suggested_alias: str | None = None
    status: DraftStatus


class ClarifyOptionItem(BaseModel):
    label: str
    contact_id: str | None = None
    amount: int | None = None


class ClarifyOptions(BaseModel):
    kind: Literal["CONTACT", "CURRENCY_UNIT"]
    options: list[ClarifyOptionItem] = Field(min_length=2)


class ChatResponse(BaseModel):
    reply_text: str
    decision: Decision
    risk_score: float = Field(ge=0.0, le=1.0)
    xai_reasons: list[str]
    draft: DraftView | None = None
    clarify_options: ClarifyOptions | None = None
    new_balance: int | None = None


class ConfirmRequest(BaseModel):
    user_id: str
    pin_code: str = Field(pattern=r"^[0-9]{4,8}$")
    save_alias: bool = False


class ConfirmResponse(BaseModel):
    status: Literal["COMPLETED"] = "COMPLETED"
    new_balance: int
    tx_id: str
    alias_saved: str | None = None


class CancelRequest(BaseModel):
    user_id: str


class CancelResponse(BaseModel):
    status: Literal["CANCELLED"] = "CANCELLED"
    tx_id: str


class ContactView(BaseModel):
    contact_id: str
    full_name: str
    bank_code: str
    account_number: str
    trust_level: TrustLevel
    aliases: list[str]


class TransactionView(BaseModel):
    tx_id: str
    recipient_name: str
    bank_code: str
    amount: int
    note: str | None = None
    completed_at: str


class WalletState(BaseModel):
    balance: int
    daily_transferred: int
    auto_approve_limit: int
    contacts: list[ContactView]
    recent_transactions: list[TransactionView]


class ResetRequest(BaseModel):
    user_id: str


class ResetResponse(BaseModel):
    status: Literal["RESET_OK"] = "RESET_OK"
    balance: int


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    gatekeeper_loaded: bool = False


class ErrorResponse(BaseModel):
    error_code: ErrorCode
    message: str
