"""
api.py — FastAPI Sandbox Wallet API tuân thủ hợp đồng backend/openapi/safepay.openapi.json.

Endpoints:
- POST /api/chat
- POST /api/drafts/{tx_id}/confirm
- POST /api/drafts/{tx_id}/cancel
- GET  /api/wallet/state
- POST /api/demo/reset
- GET  /health
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
import re

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.db.connection import get_conn, get_db_path, init_db
from app.db.seed import DEMO_USER_ID, seed_demo
from app.orchestrator.config import OrchestratorConfig, load_backend_env
from app.orchestrator.graph import SafePayOrchestrator
from app.orchestrator.ocr import OcrReceiptService
from app.orchestrator.privacy_guard import PrivacyGuard
from app.wallet.privacy_proxy import PrivacyProxy
from app.wallet.schemas import (
    CancelRequest,
    CancelResponse,
    ChatRequest,
    ChatResponse,
    ClarifyChoice,
    ConfirmRequest,
    ConfirmResponse,
    HealthResponse,
    ResetRequest,
    ResetResponse,
    WalletState,
)
from app.wallet.service import (
    ClarifyContextStore,
    WalletError,
    cancel_draft,
    confirm_draft,
    get_wallet_state,
    handle_chat_message,
    reset_wallet_demo,
)


_LOCAL_SPLIT = re.compile(
    r"chia hóa đơn cho ([1-9][0-9]?) người rồi chuyển cho ([^\r\n]{1,80})", re.IGNORECASE
)


def _read_local_split(req: ChatRequest, ocr: OcrReceiptService) -> ChatRequest:
    """Use local OCR for the explicit sandbox split command when Gemini is inactive."""
    match = _LOCAL_SPLIT.fullmatch(req.message.strip())
    if match is None or not req.image_base64 or req.clarify_choice is not None:
        return req
    if len(req.image_base64) > 7 * 1024 * 1024:
        raise WalletError(422, "VALIDATION_ERROR", "Ảnh hóa đơn vượt quá giới hạn 5 MB.")
    try:
        image_id = ocr.stage_image(req.user_id, req.image_base64)
    except ValueError as exc:
        raise WalletError(422, "VALIDATION_ERROR", "Ảnh hóa đơn không hợp lệ.") from exc

    receipt = ocr.parse_receipt(
        req.user_id, f"{req.user_id}:{req.intent_id}", image_id, require_total_label=True
    )
    total = receipt.get("detected_total_vnd")
    people = int(match[1])
    if not total or total % people:
        raise WalletError(
            422, "VALIDATION_ERROR",
            "Cần một tổng tiền đọc được từ ảnh và chia hết cho số người.",
        )

    return req.model_copy(update={
        "message": f"Chuyển cho {match[2].strip()}",
        "clarify_choice": ClarifyChoice(amount=total // people),
    })


def create_app(
    db_path: str | None = None,
    *,
    seed_if_empty: bool = True,
    orchestrator: SafePayOrchestrator | None = None,
    orchestrator_mode: str | None = None,
) -> FastAPI:
    """Factory tạo FastAPI app (dễ truyền db_path và orchestrator riêng khi chạy test)."""
    load_backend_env()
    resolved_db_path = db_path or get_db_path()
    privacy_proxy = PrivacyProxy()
    clarify_store = ClarifyContextStore()
    privacy_guard = PrivacyGuard(privacy_proxy)
    local_ocr = OcrReceiptService(privacy_guard)
    active_orchestrator = orchestrator or SafePayOrchestrator(
        config=OrchestratorConfig.from_env(mode_override=orchestrator_mode),
        privacy_guard=privacy_guard,
        clarify_store=clarify_store,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        init_db(resolved_db_path)
        if seed_if_empty:
            conn = get_conn(resolved_db_path)
            try:
                cnt = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
                if cnt == 0:
                    seed_demo(conn, DEMO_USER_ID)
            finally:
                conn.close()
        yield

    app = FastAPI(
        title="SafePay Agent API",
        version="0.2.1",
        lifespan=lifespan,
    )

    cors_origins = os.environ.get(
        "SAFEPAY_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in cors_origins if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.db_path = resolved_db_path
    app.state.privacy_proxy = privacy_proxy
    app.state.clarify_store = clarify_store
    app.state.orchestrator = active_orchestrator

    @app.exception_handler(WalletError)
    async def _wallet_error_handler(_request: Request, exc: WalletError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error_code": exc.error_code, "message": exc.message},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(_request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error_code": "VALIDATION_ERROR", "message": str(exc.errors())},
        )

    @app.post("/api/chat", response_model=ChatResponse, operation_id="postChat")
    def post_chat(req: ChatRequest) -> ChatResponse:
        conn = get_conn(resolved_db_path)
        try:
            if active_orchestrator.is_llm_enabled:
                return active_orchestrator.handle_chat(conn, req)
            return handle_chat_message(
                conn,
                _read_local_split(req, local_ocr),
                proxy=privacy_proxy,
                clarify_store=clarify_store,
            )
        finally:
            conn.close()

    @app.post(
        "/api/drafts/{tx_id}/confirm",
        response_model=ConfirmResponse,
        operation_id="confirmDraft",
    )
    def post_confirm_draft(tx_id: str, req: ConfirmRequest) -> ConfirmResponse:
        conn = get_conn(resolved_db_path)
        try:
            return confirm_draft(
                conn,
                tx_id=tx_id,
                user_id=req.user_id,
                pin_code=req.pin_code,
                save_alias=req.save_alias,
            )
        finally:
            conn.close()

    @app.post(
        "/api/drafts/{tx_id}/cancel",
        response_model=CancelResponse,
        operation_id="cancelDraft",
    )
    def post_cancel_draft(tx_id: str, req: CancelRequest) -> CancelResponse:
        conn = get_conn(resolved_db_path)
        try:
            return cancel_draft(conn, tx_id=tx_id, user_id=req.user_id)
        finally:
            conn.close()

    @app.get(
        "/api/wallet/state",
        response_model=WalletState,
        operation_id="getWalletState",
    )
    def get_state(user_id: str = Query(...)) -> WalletState:
        conn = get_conn(resolved_db_path)
        try:
            return get_wallet_state(conn, user_id)
        finally:
            conn.close()

    @app.post(
        "/api/demo/reset",
        response_model=ResetResponse,
        operation_id="resetDemo",
    )
    def post_reset_demo(req: ResetRequest) -> ResetResponse:
        conn = get_conn(resolved_db_path)
        try:
            return reset_wallet_demo(conn, req.user_id)
        finally:
            conn.close()

    @app.get("/health", response_model=HealthResponse, operation_id="getHealth")
    def get_health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            gatekeeper_loaded=active_orchestrator.gatekeeper_loaded,
        )

    return app


app = create_app()
