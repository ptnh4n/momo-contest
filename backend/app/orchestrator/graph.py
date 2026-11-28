"""
graph.py — LangGraph StateGraph Orchestrator (6 bước chuẩn CONTEXT.md + Privilege Separation).

Luồng nút:
  privacy_ingress -> pre_guardrail -> llm_planner <-> tool_executor -> post_checkpoint -> finalize
"""

import logging
import re
import sqlite3
from typing import Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

from app.db.connection import transaction
from app.db.repo.aliases import upsert_alias
from app.orchestrator.config import OrchestratorConfig
from app.orchestrator.gatekeeper_hook import (
    GatekeeperHook,
    NoOpGatekeeperHook,
    merge_guardrail_and_gatekeeper,
)
from app.orchestrator.llm_client import (
    BasePlannerClient,
    GeminiPlannerClient,
    PlannerUnavailableError,
    ToolCall,
)
from app.orchestrator.ocr import OcrReceiptService
from app.orchestrator.pending_alias import (
    PendingAliasStore,
    is_affirmative_reply,
    is_negative_reply,
)
from app.orchestrator.privacy_guard import OutboundAuditLog, PiiLeakBlockedError, PrivacyGuard
from app.orchestrator.tools import (
    ToolExecutionContext,
    ToolRegistry,
    build_gemini_tool_declarations,
)
from app.wallet.currency_parser import parse_currency
from app.wallet.guardrails import SourceType, check_scam_blacklist, enforce_no_ocr_alias
from app.wallet.schemas import (
    ChatRequest,
    ChatResponse,
    ClarifyOptionItem,
    ClarifyOptions,
)
from app.wallet.service import (
    ClarifyContextStore,
    WalletError,
    _PendingClarify,
    _build_response_from_existing_draft,
    _extract_bank_code,
    _extract_recipient_query,
    create_draft_transfer,
    lookup_contact,
)

# Cụm từ chỉ ý định chuyển tiền (dùng để kích hoạt chặn số tiền mơ hồ kiểu "chuyển 500 cho Nam")
_TRANSFER_INTENT_RE = re.compile(
    r"\b(?:chuyển|chuyen|ck|gửi|gui|bắn|ban|trả|tra|nộp|nop|đóng|dong|thanh\s*toán|thanh\s*toan)\b",
    re.IGNORECASE,
)
# Loại bỏ cụm số phi tiền tệ ("4 người", "3 bạn", "tháng 11", "bàn 5") trước khi kiểm tra currency mơ hồ
_NON_MONEY_PRE_RE = re.compile(
    r"(?i)\b(?:tháng|thang|ngày|ngay|năm|nam|bàn|ban|lần|lan|mức|muc|quận|quan|"
    r"phòng|phong|tầng|tang|lớp|lop|nhóm|nhom|top|mã|ma|số|so)\s+\d{1,4}\b"
)
_NON_MONEY_POST_RE = re.compile(
    r"(?i)\b\d{1,4}\s*(?:người|nguoi|bạn|ban|đứa|dua|phút|phut|giờ|gio|"
    r"ngày|ngay|tháng|thang|tuần|tuan|năm|nam|món|mon|ly|cốc|coc|phần|phan|suất|suat|%)\b"
)


class AgentState(TypedDict, total=False):
    req: ChatRequest
    intent_id: str
    session_id: str
    masked_message: str
    acct_token: str | None
    has_image: bool
    image_id: str | None
    prior_ctx: _PendingClarify | None
    history: list[dict[str, Any]]
    pending_tool_calls: list[ToolCall]
    llm_final_text: str
    step_count: int
    tool_ctx: ToolExecutionContext
    short_circuit: bool
    final_response: ChatResponse | None


def _strip_non_money_counts(text: str) -> str:
    cleaned = _NON_MONEY_PRE_RE.sub(" ", text)
    cleaned = _NON_MONEY_POST_RE.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _collect_user_forbidden_pii(conn: sqlite3.Connection, user_id: str) -> set[str]:
    """Thu thập STK danh bạ thật của user để đảm bảo không bao giờ lọt ra Gemini."""
    rows = conn.execute(
        "SELECT account_number FROM contacts WHERE owner_id = ?",
        (user_id,),
    ).fetchall()
    return {str(r["account_number"]) for r in rows if r["account_number"]}


class SafePayOrchestrator:
    """
    Facade điều phối LangGraph + Gemini + PrivacyGuard + RapidOCR + Hard Guardrails + Gatekeeper Hook.
    """

    def __init__(
        self,
        *,
        config: OrchestratorConfig,
        privacy_guard: PrivacyGuard,
        clarify_store: ClarifyContextStore,
        pending_alias_store: PendingAliasStore | None = None,
        ocr_service: OcrReceiptService | None = None,
        planner_client: BasePlannerClient | None = None,
        gatekeeper_hook: GatekeeperHook | None = None,
        audit_log: OutboundAuditLog | None = None,
    ) -> None:
        self.config = config
        self.privacy_guard = privacy_guard
        self.clarify_store = clarify_store
        self.pending_alias_store = pending_alias_store or PendingAliasStore()
        self.ocr_service = ocr_service or OcrReceiptService(privacy_guard=privacy_guard)
        self.audit_log = audit_log or OutboundAuditLog()
        self.gatekeeper_hook: GatekeeperHook = gatekeeper_hook or NoOpGatekeeperHook()
        self.tool_registry = ToolRegistry()
        self.tool_declarations = build_gemini_tool_declarations()

        if planner_client is not None:
            self.planner_client: BasePlannerClient | None = planner_client
        elif config.mode != "deterministic" and config.has_api_key:
            self.planner_client = GeminiPlannerClient(
                config=config,
                privacy_guard=privacy_guard,
                audit_log=self.audit_log,
            )
        else:
            self.planner_client = None

        self._graph = self._build_graph()

    @property
    def is_llm_enabled(self) -> bool:
        return self.config.mode != "deterministic" and self.planner_client is not None

    @property
    def gatekeeper_loaded(self) -> bool:
        return bool(self.gatekeeper_hook.is_loaded)

    def handle_chat(self, conn: sqlite3.Connection, req: ChatRequest) -> ChatResponse:
        if self.planner_client is None:
            raise PlannerUnavailableError("Orchestrator chưa cấu hình LLM Planner Client.")

        initial_state: AgentState = {
            "req": req,
            "intent_id": str(req.intent_id),
            "step_count": 0,
            "short_circuit": False,
            "history": [],
            "pending_tool_calls": [],
            "llm_final_text": "",
            "final_response": None,
        }
        # Truyền conn qua config["configurable"] để không đưa sqlite3.Connection vào state serialization
        out_state = self._graph.invoke(
            initial_state,
            config={"configurable": {"conn": conn}},
        )
        resp = out_state.get("final_response")
        if resp is None:
            raise RuntimeError("Orchestrator graph kết thúc mà không tạo final_response.")
        return resp

    def _build_graph(self):
        builder = StateGraph(AgentState)

        def privacy_ingress(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
            conn: sqlite3.Connection = config["configurable"]["conn"]
            req = state["req"]
            intent_str = state["intent_id"]

            # (1) Idempotency Key check — trả draft cũ ngay, không gọi LLM
            existing = conn.execute(
                "SELECT * FROM draft_transactions WHERE intent_id = ?",
                (intent_str,),
            ).fetchone()
            if existing is not None:
                return {
                    "short_circuit": True,
                    "final_response": _build_response_from_existing_draft(
                        conn, existing, req.user_id
                    ),
                }

            user = conn.execute(
                "SELECT user_id FROM users WHERE user_id = ?", (req.user_id,)
            ).fetchone()
            if user is None:
                raise WalletError(
                    404, "NOT_FOUND", f"Không tìm thấy người dùng '{req.user_id}'."
                )

            prior_ctx: _PendingClarify | None = None
            if req.clarify_choice and req.clarify_choice.ref_intent_id is not None:
                prior_ctx = self.clarify_store.pop(str(req.clarify_choice.ref_intent_id))

            session_id = prior_ctx.session_id if prior_ctx else f"{req.user_id}:{intent_str}"
            mask_res = self.privacy_guard.mask_text(session_id, req.message)
            acct_token = next(
                (tok for tok, kind in mask_res.tokens.items() if kind == "ACCT"),
                prior_ctx.acct_token if prior_ctx else None,
            )

            has_image = bool(req.image_base64)
            image_id: str | None = None
            if has_image and req.image_base64 is not None:
                try:
                    image_id = self.ocr_service.stage_image(req.user_id, req.image_base64)
                except ValueError as exc:
                    raise WalletError(422, "VALIDATION_ERROR", str(exc)) from exc

            tool_ctx = ToolExecutionContext(
                conn=conn,
                user_id=req.user_id,
                intent_id=intent_str,
                session_id=session_id,
                masked_user_message=mask_res.masked_text,
                has_image=has_image,
                image_id=image_id,
                privacy_guard=self.privacy_guard,
                clarify_store=self.clarify_store,
                pending_alias_store=self.pending_alias_store,
                ocr_service=self.ocr_service,
                resolved_by_clarify_contact_id=(
                    req.clarify_choice.contact_id if req.clarify_choice else None
                ),
            )

            return {
                "session_id": session_id,
                "masked_message": mask_res.masked_text,
                "acct_token": acct_token,
                "has_image": has_image,
                "image_id": image_id,
                "prior_ctx": prior_ctx,
                "tool_ctx": tool_ctx,
            }

        def pre_guardrail(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
            conn: sqlite3.Connection = config["configurable"]["conn"]
            req = state["req"]
            intent_str = state["intent_id"]
            session_id = state["session_id"]
            masked_text = state["masked_message"]
            acct_token = state["acct_token"]
            has_image = state["has_image"]
            image_id = state["image_id"]
            prior_ctx = state.get("prior_ctx")
            tool_ctx = state["tool_ctx"]

            # (A) Option B: Lượt 2 xác nhận lưu alias cho STK mới
            if not has_image and req.clarify_choice is None:
                pending = self.pending_alias_store.get(req.user_id)
                if pending is not None:
                    if is_affirmative_reply(req.message):
                        self.pending_alias_store.pop(req.user_id)
                        bl = check_scam_blacklist(conn, pending.raw_account, pending.bank_code)
                        if bl.is_blacklisted:
                            reason = f"BLACKLIST_HARD_VETO: Tài khoản nằm trong danh sách đen ({bl.scam_type})."
                            return {
                                "short_circuit": True,
                                "final_response": ChatResponse(
                                    reply_text=f"Từ chối lưu danh bạ vì tài khoản thuộc danh sách đen ({bl.scam_type}).",
                                    decision="BLOCK_ANOMALY",
                                    risk_score=1.0,
                                    xai_reasons=[reason],
                                    draft=None,
                                    clarify_options=None,
                                    new_balance=None,
                                ),
                            }
                        with transaction(conn):
                            found = conn.execute(
                                """
                                SELECT contact_id FROM contacts
                                WHERE owner_id = ? AND account_number = ? AND UPPER(bank_code) = UPPER(?)
                                """,
                                (req.user_id, pending.raw_account, pending.bank_code),
                            ).fetchone()
                            if found is not None:
                                target_cid = str(found["contact_id"])
                            else:
                                from uuid import uuid4

                                target_cid = f"C_{uuid4().hex[:8]}"
                                conn.execute(
                                    """
                                    INSERT INTO contacts
                                        (contact_id, owner_id, full_name, account_number, bank_code, trust_level)
                                    VALUES (?, ?, ?, ?, ?, 'NEW')
                                    """,
                                    (
                                        target_cid,
                                        req.user_id,
                                        pending.alias_raw,
                                        pending.raw_account,
                                        pending.bank_code,
                                    ),
                                )
                            upsert_alias(conn, target_cid, pending.alias_raw, from_ocr=False)

                        return {
                            "short_circuit": True,
                            "final_response": ChatResponse(
                                reply_text=(
                                    f"Đã lưu biệt danh '{pending.alias_raw}' cho tài khoản "
                                    f"{pending.masked_acct} ({pending.bank_code})."
                                ),
                                decision="PASS_AUTO",
                                risk_score=0.05,
                                xai_reasons=[],
                                draft=None,
                                clarify_options=None,
                                new_balance=None,
                            ),
                        }
                    if is_negative_reply(req.message):
                        self.pending_alias_store.pop(req.user_id)
                        return {
                            "short_circuit": True,
                            "final_response": ChatResponse(
                                reply_text="Đã hủy thao tác lưu biệt danh mới.",
                                decision="CLARIFY",
                                risk_score=0.0,
                                xai_reasons=[
                                    "USER_CANCELLED_ALIAS: Người dùng đã hủy thao tác lưu biệt danh."
                                ],
                                draft=None,
                                clarify_options=None,
                                new_balance=None,
                            ),
                        }
                    # Người dùng gửi lệnh khác -> bỏ pending cũ và xử lý lệnh mới bình thường
                    self.pending_alias_store.pop(req.user_id)

            # (B) Xử lý phản hồi clarify_choice từ lượt CLARIFY trước (xác định 100%, không cần gọi LLM)
            if prior_ctx is not None and req.clarify_choice is not None:
                resolved_amount = (
                    req.clarify_choice.amount
                    if req.clarify_choice.amount is not None
                    else prior_ctx.amount
                )
                resolved_cid = req.clarify_choice.contact_id
                if resolved_amount is not None and (resolved_cid or prior_ctx.acct_token):
                    suggested_alias = enforce_no_ocr_alias(
                        prior_ctx.source_type,
                        prior_ctx.recipient_query if resolved_cid else prior_ctx.suggested_alias,
                    )
                    resp = create_draft_transfer(
                        conn,
                        intent_id=intent_str,
                        sender_id=req.user_id,
                        amount=resolved_amount,
                        source_type=prior_ctx.source_type,
                        bank_code=prior_ctx.bank_code,
                        recipient_account=prior_ctx.acct_token if resolved_cid is None else None,
                        contact_id=resolved_cid,
                        note=prior_ctx.masked_message,
                        suggested_alias=suggested_alias,
                        proxy=self.privacy_guard.proxy,
                        session_id=session_id,
                    )
                    tool_ctx.draft_response = resp
                    tool_ctx.last_contact_id = resolved_cid
                    return {"short_circuit": False, "final_response": None, "tool_ctx": tool_ctx}

                if resolved_amount is not None and prior_ctx.recipient_query:
                    matches = lookup_contact(conn, req.user_id, prior_ctx.recipient_query)["matches"]
                    if len(matches) == 1:
                        resp = create_draft_transfer(
                            conn,
                            intent_id=intent_str,
                            sender_id=req.user_id,
                            amount=resolved_amount,
                            source_type=prior_ctx.source_type,
                            bank_code=prior_ctx.bank_code,
                            contact_id=matches[0]["contact_id"],
                            note=prior_ctx.masked_message,
                            suggested_alias=prior_ctx.suggested_alias,
                            proxy=self.privacy_guard.proxy,
                            session_id=session_id,
                        )
                        tool_ctx.draft_response = resp
                        tool_ctx.last_contact_id = matches[0]["contact_id"]
                        tool_ctx.last_matched_count = 1
                        return {"short_circuit": False, "final_response": None, "tool_ctx": tool_ctx}
                    if len(matches) >= 2:
                        self.clarify_store.put(
                            intent_str,
                            _PendingClarify(
                                user_id=req.user_id,
                                masked_message=prior_ctx.masked_message,
                                source_type=prior_ctx.source_type,
                                amount=resolved_amount,
                                recipient_query=prior_ctx.recipient_query,
                                acct_token=None,
                                bank_code=prior_ctx.bank_code,
                                suggested_alias=None,
                                session_id=session_id,
                                expires_at=0.0,
                            ),
                        )
                        options = [
                            ClarifyOptionItem(
                                label=f"{m['alias']} — {m['bank_code']} {m['masked_acct']}",
                                contact_id=m["contact_id"],
                            )
                            for m in matches
                        ]
                        return {
                            "short_circuit": True,
                            "final_response": ChatResponse(
                                reply_text="Tìm thấy nhiều người nhận trùng tên. Vui lòng chọn đúng người.",
                                decision="CLARIFY",
                                risk_score=0.0,
                                xai_reasons=["AMBIGUOUS_CONTACT: Có từ 2 danh bạ khớp từ khóa."],
                                draft=None,
                                clarify_options=ClarifyOptions(kind="CONTACT", options=options),
                                new_balance=None,
                            ),
                        }

            # (C) Deterministic Currency Parser chặn số mơ hồ ("500") hoặc <= 0 trước khi gọi LLM
            if not has_image and req.clarify_choice is None:
                cur_scan = _strip_non_money_counts(masked_text)
                cur_res = parse_currency(cur_scan)
                if cur_res.status == "INVALID":
                    raise WalletError(
                        422, "VALIDATION_ERROR", cur_res.reason or "Số tiền không hợp lệ."
                    )
                if cur_res.status == "AMBIGUOUS" and _TRANSFER_INTENT_RE.search(masked_text):
                    bank_code = _extract_bank_code(masked_text, default="MB")
                    recipient_query = _extract_recipient_query(masked_text)
                    suggested_alias = enforce_no_ocr_alias(
                        "CHAT_RAW_ACCT" if acct_token is not None else "CHAT_CONTACT",
                        recipient_query if acct_token is not None else None,
                    )
                    self.clarify_store.put(
                        intent_str,
                        _PendingClarify(
                            user_id=req.user_id,
                            masked_message=masked_text,
                            source_type="CHAT_RAW_ACCT" if acct_token is not None else "CHAT_CONTACT",
                            amount=None,
                            recipient_query=recipient_query,
                            acct_token=acct_token,
                            bank_code=bank_code,
                            suggested_alias=suggested_alias,
                            session_id=session_id,
                            expires_at=0.0,
                        ),
                    )
                    return {
                        "short_circuit": True,
                        "final_response": ChatResponse(
                            reply_text=cur_res.reason or "Vui lòng xác nhận đơn vị tiền tệ.",
                            decision="CLARIFY",
                            risk_score=0.0,
                            xai_reasons=[
                                "AMBIGUOUS_CURRENCY_UNIT: Số tiền không kèm đơn vị rõ ràng."
                            ],
                            draft=None,
                            clarify_options=ClarifyOptions(
                                kind="CURRENCY_UNIT",
                                options=[
                                    ClarifyOptionItem(label=opt.label, amount=opt.amount)
                                    for opt in cur_res.options
                                ],
                            ),
                            new_balance=None,
                        ),
                    }

            # (D) Chuẩn bị prompt ban đầu (100% Zero-PII)
            if has_image and image_id:
                user_prompt = (
                    f"{masked_text}\n"
                    f'[IMAGE_ATTACHED: image_id="{image_id}", source_type="OCR_RECEIPT"]\n'
                    '[CONTEXT: intent_id="CURRENT_TURN"]'
                )
            else:
                user_prompt = f'{masked_text}\n[CONTEXT: intent_id="CURRENT_TURN"]'

            return {
                "history": [{"role": "user", "text": user_prompt}],
                "short_circuit": False,
            }

        def llm_planner(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
            conn: sqlite3.Connection = config["configurable"]["conn"]
            tool_ctx = state["tool_ctx"]
            # Nếu pre_guardrail đã tạo draft_response từ clarify_choice, bỏ qua LLM
            if tool_ctx.draft_response is not None:
                return {"pending_tool_calls": [], "llm_final_text": ""}

            history = list(state.get("history") or [])
            extra_forbidden = _collect_user_forbidden_pii(conn, tool_ctx.user_id)

            assert self.planner_client is not None
            try:
                turn = self.planner_client.plan_step(
                    session_id=state["session_id"],
                    history=history,
                    tool_declarations=self.tool_declarations,
                    extra_forbidden_pii=extra_forbidden,
                )
            except PiiLeakBlockedError as exc:
                tool_ctx.xai_reasons.append(f"PRIVACY_PROXY_BLOCK: {exc}")
                return {
                    "pending_tool_calls": [],
                    "final_response": ChatResponse(
                        reply_text="Hệ thống bảo vệ dữ liệu đã chặn yêu cầu vì phát hiện PII chưa ẩn danh.",
                        decision="BLOCK_ANOMALY",
                        risk_score=1.0,
                        xai_reasons=list(tool_ctx.xai_reasons),
                        draft=None,
                        clarify_options=None,
                        new_balance=None,
                    ),
                }
            except PlannerUnavailableError as exc:
                logger.warning("PlannerUnavailableError: %s", exc)
                tool_ctx.xai_reasons.append(
                    "LLM_PLANNER_UNAVAILABLE: Tạm hoãn để đảm bảo an toàn giao dịch."
                )
                return {
                    "pending_tool_calls": [],
                    "final_response": ChatResponse(
                        reply_text="Trợ lý AI tạm thời quá tải hoặc mất kết nối. Vui lòng thử lại sau giây lát.",
                        decision="CLARIFY",
                        risk_score=0.0,
                        xai_reasons=list(tool_ctx.xai_reasons),
                        draft=None,
                        clarify_options=None,
                        new_balance=None,
                    ),
                }

            if turn.tool_calls:
                history.append(
                    {
                        "role": "model",
                        "raw_content": turn.raw_model_content,
                        "text": turn.text,
                        "tool_calls": [
                            {"name": tc.name, "args": tc.args} for tc in turn.tool_calls
                        ],
                    }
                )
                return {
                    "history": history,
                    "pending_tool_calls": turn.tool_calls,
                    "step_count": int(state.get("step_count", 0)) + 1,
                }

            return {
                "history": history,
                "pending_tool_calls": [],
                "llm_final_text": turn.text,
            }

        def tool_executor(state: AgentState) -> dict[str, Any]:
            tool_ctx = state["tool_ctx"]
            history = list(state.get("history") or [])
            calls = list(state.get("pending_tool_calls") or [])
            responses: list[dict[str, Any]] = []

            for tc in calls:
                res = self.tool_registry.execute(tool_ctx, tc.name, tc.args)
                responses.append({"name": tc.name, "response": res})
                if (
                    tool_ctx.terminal_wallet_error is not None
                    or tool_ctx.draft_response is not None
                    or tool_ctx.clarify_response is not None
                ):
                    break

            history.append({"role": "tool_responses", "responses": responses})
            return {
                "history": history,
                "pending_tool_calls": [],
                "tool_ctx": tool_ctx,
            }

        def post_checkpoint(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
            if state.get("final_response") is not None:
                return {}

            conn: sqlite3.Connection = config["configurable"]["conn"]
            tool_ctx = state["tool_ctx"]

            if tool_ctx.terminal_wallet_error is not None:
                raise tool_ctx.terminal_wallet_error

            if tool_ctx.clarify_response is not None:
                return {"final_response": tool_ctx.clarify_response}

            if tool_ctx.draft_response is not None:
                hard_resp = tool_ctx.draft_response
                if tool_ctx.xai_reasons:
                    merged_xai = list(dict.fromkeys([*hard_resp.xai_reasons, *tool_ctx.xai_reasons]))
                    hard_resp = hard_resp.model_copy(update={"xai_reasons": merged_xai})

                draft = hard_resp.draft
                gk_res = None
                if draft is not None and self.gatekeeper_hook.is_loaded:
                    gk_res = self.gatekeeper_hook.evaluate(
                        conn,
                        user_id=tool_ctx.user_id,
                        masked_message=tool_ctx.masked_user_message,
                        tool_name=tool_ctx.last_tool_name,
                        note=draft.note,
                        source_type=draft.source_type,
                        amount=draft.amount,
                        contact_id=tool_ctx.last_contact_id,
                        matched_contact_count=tool_ctx.last_matched_count,
                    )

                blocked_by_bl = any(
                    "BLACKLIST_HARD_VETO" in r for r in hard_resp.xai_reasons
                )
                is_ocr = (draft is not None and draft.source_type == "OCR_RECEIPT") or (
                    tool_ctx.has_image or tool_ctx.ocr_used
                )
                final_resp = merge_guardrail_and_gatekeeper(
                    hard_resp,
                    gk_res,
                    is_ocr=is_ocr,
                    blocked_by_blacklist=blocked_by_bl,
                )
                return {"final_response": final_resp}

            # Trường hợp LLM gọi xong upsert_contact_alias (lưu trực tiếp) hoặc chỉ trả lời text
            llm_text = (state.get("llm_final_text") or "").strip()
            if tool_ctx.alias_saved_direct:
                reply = llm_text or f"Đã lưu biệt danh '{tool_ctx.alias_saved_direct}' vào danh bạ."
                return {
                    "final_response": ChatResponse(
                        reply_text=reply,
                        decision="PASS_AUTO",
                        risk_score=0.05,
                        xai_reasons=list(tool_ctx.xai_reasons),
                        draft=None,
                        clarify_options=None,
                        new_balance=None,
                    )
                }

            if tool_ctx.no_ocr_alias_blocked:
                reply = (
                    llm_text
                    or "Hệ thống đã chặn thao tác lưu biệt danh từ ảnh hóa đơn OCR để chống Memory Poisoning."
                )
                return {
                    "final_response": ChatResponse(
                        reply_text=reply,
                        decision="BLOCK_ANOMALY",
                        risk_score=0.95,
                        xai_reasons=list(tool_ctx.xai_reasons),
                        draft=None,
                        clarify_options=None,
                        new_balance=None,
                    )
                }

            # Nếu LLM tự hỏi lại khi lookup_contact trả >= 2 người trùng tên -> gắn ClarifyOptions chuẩn OpenAPI
            if len(tool_ctx.last_lookup_matches) >= 2:
                cur_res = parse_currency(_strip_non_money_counts(tool_ctx.masked_user_message))
                parsed_amt = cur_res.amount if cur_res.status == "OK" else None
                is_ocr = tool_ctx.has_image or tool_ctx.ocr_used
                self.clarify_store.put(
                    tool_ctx.intent_id,
                    _PendingClarify(
                        user_id=tool_ctx.user_id,
                        masked_message=tool_ctx.masked_user_message,
                        source_type="OCR_RECEIPT" if is_ocr else "CHAT_CONTACT",
                        amount=parsed_amt,
                        recipient_query=tool_ctx.last_lookup_query,
                        acct_token=None,
                        bank_code=_extract_bank_code(tool_ctx.masked_user_message, default="MB"),
                        suggested_alias=None,
                        session_id=tool_ctx.session_id,
                        expires_at=0.0,
                    ),
                )
                options = [
                    ClarifyOptionItem(
                        label=f"{m['alias']} — {m['bank_code']} {m['masked_acct']}",
                        contact_id=m["contact_id"],
                    )
                    for m in tool_ctx.last_lookup_matches
                ]
                merged_xai = list(
                    dict.fromkeys(
                        [
                            *tool_ctx.xai_reasons,
                            "AMBIGUOUS_CONTACT: Có từ 2 danh bạ khớp từ khóa.",
                        ]
                    )
                )
                return {
                    "final_response": ChatResponse(
                        reply_text=llm_text
                        or "Tìm thấy nhiều người nhận trùng tên. Vui lòng chọn đúng người.",
                        decision="CLARIFY",
                        risk_score=0.0,
                        xai_reasons=merged_xai,
                        draft=None,
                        clarify_options=ClarifyOptions(kind="CONTACT", options=options),
                        new_balance=None,
                    )
                }

            # Truy vấn thông tin thuần túy (lịch sử giao dịch) -> PASS_AUTO thay vì CLARIFY
            if "get_recent_transactions" in tool_ctx.tool_trace and llm_text:
                return {
                    "final_response": ChatResponse(
                        reply_text=llm_text,
                        decision="PASS_AUTO",
                        risk_score=0.0,
                        xai_reasons=list(tool_ctx.xai_reasons),
                        draft=None,
                        clarify_options=None,
                        new_balance=None,
                    )
                }

            return {
                "final_response": ChatResponse(
                    reply_text=llm_text
                    or "Vui lòng cung cấp thêm thông tin người nhận và số tiền bạn muốn chuyển.",
                    decision="CLARIFY",
                    risk_score=0.0,
                    xai_reasons=list(tool_ctx.xai_reasons),
                    draft=None,
                    clarify_options=None,
                    new_balance=None,
                )
            }

        def finalize(state: AgentState) -> dict[str, Any]:
            return {"final_response": state.get("final_response")}

        def route_after_ingress(state: AgentState) -> Literal["finalize", "pre_guardrail"]:
            return "finalize" if state.get("short_circuit") else "pre_guardrail"

        def route_after_pre(state: AgentState) -> Literal["finalize", "llm_planner"]:
            return "finalize" if state.get("short_circuit") else "llm_planner"

        def route_after_llm(state: AgentState) -> Literal["tool_executor", "post_checkpoint"]:
            if state.get("final_response") is not None:
                return "post_checkpoint"
            calls = state.get("pending_tool_calls") or []
            step_cnt = int(state.get("step_count", 0))
            if calls and step_cnt <= self.config.max_tool_steps:
                return "tool_executor"
            return "post_checkpoint"

        def route_after_tools(state: AgentState) -> Literal["post_checkpoint", "llm_planner"]:
            tool_ctx = state["tool_ctx"]
            if (
                tool_ctx.terminal_wallet_error is not None
                or tool_ctx.draft_response is not None
                or tool_ctx.clarify_response is not None
            ):
                return "post_checkpoint"
            if int(state.get("step_count", 0)) >= self.config.max_tool_steps:
                return "post_checkpoint"
            return "llm_planner"

        builder.add_node("privacy_ingress", privacy_ingress)
        builder.add_node("pre_guardrail", pre_guardrail)
        builder.add_node("llm_planner", llm_planner)
        builder.add_node("tool_executor", tool_executor)
        builder.add_node("post_checkpoint", post_checkpoint)
        builder.add_node("finalize", finalize)

        builder.add_edge(START, "privacy_ingress")
        builder.add_conditional_edges("privacy_ingress", route_after_ingress)
        builder.add_conditional_edges("pre_guardrail", route_after_pre)
        builder.add_conditional_edges("llm_planner", route_after_llm)
        builder.add_conditional_edges("tool_executor", route_after_tools)
        builder.add_edge("post_checkpoint", "finalize")
        builder.add_edge("finalize", END)

        return builder.compile()
