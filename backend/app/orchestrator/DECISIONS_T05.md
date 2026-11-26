# DECISIONS_T05.md — Quyết định kỹ thuật T05 (LLM Orchestrator & Privilege Separation)

## 1. LangGraph StateGraph gọi trực tiếp `google-genai` (Model-Agnostic)
- **Quyết định:** Dựng `SafePayOrchestrator` bằng `langgraph.graph.StateGraph` (6 bước: `privacy_ingress` → `pre_guardrail` → `llm_planner` ↔ `tool_executor` → `post_checkpoint` → `finalize`) và gọi SDK `google-genai` trực tiếp thông qua giao diện `BasePlannerClient` thay vì `langchain-google-genai`.
- **Lý do:** Kiểm soát 100% payload JSON gửi ra ngoài để thực thi kiểm tra fail-closed Zero-PII (`assert_zero_pii_outbound`) + `OutboundAuditLog`, giữ trọn `thought_signature` cho Gemini 4 (`gemini-4-hydrogen`, `thinking_level="LOW"`), hỗ trợ xoay vòng API key (chuẩn bị T13) và giữ sẵn `GatekeeperHook` cho T12.

## 2. OCR hóa đơn 100% cục bộ bằng RapidOCR trước Privacy Proxy
- **Quyết định:** Ảnh `image_base64` chỉ lưu tạm trong RAM (`ImageStagingStore`, TTL 300s) và được trích text trên Backend bằng `RapidOCR`, sau đó đi qua `PrivacyGuard` để mask STK/SĐT/CCCD (cả dạng viết liền và tách dấu cách/chấm/gạch ngang theo E07) thành `<ACCT_k>`, `<PHONE_k>`, `<CCCD_k>` trước khi gửi tới Gemini.
- **Lý do:** Đảm bảo 0% ảnh hóa đơn gốc và 0% PII thô rời khỏi Backend. Nếu request có `image_base64`, Backend ép cứng `source_type = "OCR_RECEIPT"` và `suggested_alias = None` bất kể nội dung LLM truyền vào (`OCR_ALWAYS_HITL` + `NO_OCR_ALIAS`).

## 3. Hành vi `upsert_contact_alias` theo Phương án B (Xác nhận 2 bước cho STK mới)
- **Quyết định:**
  - Nếu có ảnh / nguồn OCR → chặn cứng (`NO_OCR_ALIAS`).
  - Nếu `contact_ref` là danh bạ đã có → ghi ngay qua `upsert_alias` (`NO_SILENT_OVERWRITE`).
  - Nếu `contact_ref` là STK thô mới chưa có trong `contacts` → kiểm tra blacklist; nếu sạch, lưu tạm vào `PendingAliasStore` (RAM, TTL 300s, chưa ghi DB) và trả `CLARIFY` yêu cầu xác nhận 2 bước; lượt sau user trả lời đồng ý mới tạo contact (`trust_level='NEW'`) và ghi `contact_aliases`.
- **Lý do:** Không phá vỡ hợp đồng OpenAPI 6 endpoint đã chốt ở T01, đồng thời ngăn chặn ghi rác/đầu độc danh bạ từ một lượt chat đơn lẻ.

## 4. Cô lập test T04 khi chạy `pytest`
- **Quyết định:** `load_backend_env()` bỏ qua nạp `backend/.env` mặc định khi `PYTEST_CURRENT_TEST` đang chạy (trừ khi truyền `force=True`) và `SAFEPAY_ORCHESTRATOR_MODE=auto` tự chạy T04 deterministic nếu không có key.
- **Lý do:** Toàn bộ bộ test T01/T04 (`test_wallet_guardrails.py`) tiếp tục chạy nhanh, offline, xác định 100% mà không gọi nhầm API Gemini thật.
