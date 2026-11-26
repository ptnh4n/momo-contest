# DECISION.md — Quyết định thiết kế nhỏ (app/wallet)

## 1. Tổ chức gói mới trong `backend/app/wallet/`
**Quyết định:** Gom toàn bộ API ví giả lập, Hard Guardrails, Deterministic Currency Parser, Privacy Proxy và luật chống Memory Poisoning vào một package mới `backend/app/wallet/` và không chỉnh sửa các file nền ở `app/db/`.
**Lý do:** Giữ rõ ranh giới tầng DB (`app/db/`) với tầng nghiệp vụ ví & bảo vệ cứng (`app/wallet/`), đảm bảo toàn bộ 88 test hiện có không bị ảnh hưởng.

## 2. Thứ tự ưu tiên Regex trong Privacy Proxy
**Quyết định:**
1. Ưu tiên nhận diện theo từ khóa ngữ cảnh tường minh trước: STK (`stk`, `số tài khoản`, `tài khoản` + 8–16 số -> `<ACCT_k>`), CCCD (`cccd`, `cmnd`, `căn cước` + 12 số -> `<CCCD_k>`), SĐT (`sđt`, `phone` + `0[35789]\d{8}` -> `<PHONE_k>`).
2. Sau đó mới quét số trần không nhãn: SĐT VN 10 số (`0[35789]\d{8}` -> `<PHONE_k>`) rồi đến STK 8–16 chữ số (`\d{8,16}` -> `<ACCT_k>`).
**Lý do:** Tại Việt Nam nhiều ngân hàng cho dùng chính SĐT làm STK (VD: `"STK 0987654321 MB"`). Kiểm tra ngữ cảnh trước tránh mask nhầm STK thành `<PHONE_1>`, đồng thời vẫn đảm bảo 0% số 8–16 chữ số hoặc SĐT lọt ra ngoài.

## 3. Khóa đầu ra Zero-PII cho tool `lookup_contact`
**Quyết định:** `app/wallet/privacy_proxy.py::lookup_contact` bao bọc `app/db/repo/contacts.py::lookup_contact_zero_pii`, khóa cứng đúng whitelist 5 trường (`contact_id`, `alias`, `bank_code`, `masked_acct`, `trust_level`) khớp `CONTEXT.md` mục 3.2 và `tutorial.md` mục 2; ép định dạng `masked_acct` dạng `****xxx` (3 số cuối) và lọc/ẩn mọi chuỗi số PII nếu vô tình xuất hiện trong `alias`.
**Lý do:** Đảm bảo 0% rò rỉ STK/SĐT/CCCD/họ tên đầy đủ sang LLM mà vẫn tương thích nguyên vẹn với hợp đồng dữ liệu đã chốt.

## 4. Ngưỡng nhận diện số tiền mơ hồ trong Deterministic Currency Parser
**Quyết định:**
- Số nguyên trần không có hậu tố đơn vị (`k`/`nghìn`/`tr`/`triệu`/`đ`/`vnd`) và không có dấu phân cách hàng nghìn (`500.000`): nếu nằm trong khoảng `1..9999` (VD: `"500"`, `"200"`, `"1000"`) → bắt buộc trả `AMBIGUOUS` (ép `CLARIFY` với 2 phương án `val * 1000 VNĐ` và `val VNĐ`).
- Số có hậu tố đơn vị rõ (`500k`, `1.5tr`, `1tr5`, `500đ`, `500000 vnd`), có dấu chấm/phẩy hàng nghìn (`500.000`, `10.000.000`), hoặc số nguyên trần `>= 10.000` (VD: `500000`) → trả `OK`.
- Giá trị `<= 0` (VD: `0đ`, `-500k`) → trả `INVALID` (HTTP 422 `VALIDATION_ERROR`).
- Bỏ qua các token `<ACCT_k>`, `<PHONE_k>`, `<CCCD_k>` trước khi parse để không đọc nhầm chữ số trong token làm số tiền.
**Lý do:** Chặn triệt để bẫy đơn vị tiền tệ tiếng Việt (`"500"` -> 500đ vs 500k) mà không gây phiền cho các câu lệnh đã ghi rõ số tiền đầy đủ.

## 5. Thứ tự ưu tiên các lớp bảo vệ cứng (Hard Guardrails)
**Quyết định:** Đánh giá theo thứ tự nghiêm ngặt từ cao xuống thấp:
1. **Hard Veto Blacklist** → `BLOCK_ANOMALY` (lưu draft `status='BLOCKED'`, phủ quyết tất cả, kể cả `PASS_AUTO` hay giao dịch nhỏ).
2. **Kiểm tra số dư** (`amount > balance`) → trả HTTP 422 `INSUFFICIENT_BALANCE` (không tạo draft).
3. **Luật cứng `OCR_ALWAYS_HITL`** (`source_type == 'OCR_RECEIPT'`) → tối thiểu `REQUIRE_HITL`, không bao giờ `PASS_AUTO`.
4. **Ngưỡng NHNN** (`amount >= 10_000_000` VNĐ/lần HOẶC `daily_transferred + amount >= 20_000_000` VNĐ/ngày) → bắt buộc `REQUIRE_HITL` (kể cả khi user nâng `auto_approve_limit` cao hơn và người nhận là `TRUSTED`).
5. **Hạn mức tự duyệt** (`trust_level == 'TRUSTED'` và `amount <= auto_approve_limit`) → `PASS_AUTO`, thực thi trừ tiền và ghi `transactions` ngay trong cùng 1 transaction SQLite; các trường hợp còn lại → `REQUIRE_HITL`.

## 6. Idempotency & Chống trừ tiền đúp
**Quyết định:**
- `POST /api/chat` / `create_draft_transfer`: kiểm tra `draft_transactions.intent_id` (đã có `UNIQUE` ở DB) trước khi tạo mới, và xử lý `sqlite3.IntegrityError` phòng race condition → trả lại đúng draft cũ, không tạo thêm bản ghi và không trừ tiền lần 2.
- `POST /api/drafts/{tx_id}/confirm`: nếu draft đã ở trạng thái `COMPLETED`, trả lại `ConfirmResponse` với số dư hiện tại (idempotent), không trừ tiền thêm. Trừ số dư dùng câu lệnh có điều kiện `WHERE user_id = ? AND balance >= ?` trong `BEGIN IMMEDIATE` transaction.

## 7. Luật chống Memory Poisoning & Post-PIN Commit
**Quyết định:**
- `NO_OCR_ALIAS`: `enforce_no_ocr_alias` luôn ép `suggested_alias = None` khi `source_type == 'OCR_RECEIPT'`; `commit_alias_post_pin` cũng từ chối ghi alias cho nguồn OCR kể cả khi client truyền `save_alias=True`. Lớp DB `CHECK` tiếp tục là hàng rào thứ hai.
- `NO_SILENT_OVERWRITE` + Post-PIN Commit: `suggested_alias` chỉ nằm tạm trên `draft_transactions` cho tới khi `confirm_draft` xác thực PIN thành công (`argon2`) và `save_alias=True`. `PASS_AUTO` (không nhập PIN) và trường hợp nhập sai PIN (401) tuyệt đối không ghi vào `contact_aliases`. Khi commit, gọi `upsert_alias` nên alias trùng trên cùng contact chỉ tăng `usage_count` (giữ `alias_raw` cũ), còn alias trùng trên contact khác thêm dòng mới để lần sau `lookup_contact` trả >=2 kết quả và ép `CLARIFY`.

## 8. `CLARIFY` giữ ngữ cảnh tạm trong RAM & `/health`
**Quyết định:** Theo `tutorial.md` mục 3, trạng thái `CLARIFY` không ghi vào SQLite mà lưu tạm trong `ClarifyContextStore` (RAM, TTL 5 phút). Endpoint `GET /health` trả `gatekeeper_loaded: false` vì mô hình ML Gatekeeper nằm ngoài phạm vi nhiệm vụ này.
