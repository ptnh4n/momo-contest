# SafePay Agent

Trợ lý ví giả lập cho MoMo Contest. Frontend Next.js kết nối FastAPI và SQLite. Luồng demo mặc định chạy cục bộ, sử dụng duy nhất dữ liệu giả lập. Tài khoản demo là `u_demo` và PIN xác nhận là `123456`.

## Chạy sản phẩm

### Docker Compose

Cần Docker với Compose v2. Từ thư mục gốc:

```bash
docker compose up --build
```

Mở http://localhost:3000. API ở http://localhost:8000/docs. Cấu hình Compose mở cổng loopback và dùng một cơ sở dữ liệu ví giả lập.

### Chạy trực tiếp

Dùng Node.js 24 và Python 3.12. Mở hai terminal trong repo. Trên Windows có thể dùng WSL hoặc kích hoạt môi trường ảo rồi chạy từng lệnh.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app.wallet.api:app --host 127.0.0.1 --port 8000
```

```bash
cd frontend
npm ci
npm run dev
```

Nếu dùng API từ xa, cấu hình `NEXT_PUBLIC_API_BASE_URL` trước khi build frontend và `SAFEPAY_CORS_ORIGINS` với origin HTTPS của UI. Ví mặc định dùng chung một tài khoản; khi dùng cho nhiều người, chạy các instance tách biệt hoặc tích hợp tài khoản theo phiên.

Muốn bật Gemini cho luồng hội thoại mở rộng, đặt `GEMINI_API_KEY` ở môi trường backend và `SAFEPAY_ORCHESTRATOR_MODE=llm`. Có thể đặt `GEMINI_MODEL` để dùng model mà project được cấp quyền. Mẫu cấu hình ở `backend/.env.example` và `frontend/.env.example`.

## Ba kịch bản demo

1. **Chuyển tiền theo biệt danh.** Bấm **Reset Demo**, gửi **Chuyển 500k cho thằng Nam**. Xác nhận bằng PIN demo. Số dư còn 9.500.000 ₫.
2. **Chia hóa đơn.** Reset, đính kèm `qa/receipts/ocr-006.png`. Gửi **Chia hóa đơn cho 4 người rồi chuyển cho thằng Nam**. OCR local đọc 400.000 ₫ và tạo bản nháp 100.000 ₫. Số dư chỉ đổi sau khi xác nhận. Cú pháp này hoạt động ở chế độ deterministic.
3. **Danh sách chặn.** Gửi **Chuyển 50k tới STK 0000000001 MB**. Đây là tài khoản giả lập có trong seed blacklist. Hệ thống chặn và giữ nguyên số dư.

## Đánh giá và QA

Cài dependencies backend trước, rồi chạy từ thư mục gốc:

```bash
python -m pytest backend/tests
python -m ml.gatekeeper.dedup_check --train ml/data/t08/train.jsonl --val ml/data/t08/val.jsonl --redteam qa/test_redteam.jsonl
python qa/evaluate.py --mode rule-only
python qa/evaluate.py --mode gemini-zero-shot --model gemini-3.6-flash
```

Lệnh Gemini cần biến môi trường `GEMINI_API_KEY`. Bộ ca chỉ dùng ví và danh tính giả lập. File CSV và JSON được ghi vào `qa/results`; xem [phương pháp và phạm vi đo](qa/README.md).

## Mã nguồn và kết quả

- `frontend/` chứa Chat UI, trình tải ảnh, thẻ giao dịch và modal PIN.
- `backend/` chứa API, orchestrator, SQLite, bộ luật bảo vệ và OCR local RapidOCR.
- `ml/` chứa scripts tokenizer, sinh dữ liệu, tokenizer đã train và bộ T08 của nhóm. Commit cơ sở chưa chứa script train/export ONNX của mô hình Gatekeeper.
- `qa/` chứa 160 tình huống độc lập, ảnh mẫu, baseline và báo cáo.

Dự án sử dụng Next.js, React, Tailwind, FastAPI, SQLite, Argon2, LangGraph, Gemini SDK, RapidOCR, ONNX Runtime, Pillow và SentencePiece. Gemini được thiết kế cho lập kế hoạch hội thoại tùy cấu hình. Phần mã tích hợp, tài liệu bàn giao và dữ liệu QA của đợt này có hỗ trợ từ ChatGPT/Codex.
