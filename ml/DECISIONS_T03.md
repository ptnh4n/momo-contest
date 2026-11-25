# DECISIONS_T03.md — Quyết định kỹ thuật T03 (Feature Schema & Tokenizer VN thu gọn)

Các quyết định dưới đây đã được thống nhất với người phụ trách và kiểm chứng thực nghiệm trên Kaggle (`fragilov/safepay-t03-tokenizer-bench`):

## 1. Mở rộng Tabular Feature Vector từ 10 lên 14 đặc trưng
- **Quyết định:** Giữ đủ 10 đặc trưng gốc từ Blueprint v0.2.1 §5.1 (index `0..9`) và bổ sung 4 đặc trưng sẵn có từ `backend/app/db/schema.sql` (index `10..13`): `daily_total_ratio`, `balance_drain_ratio`, `is_raw_account`, `has_ambiguous_amount`.
- **Ảnh hưởng tham số:** `Linear(14, 128)` chỉ tăng `+512` tham số (`tabular_params = 10.432`, tổng mô hình `4.246.661` tham số `< 20M`).
- **Chuẩn hóa cố định (Zero Train–Serve Skew):** Toàn bộ 14 đặc trưng dùng chặn/log chuẩn hóa hằng số trong `ml/gatekeeper/feature_schema.json` và `ml/gatekeeper/preprocessing.py`, không phụ thuộc thống kê tập train.

## 2. Tương thích ngữ nghĩa `seed.py` và `schema.sql`
- Trong `backend/app/db/seed.py`, `C_001` (Nam) và `C_002` (sếp Hùng) có `trust_level='TRUSTED'` dù `contacts.transfer_count=0`.
- Do đó, `is_new_recipient` định nghĩa là `1.0 if (contact_id is None or trust_level != 'TRUSTED') else 0.0`, và `effective_transfer_count = max(contacts.transfer_count, COUNT(transactions COMPLETED))`.

## 3. Chiến lược Tokenizer tiếng Việt & Kết quả Vòng lặp Kaggle
- **Vòng 1:** So sánh 6 cấu hình `{unigram, bpe} x {12000, 14000, 16000}` trên 398.716 dòng. `spm_bpe_16000` dẫn đầu (`overall_fertility=1.2906`), nhưng lộ 3 vấn đề: `prompt_injection_bi=1.8091` (> 1.65), `ĐẶT`/`PHÒNG` ALL-CAPS rơi vào byte fallback (`character_coverage=0.9999`), và kiểm tra atomicity tính nhầm ký tự prefix `▁`.
- **Vòng 2:** Nâng `character_coverage=1.0`, bổ sung 15k câu ViFiC ALL-CAPS có dấu + ngữ liệu tiếng Anh bảo mật/system prompt đa dạng (tổng 423.716 dòng), chuẩn hóa kiểm tra atomicity đúng cơ chế SentencePiece.
- **Kết quả Vòng 2:** `spm_bpe_16000` đạt **17/17 tiêu chí** trên Kaggle và **8/8 tiêu chí** tổng thể:
  - `overall_unk_rate = 0.0`, `overall_byte_fallback_rate = 0.0`, `special_token_atomicity = 1.0`.
  - `overall_fertility = 1.233` (`normal_accented = 1.1172`, `normal_unaccented = 1.1253`, `gap = 0.0081`, `teencode_slang = 1.137`, `scam_social_eng = 1.1783`, `prompt_injection_bi = 1.56`, `ocr_receipts = 1.2886`).
  - `keyword_integrity_le2 = 0.9536` (`>= 0.95`), `p95_seq_len = 41` (`<= 96`), `truncation_rate_128 = 0.0`.
  - `embedding_params = 1.074.432` (`< 2M`), `total_params = 4.246.661` (`< 20M`), ước lượng ONNX INT8 `~4.05 MB`.

## 4. Phòng thủ Tiền xử lý Văn bản
- **Giữ HOA/thường & dấu tiếng Việt (NFC):** bảo toàn tín hiệu thúc ép IN HOA (`CHUYỂN GẤP`, `KHẨN CẤP`) và phân biệt biệt danh.
- **Xóa ký tự zero-width (`\u200b`, `\u200c`, `\u200d`, `\ufeff`, `\u00ad`, `\u2060`):** chống kỹ thuật chèn ký tự ẩn né bộ lọc.
- **Chống Delimiter Spoofing:** chuyển `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`, `[UNK]` giả mạo trong tin nhắn/ghi chú sang `(SEP)`... và bảo toàn cứng `tool_name` + `source_type` ở Segment B kèm cắt head+tail cho Segment A khi vượt `max_length = 128`.
