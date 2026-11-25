# Nguồn gốc & Minh bạch Dữ liệu Tokenizer (T03 Attribution)

Tài liệu này phục vụ mục khai báo minh bạch thư viện mã nguồn mở và nguồn dữ liệu trong bài nộp MoMo Contest (Task T03 / T17).

## 1. Thư viện & Thuật toán
- **Thư viện huấn luyện:** `sentencepiece` (Apache-2.0 License, phiên bản `0.2.2` tại local).
- **Thuật toán chọn lọc:** SentencePiece **BPE** (`vocab_size = 16000`, `byte_fallback = true`, `character_coverage = 1.0`, `normalization_rule_name = "nmt_nfkc"`).
- **Lý do chọn BPE 16k qua thực nghiệm đối chứng trên Kaggle:** Vòng 2 đạt 17/17 tiêu chí kiểm định (`overall_fertility = 1.233`, `prompt_injection_bi = 1.56 <= 1.65`, `ocr_receipts = 1.2886`, `byte_fallback_rate = 0.0`, `special_token_atomicity = 1.0`), trong khi cả 3 biến thể Unigram (12k/14k/16k) đều không đạt ngưỡng `prompt_injection_bi <= 1.65`.

## 2. Nguồn Corpus Huấn luyện (Chỉ dùng thống kê từ vựng subword — Không chứa nhãn)
1. **ViFiC-120M (`daddychillonkaggle/vietnamese-financial-corpus`, CC-BY-NC-SA-4.0):**
   - Dùng `165.000` câu tin tức tài chính/ngân hàng tiếng Việt có dấu + `60.000` câu khử dấu tiếng Việt + `15.000` câu IN HOA có dấu (hỗ trợ tiêu đề hóa đơn OCR).
2. **VSA Food Reviews (`guutran/vietnamese-sentiment-analysis-food-reviews`):**
   - Dùng `11.858` bình luận ăn uống tiếng Việt có dấu + `11.858` bản khử dấu (lấy văn phong chat tự nhiên, chia bill, khẩu ngữ).
3. **Ngữ liệu miền thanh toán tổng hợp (`train_eval_tokenizer.py`, seed cố định `42`):**
   - `160.000` mẫu hội thoại chuyển khoản, teencode (`ck`, `stk`, `ko`, `đc`, `củ`, `lít`, `xị`), hóa đơn OCR, mẫu Privacy Proxy (`<ACCT_1..5>`, `<PHONE_1..5>`, `<CCCD_1..5>`), từ vựng cảnh báo lừa đảo tâm lý & prompt injection song ngữ (Việt–Anh).
   - Hoàn toàn độc lập với tập kiểm thử Red-Team `T07` (Zero-Leakage).
