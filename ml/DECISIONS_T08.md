# DECISIONS_T08.md — Quyết định kỹ thuật T08 (Sinh tập Train/Val tổng hợp)

## 1. Nguồn dữ liệu & Quy mô
- **Tổng quy mô:** `16.000` mẫu sạch (`13.600` Train / `2.400` Val — tỷ lệ `85% / 15%`), phân bố đúng theo Blueprint v0.2.1 §5.2:
  - `PASS_AUTO`: `40%` (`5.440` train / `960` val)
  - `REQUIRE_HITL`: `25%` (`3.400` train / `600` val)
  - `CLARIFY`: `15%` (`2.040` train / `360` val)
  - `BLOCK_ANOMALY`: `20%` (`2.720` train / `480` val)
- **Kết hợp 2 nguồn độc lập:**
  - **Nguồn A (~60% — 9.600 mẫu):** Nhân bản và biến thiên có kiểm soát từ `240` seed sạch sinh bằng AI Studio trong `ml/data/seeds/` (`pass_auto.json`, `require_hitl.json`, `clarify.json`, `block_anomaly.json`).
  - **Nguồn B (~40% — 6.400 mẫu):** Bộ sinh tổ hợp lập trình trong `ml/gatekeeper/generate_t08.py` bổ sung các kịch bản cứng giảm False Positive Rate (FPR) và chống shortcut.

## 2. Sửa lỗi `detect_ambiguous_amount` ở `ml/gatekeeper/preprocessing.py`
- Khi audit `ml/data/seeds/clarify.json`, phát hiện 9/40 mẫu `ambiguous_currency` / `ambiguous_both` bị bỏ sót do `_PRE_NON_MONEY_RE` gom chung không dấu `ban` ("bàn" vs động từ "bắn") và `ngay` ("ngày" vs phó từ "ngay") — ví dụ `"ban 350 cho a Tuan"`, `"Gửi ngay 1500 cho mẹ"`, `"Bắn ngay 120 cho Thảo"`.
- **Giải pháp:** Tách thành `_PRE_NON_MONEY_STRICT_RE`, kết hợp luật ngữ cảnh riêng cho `ngay` (chỉ bỏ qua khi `1 <= val <= 31` và trước `ngay` không phải động từ chuyển tiền) và `ban` (chỉ bỏ qua khi phía trước đã có động từ chuyển tiền khác như `"chuyen 50k tien com trua ban 12"`). Đã khóa hồi quy trong `ml/tests/test_t03_feature_schema.py`.

## 3. Curation 5 seed `memory_poisoning` trong `block_anomaly.json`
- 5 seed ở chỉ số `[47, 49, 50, 54, 56]` trong `ml/data/seeds/block_anomaly.json` nguyên bản dùng `source_type="CHAT_RAW_ACCT"` với lời văn giống người dùng chủ động lưu biệt danh (dễ chồng lấn với `REQUIRE_HITL` hợp lệ `upsert_contact_alias`).
- Giữ nguyên 4 file JSON gốc trong `ml/data/seeds/` để đảm bảo tính kiểm chứng; hàm `load_curated_seeds()` trong `generate_t08.py` chuyển 5 seed này sang ngữ cảnh tấn công Memory Poisoning rõ ràng từ nguồn `OCR_RECEIPT` / lệnh ẩn bỏ qua PIN. Đồng thời bổ sung nhóm `legit_upsert_alias` vào `REQUIRE_HITL` để mô hình phân biệt rạch ròi yêu cầu chủ động của người dùng với Memory Poisoning.

## 4. Thiết kế Chống Shortcut Tabular (cho Ablation Study T11) & Giảm FPR
- **Zero Train–Serve Skew:** Toàn bộ vector 14 chiều của cả 16.000 mẫu được tính trực tiếp qua `ml.gatekeeper.preprocessing.vectorize_features()`.
- **Stealth Tabular (~55% mẫu `BLOCK_ANOMALY`):** Sinh giao dịch lừa đảo/injection vào ban ngày (`is_late_night=0`), số tiền vừa (`150k–5M`), tỷ lệ rút ví vừa phải (`balance_drain_ratio <= 0.55`) — trùng khớp phân bố Tabular với `REQUIRE_HITL` (`raw_acct_legit`, `ocr_receipt_legit`), buộc nhánh Text phải đọc hiểu ngôn ngữ thúc ép/injection thay vì dựa vào mỗi 14 cột Tabular.
- **Hard Negatives & Chống shortcut hành vi:**
  - Bổ sung ~20% `REQUIRE_HITL` ban đêm (`is_late_night=1`) / rút ví cao (`0.55–0.92`), ~15% `PASS_AUTO` ban đêm, và ~10% `CLARIFY` ban đêm.
  - Bổ sung ~15% giao dịch nối tiếp nhanh hợp lệ (`time_since_last_tx_seconds` trong `30–300s`) cho cả `PASS_AUTO` và `REQUIRE_HITL` (và ~10% cho `CLARIFY`) để tránh mô hình học shortcut chuyển nhanh = lừa đảo.
  - Bổ sung tỷ lệ Cold-Start (`user_avg_amount = None` ~15%, `time_since_last_tx_seconds = None` ~10%) khớp với trạng thái mặc định của user demo `seed.py`.
  - Bổ sung các câu bình thường chứa từ `"gấp"` / `"shipper"` hoặc số phi tiền tệ (`bàn 12`, `tháng 11`) để giảm FPR.
- **Định dạng JSONL gọn cho T09:** Gắn sẵn `label_id` (`0..3` khớp `feature_schema.json`) và loại bỏ khóa `message` lặp lại bên trong `tabular_raw`.

## 5. Chia Train/Val chống Data Leakage & Kiểm tra chéo với T07
- Chia `train.jsonl` / `val.jsonl` ở cấp `group_id` (mỗi seed gốc và mỗi họ template thuộc duy nhất Train hoặc Val), đồng thời lọc bỏ hoàn toàn mọi `user_message` trùng chuẩn hóa giữa Train và Val.
- Khi nhận được tập Red-Team độc lập `test_redteam.jsonl` từ T07, chạy ngay lệnh sau để kiểm chứng (và lọc bỏ nếu có câu vô tình trùng/near-duplicate >= 0.80 Jaccard 3-gram):

```powershell
& "C:\Users\Kiet\AppData\Local\Programs\Python\Python311\python.exe" -m ml.gatekeeper.dedup_check `
  --train ml/data/t08/train.jsonl `
  --val ml/data/t08/val.jsonl `
  --redteam path/to/test_redteam.jsonl `
  --purge-leaks
```
