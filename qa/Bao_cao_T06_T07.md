# Kết quả chạy lại T06 và T07 · 29/11/2026

Đối chiếu với [bảng kế hoạch SafePay](https://docs.google.com/spreadsheets/d/1wth9sGvr2zsXwiP5rqcj9_rV94Kwd60Nuf7sF7xljL4/edit?gid=0#gid=0). Chạy trên bản bàn giao trước khi tạo commit báo cáo. Mốc GitHub dùng để đối chiếu là `d2ff813`.

## T06 · Frontend và backend ví giả lập

| Tiêu chí | Kết quả |
|---|---|
| Chuyển nhanh | Chuyển 24.000 ₫ cho người nhận có trong danh bạ. Backend và UI đều trả số dư 9.976.000 ₫. Tải lại trang vẫn thấy số dư này. |
| Chia hóa đơn | OCR đọc ảnh tổng tiền 400.000 ₫, chia bốn ra 100.000 ₫ và tạo bản nháp chờ PIN. |
| Hủy giao dịch | Bản nháp chuyển sang trạng thái đã hủy và số dư giữ nguyên. |
| Lưu người nhận | PIN sai giữ nguyên danh bạ. PIN đúng yêu cầu người dùng chọn lưu trước khi ghi biệt danh và cập nhật số dư. |

Kết quả **27/27 kiểm tra trình duyệt đạt** với Next.js gọi FastAPI chạy cục bộ. Frontend qua lint và build. **171/171 test backend và ML đạt**. Luồng Gemini thật còn chờ khóa API của nhóm. Kết quả từng ca ở [browser-local.json](results/browser-local.json).

## T07 · Bộ ca đánh giá độc lập

**192 ca, đủ 4 nhãn**. Đối chiếu với **13.600 mẫu train và 2.400 mẫu validation của T08** cho 0 trùng chính xác và 0 gần trùng theo Jaccard 3-gram ≥ 0,80. Không có câu trùng sau chuẩn hóa trong T07.

**32/32 ca phòng vệ giả lập đạt**, gồm **16 ca OCR đọc ảnh**. Các ca dùng nội dung trung tính để kiểm tra người nhận, số dư và danh bạ trước và sau PIN. Nhóm đầu vào đối kháng cụ thể theo yêu cầu T07 vẫn chưa đầy đủ. Báo cáo kiểm tra và SHA-256 ở [t07-acceptance.json](results/t07-acceptance.json).

Chạy lại đối chứng Rule-only trên đúng tập này cho **148/192 ca khớp nhãn (77,1%)**. Trong 44 ca lệch, 20 ca giao dịch thường và 24 ca rủi ro đều trả về hỏi lại. Kết quả này thuộc lượt đo T10 và chưa có dự đoán từ Gatekeeper của Kiệt.

## Tin nhắn báo Kiệt

> T6 tao chạy lại local: 27/27 kiểm tra qua, gồm chuyển nhanh, chia bill OCR, hủy và lưu người nhận sau PIN. T7 có 192 ca đủ 4 nhãn, so với 16.000 mẫu T8 không thấy trùng theo checker trong repo. 32 ca phòng vệ giả lập đều qua, còn thiếu nhóm ca đối kháng cụ thể trong sheet. Rule-only vẫn đúng 148/192. Đây là điểm baseline, chưa phải điểm Gatekeeper.
