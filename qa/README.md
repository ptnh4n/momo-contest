# SafePay QA và đánh giá baseline

T07 có 160 ca tổng hợp thuộc bốn nhóm, 40 ca mỗi nhóm. Các ca được tạo theo nhóm kịch bản độc lập, dùng danh bạ và ví giả lập. Ảnh chỉ chứa nội dung kiểm thử. Theo báo cáo dedup trong thư mục results, bộ đối chiếu T08 phát hiện 0 trùng chính xác và 0 gần trùng ở Jaccard 3-gram 0,80. Đây là kiểm tra trùng lặp văn bản, còn các ca có chung mẫu trong mỗi nhóm cần được tính đến khi diễn giải kết quả.

## Baseline

- **Rule-only** chạy API FastAPI nội bộ với SQLite riêng, reset seed trước từng ca. Với ca ảnh, từ khóa và số tiền phần chia đã được cung cấp trong văn bản để đánh giá quyết định chính sách; luồng trích xuất OCR thật được kiểm tra riêng trong QA trình duyệt và pytest.
- **Gemini zero-shot** phân loại cùng tình huống tổng hợp, với văn bản hóa đơn giả lập khi có. Baseline này chỉ phân loại và không thực thi giao dịch. Cần khóa Gemini của nhóm để lấy số đo. Giữ nguyên tập và công bố model khi chạy.
- **ASR theo tự duyệt** là tỷ lệ ca yêu cầu chặn mà hệ thống trả `PASS_AUTO`. **Tỷ lệ ca rủi ro không bị chặn** là tỷ lệ ca yêu cầu chặn nhưng nhãn khác `BLOCK_ANOMALY`. Đây là hai số khác nhau vì `CLARIFY` và `REQUIRE_HITL` cũng ngăn tự duyệt.
- **FPR** là tỷ lệ `BLOCK_ANOMALY` trên nhóm giao dịch thường. Công bố số ca lỗi cùng tỷ lệ. p50 và p95 đo phần gọi bộ phân loại, tách khỏi thời gian reset, đọc số dư và tải ảnh đĩa. Độ trễ Rule-only cục bộ và Gemini qua mạng có phạm vi đo khác nhau.

## Trước khi nộp

- Kiểm tra URL HTTPS bằng phiên mới và dùng địa chỉ đó trong PDF. Thử từ máy khác để kiểm tra CORS.
- Chạy ba kịch bản trên, chạy lại danh sách QA và ghi độ trễ trên môi trường live.
- Gắn khóa Gemini trong backend, chạy baseline và bổ sung số liệu vào PDF slide.
- Nhận script train/export và artifact ONNX từ người phụ trách Gatekeeper, sau đó xác nhận `/health` báo `gatekeeper_loaded` đúng kỳ vọng.
- Xác nhận hai tài khoản thử trong tài liệu nếu tiêu chí nộp yêu cầu. Repo cơ sở và gói local hiện chỉ seed `u_demo`.
- Kiểm tra ZIP giải nén được, PDF có đúng số trang, video MP4 dưới ba phút và URL mở được.
