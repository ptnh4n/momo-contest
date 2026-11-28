# Rà soát T04, T06 và T07 · 29/11/2026

Đối chiếu với [bảng kế hoạch SafePay](https://docs.google.com/spreadsheets/d/1wth9sGvr2zsXwiP5rqcj9_rV94Kwd60Nuf7sF7xljL4/edit?gid=0#gid=0) và nhánh chính tại `d2ff813`.

| Task | Kết quả |
|---|---|
| T04 | Sửa luồng giao dịch có bước hỏi lại người nhận. Cả hai chế độ Rule-only và planner chỉ giữ biệt danh đề xuất trên bản nháp. Người dùng phải chọn lưu và nhập PIN đúng để cập nhật danh bạ. Các test kiểm tra chống trừ lặp, blacklist, số tiền mơ hồ, ngưỡng xác nhận và che dữ liệu vẫn đạt. |
| T06 | 28/28 kiểm tra trình duyệt đạt với FastAPI và ví giả lập local. Kiểm tra mới xác nhận khi chọn người trùng tên, UI đưa ra tùy chọn lưu biệt danh và mặc định bỏ chọn. Đóng hộp PIN và hủy giao dịch giữ nguyên số dư. Chuyển nhanh, tải lại số dư, OCR, hủy và lưu người nhận đều được chạy lại. Chưa chạy với Gemini thật. |
| T07 | Kiểm tra lại cùng 192 ca và 4 nhãn. Đối chiếu 16.000 mẫu T08 cho 0 trùng chính xác và 0 gần trùng theo ngưỡng của checker. Cả 32 ca phòng vệ giả lập đạt, gồm 16 ca đọc ảnh OCR. Script nay kiểm tra thêm PIN sai và xác nhận lặp. Nhóm ca đầu vào đối kháng theo Excel vẫn chưa đầy đủ. |

**172/172 test backend và ML đạt.** Lần chạy Rule-only vẫn khớp **148/192 ca (77,1%)**. Đây là baseline thuộc T10. Lượt này chưa có kết quả Gatekeeper.

Chi tiết ở [báo cáo trình duyệt](results/browser-local.json) và [báo cáo T07](results/t07-acceptance.json).

## Gửi Kiệt

> T4 tao sửa luồng chọn người nhận lúc hỏi lại, giờ chỉ đề xuất biệt danh trên draft, chọn lưu và nhập PIN đúng mới ghi danh bạ. T6 local pass 28/28. T7 vẫn 192 ca đủ 4 nhãn, checker không phát hiện trùng T8, 32 ca phòng vệ giả lập pass. Nhóm đầu vào đối kháng theo sheet vẫn chưa đủ. Rule-only vẫn 148/192, chưa phải kết quả Gatekeeper.
