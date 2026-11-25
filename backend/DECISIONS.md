# DECISIONS.md — Quyết định thiết kế nhỏ

## 1. Thư viện băm PIN: argon2 (không dùng bcrypt)

**Quyết định:** Dùng `argon2-cffi` (đã cài sẵn, `argon2 ok`).  
**Lý do:** `bcrypt` không có trong môi trường (`ModuleNotFoundError`). `argon2-cffi` đã có sẵn (`Version: 25.1.0`). Tutorial mục 1 chỉ nói "bcrypt/argon2" — không ép phải bcrypt.

## 2. Schema: đúng theo tutorial.md mục 1 (không thêm cột từ backend_old)

**Quyết định:** `schema.sql` copy chính xác DDL từ mục 1 tutorial.md.  
**Lý do:** Backend_old có thêm các cột như `full_name_norm`, `alias_core`, `source`, `expires_at`, `resolved_at`, `approved_by`, `balance_after`... nhưng những cột này **không có trong tutorial.md mục 1**. Yêu cầu rõ ràng: "Không thêm cột, không bớt cột."

**Hệ quả:** Test cũ dùng `full_name_norm`, `alias_core`, `source`, `approved_by`, `balance_after` phải bỏ. Test mới viết theo schema tutorial.

## 3. contacts: không có UNIQUE(owner_id, account_number, bank_code)

**Quyết định:** Không thêm constraint UNIQUE này vào contacts.  
**Lý do:** Tutorial.md mục 1 không có constraint này. Backend_old có nhưng backend_old chỉ để tham khảo.  
**Hệ quả:** Test `test_duplicate_contact_account_bank_fails` của backend_old bị bỏ. Test mới không test constraint này.

## 4. reset_demo: không xóa scam_blacklist, dùng INSERT OR REPLACE

**Quyết định:** Không `DELETE FROM scam_blacklist` trong `reset_demo`; khi seed `scam_blacklist` dùng `INSERT OR REPLACE`. Toàn bộ bước xóa user data và chèn lại demo data chạy trong cùng một transaction.
**Lý do:** `scam_blacklist` không có cột `user_id` nên không xóa dữ liệu blacklist dùng chung khi reset một user; `INSERT OR REPLACE` đảm bảo các dòng blacklist mẫu luôn tồn tại đúng giá trị và idempotent.

## 5. upsert_alias: alias_raw cũ được giữ nguyên khi ON CONFLICT

**Quyết định:** `ON CONFLICT DO UPDATE SET usage_count = usage_count + 1` (không UPDATE alias_raw).  
**Lý do:** Tutorial mục 2 nói "giữ alias_raw cũ". SQLite ON CONFLICT không touch alias_raw nếu không SET nó.

## 6. seed_demo: hai Minh có cùng alias_norm 'minh'

**Quyết định:** C_003 và C_004 đều có `alias_norm='minh'`.  
**Lý do:** Tutorial mục 5 yêu cầu `lookup("Minh")` → 2 kết quả. Điều này chỉ xảy ra khi 2 contact khác nhau có cùng alias_norm.

## 7. Test _insert_contact không có full_name_norm

**Quyết định:** Bỏ cột `full_name_norm` khỏi helper `_insert_contact` trong test.  
**Lý do:** Schema mới (tutorial mục 1) không có cột này trong `contacts`.
