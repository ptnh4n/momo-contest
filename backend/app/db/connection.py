"""
connection.py — SQLite connection factory cho SafePay Agent.

Quy tắc:
- Mỗi kết nối phải bật: foreign_keys=ON, WAL journal, busy_timeout=5000.
- isolation_level=None để tự điều khiển BEGIN IMMEDIATE / COMMIT / ROLLBACK.
- row_factory = sqlite3.Row để truy cập cột theo tên.
- Thời gian: luôn dùng 'YYYY-MM-DD HH:MM:SS' (UTC), khớp CURRENT_TIMESTAMP của SQLite.
  Không dùng datetime.isoformat() (có chữ 'T') để tránh lỗi so sánh chuỗi.
"""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------
_DEFAULT_DB_PATH = "./safepay.db"
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_db_path() -> str:
    return os.environ.get("SAFEPAY_DB_PATH", _DEFAULT_DB_PATH)


# ---------------------------------------------------------------------------
# Factory kết nối
# ---------------------------------------------------------------------------
def get_conn(db_path: str | None = None) -> sqlite3.Connection:
    """
    Mở và trả về một kết nối SQLite đã cấu hình đầy đủ PRAGMA.
    Người gọi chịu trách nhiệm đóng kết nối.
    """
    path = db_path or get_db_path()
    conn = sqlite3.connect(path, isolation_level=None)  # autocommit off, tự quản transaction
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    return conn


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    """Áp dụng PRAGMA bắt buộc cho mỗi kết nối."""
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")


# ---------------------------------------------------------------------------
# Context manager: transaction nguyên tử
# ---------------------------------------------------------------------------
@contextmanager
def transaction(conn: sqlite3.Connection):
    """
    Context manager chạy BEGIN IMMEDIATE / COMMIT / ROLLBACK.

    Dùng:
        with transaction(conn):
            conn.execute(...)   # nếu raise → tự ROLLBACK
    """
    conn.execute("BEGIN IMMEDIATE;")
    try:
        yield conn
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise


# ---------------------------------------------------------------------------
# Khởi tạo schema (idempotent — dùng CREATE TABLE IF NOT EXISTS)
# ---------------------------------------------------------------------------
def init_db(db_path: str | None = None) -> None:
    """
    Chạy schema.sql lên DB. An toàn gọi nhiều lần (idempotent).
    """
    path = db_path or get_db_path()
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    conn = get_conn(path)
    try:
        # Chạy từng statement riêng lẻ (executescript tắt FK check tạm thời)
        # Thay bằng executescript nhưng bật lại FK sau
        conn.executescript(sql)
        # executescript tự COMMIT; bật lại FK (executescript tắt FK tạm)
        _apply_pragmas(conn)
    finally:
        conn.close()
