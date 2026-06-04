import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def init_index_state(db_path: Path) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS indexed_files (
                text_path TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                indexed_at TEXT NOT NULL
            )
            """
        )


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_indexed_file(db_path: Path, text_path: str) -> dict | None:
    if not Path(db_path).exists():
        return None
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM indexed_files WHERE text_path = ?",
            (text_path,),
        ).fetchone()
        return dict(row) if row else None


def upsert_indexed_file(
    db_path: Path,
    text_path: str,
    sha256: str,
    chunk_count: int,
) -> None:
    init_index_state(db_path)
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO indexed_files (text_path, sha256, chunk_count, indexed_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(text_path) DO UPDATE SET
                sha256 = excluded.sha256,
                chunk_count = excluded.chunk_count,
                indexed_at = excluded.indexed_at
            """,
            (text_path, sha256, chunk_count, now),
        )


def delete_indexed_file(db_path: Path, text_path: str) -> None:
    init_index_state(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM indexed_files WHERE text_path = ?", (text_path,))


def clear_index_state(db_path: Path) -> None:
    init_index_state(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM indexed_files")


def count_indexed_files(db_path: Path) -> int:
    init_index_state(db_path)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()
        return int(row[0])
