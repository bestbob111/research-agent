import sqlite3
from datetime import datetime, timezone
from pathlib import Path


PAPER_FIELDS = (
    "title",
    "authors",
    "year",
    "venue",
    "source",
    "url",
    "doi",
    "file_path",
    "text_path",
    "abstract",
    "keywords",
    "notes",
)


def init_db(db_path: Path) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                authors TEXT,
                year TEXT,
                venue TEXT,
                source TEXT,
                url TEXT,
                doi TEXT,
                file_path TEXT UNIQUE,
                text_path TEXT,
                abstract TEXT,
                keywords TEXT,
                notes TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )


def upsert_paper(db_path: Path, paper: dict) -> int:
    init_db(db_path)
    existing = _find_existing_paper(db_path, paper)
    now = datetime.now(timezone.utc).isoformat()

    values = {field: paper.get(field) for field in PAPER_FIELDS}
    values["updated_at"] = now

    with sqlite3.connect(db_path) as conn:
        if existing:
            assignments = ", ".join(f"{field} = ?" for field in PAPER_FIELDS)
            params = [values[field] for field in PAPER_FIELDS]
            params.extend([values["updated_at"], existing["id"]])
            conn.execute(
                f"UPDATE papers SET {assignments}, updated_at = ? WHERE id = ?",
                params,
            )
            return int(existing["id"])

        fields = (*PAPER_FIELDS, "created_at", "updated_at")
        placeholders = ", ".join("?" for _field in fields)
        params = [values.get(field) for field in PAPER_FIELDS]
        params.extend([now, now])
        cursor = conn.execute(
            f"INSERT INTO papers ({', '.join(fields)}) VALUES ({placeholders})",
            params,
        )
        return int(cursor.lastrowid)


def list_papers(db_path: Path) -> list[dict]:
    if not Path(db_path).exists():
        return []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM papers ORDER BY id").fetchall()
        return [dict(row) for row in rows]


def get_paper_by_text_path(db_path: Path, text_path: str) -> dict | None:
    if not Path(db_path).exists():
        return None
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM papers WHERE text_path = ?",
            (text_path,),
        ).fetchone()
        return dict(row) if row else None


def _find_existing_paper(db_path: Path, paper: dict) -> dict | None:
    file_path = paper.get("file_path")
    text_path = paper.get("text_path")
    clauses = []
    params = []
    if file_path:
        clauses.append("file_path = ?")
        params.append(file_path)
    if text_path:
        clauses.append("text_path = ?")
        params.append(text_path)
    if not clauses:
        return None

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            f"SELECT * FROM papers WHERE {' OR '.join(clauses)} LIMIT 1",
            params,
        ).fetchone()
        return dict(row) if row else None
