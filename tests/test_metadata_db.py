import sqlite3

from research_agent.metadata_db import (
    create_topic,
    get_paper_by_text_path,
    get_topic,
    init_db,
    list_papers,
    list_topics,
    update_topic,
    upsert_paper,
)


def test_init_db_creates_papers_table(tmp_path):
    db_path = tmp_path / "metadata" / "papers.sqlite"

    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'papers'"
        ).fetchone()
    assert row == ("papers",)


def test_init_db_creates_topics_table(tmp_path):
    db_path = tmp_path / "metadata" / "papers.sqlite"

    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'topics'"
        ).fetchone()
    assert row == ("topics",)


def test_upsert_paper_inserts(tmp_path):
    db_path = tmp_path / "papers.sqlite"

    paper_id = upsert_paper(
        db_path,
        {
            "title": "Paper A",
            "file_path": "/papers/a.pdf",
            "text_path": "/texts/a.txt",
            "source": "local_pdf",
        },
    )

    papers = list_papers(db_path)
    assert paper_id == 1
    assert len(papers) == 1
    assert papers[0]["title"] == "Paper A"


def test_upsert_paper_updates_existing(tmp_path):
    db_path = tmp_path / "papers.sqlite"
    first_id = upsert_paper(
        db_path,
        {
            "title": "Old Title",
            "file_path": "/papers/a.pdf",
            "text_path": "/texts/a.txt",
        },
    )

    second_id = upsert_paper(
        db_path,
        {
            "title": "New Title",
            "file_path": "/papers/a.pdf",
            "text_path": "/texts/a.txt",
            "year": "2024",
        },
    )

    papers = list_papers(db_path)
    assert second_id == first_id
    assert len(papers) == 1
    assert papers[0]["title"] == "New Title"
    assert papers[0]["year"] == "2024"


def test_list_papers_returns_all_rows(tmp_path):
    db_path = tmp_path / "papers.sqlite"
    upsert_paper(db_path, {"title": "A", "file_path": "/papers/a.pdf"})
    upsert_paper(db_path, {"title": "B", "file_path": "/papers/b.pdf"})

    papers = list_papers(db_path)

    assert [paper["title"] for paper in papers] == ["A", "B"]


def test_get_paper_by_text_path(tmp_path):
    db_path = tmp_path / "papers.sqlite"
    upsert_paper(
        db_path,
        {
            "title": "Paper A",
            "file_path": "/papers/a.pdf",
            "text_path": "/texts/a.txt",
        },
    )

    paper = get_paper_by_text_path(db_path, "/texts/a.txt")

    assert paper is not None
    assert paper["title"] == "Paper A"
    assert get_paper_by_text_path(db_path, "/texts/missing.txt") is None


def test_topic_crud(tmp_path):
    db_path = tmp_path / "papers.sqlite"

    topic_id = create_topic(
        db_path,
        {
            "name": "SERF noise",
            "description": "Noise suppression",
            "keywords_cn": "噪声\n抑制",
        },
    )
    update_topic(db_path, topic_id, {"notes": "important"})

    topic = get_topic(db_path, topic_id)
    topics = list_topics(db_path)

    assert topic is not None
    assert topic["name"] == "SERF noise"
    assert topic["notes"] == "important"
    assert len(topics) == 1
    assert get_topic(db_path, 999) is None
