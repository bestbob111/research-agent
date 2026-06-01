import json

from research_agent.history import append_jsonl


def test_append_jsonl_appends_records(tmp_path):
    path = tmp_path / "history" / "qa.jsonl"

    append_jsonl(path, {"question": "q1", "answer": "a1"})
    append_jsonl(path, {"question": "q2", "answer": "a2"})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2"},
    ]


def test_append_jsonl_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "history" / "qa.jsonl"

    append_jsonl(path, {"ok": True})

    assert path.exists()
