from research_agent.index_state import (
    calculate_sha256,
    clear_index_state,
    count_indexed_files,
    get_indexed_file,
    upsert_indexed_file,
)


def test_index_state_records_and_counts_file(tmp_path):
    db_path = tmp_path / "metadata" / "index_state.sqlite"
    text_path = tmp_path / "texts" / "paper.txt"
    text_path.parent.mkdir()
    text_path.write_text("content", encoding="utf-8")

    sha256 = calculate_sha256(text_path)
    upsert_indexed_file(db_path, str(text_path), sha256, 3)

    record = get_indexed_file(db_path, str(text_path))
    assert record is not None
    assert record["sha256"] == sha256
    assert record["chunk_count"] == 3
    assert count_indexed_files(db_path) == 1


def test_clear_index_state_removes_records(tmp_path):
    db_path = tmp_path / "metadata" / "index_state.sqlite"
    upsert_indexed_file(db_path, "/tmp/a.txt", "abc", 1)

    clear_index_state(db_path)

    assert count_indexed_files(db_path) == 0
