import pytest

from research_agent.vector_store import VectorStore


def test_add_chunks_writes_and_count(tmp_path):
    store = VectorStore(str(tmp_path / "chroma"), "test_collection")

    written = store.add_chunks(
        doc_id="doc1",
        chunks=["hello", "world"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[{"source": "a.txt"}, {"source": "a.txt"}],
    )

    assert written == 2
    assert store.count() == 2


def test_query_returns_expected_structure(tmp_path):
    store = VectorStore(str(tmp_path / "chroma"), "test_collection")
    store.add_chunks(
        doc_id="doc1",
        chunks=["hello"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"source": "a.txt"}],
    )

    results = store.query([1.0, 0.0], n_results=1)

    assert len(results) == 1
    assert set(results[0]) == {"id", "text", "metadata", "distance"}
    assert results[0]["text"] == "hello"
    assert results[0]["metadata"]["doc_id"] == "doc1"


def test_add_chunks_length_mismatch_raises_value_error(tmp_path):
    store = VectorStore(str(tmp_path / "chroma"), "test_collection")

    with pytest.raises(ValueError):
        store.add_chunks("doc1", ["hello"], [], [{}])


def test_reset_collection_requires_confirm_true(tmp_path):
    store = VectorStore(str(tmp_path / "chroma"), "test_collection")

    with pytest.raises(ValueError):
        store.reset_collection()


def test_reset_collection_clears_collection(tmp_path):
    store = VectorStore(str(tmp_path / "chroma"), "test_collection")
    store.add_chunks("doc1", ["hello"], [[1.0, 0.0]], [{}])

    store.reset_collection(confirm=True)

    assert store.count() == 0


def test_add_chunks_empty_chunks_returns_zero(tmp_path):
    store = VectorStore(str(tmp_path / "chroma"), "test_collection")

    assert store.add_chunks("doc1", [], [], []) == 0
    assert store.count() == 0


def test_delete_doc_removes_matching_chunks(tmp_path):
    store = VectorStore(str(tmp_path / "chroma"), "test_collection")
    store.add_chunks("doc1", ["hello"], [[1.0, 0.0]], [{}])
    store.add_chunks("doc2", ["world"], [[0.0, 1.0]], [{}])

    store.delete_doc("doc1")

    assert store.count() == 1
    results = store.query([0.0, 1.0], n_results=1)
    assert results[0]["metadata"]["doc_id"] == "doc2"
