from research_agent import indexer
from research_agent.indexer import build_index


class FakeOllamaClient:
    def __init__(self, base_url, chat_model, embedding_model):
        self.base_url = base_url
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def embed(self, text):
        return [float(len(text)), 1.0]


class PartiallyFailingOllamaClient:
    def __init__(self, base_url, chat_model, embedding_model):
        self.base_url = base_url
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def embed(self, text):
        if "FAIL" in text:
            raise RuntimeError("Ollama request failed: 500 test failure")
        return [float(len(text)), 1.0]


class NanEmbeddingOllamaClient:
    def __init__(self, base_url, chat_model, embedding_model):
        self.base_url = base_url
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def embed(self, text):
        if "BAD" in text:
            return [0.1, float("nan")]
        return [float(len(text)), 1.0]


def test_build_index_processes_one_text_file(tmp_path, monkeypatch):
    texts_dir = tmp_path / "texts"
    chroma_dir = tmp_path / "chroma"
    texts_dir.mkdir()
    (texts_dir / "paper.txt").write_text("第一段内容。\n\nSecond paragraph.", encoding="utf-8")
    monkeypatch.setattr(indexer, "OllamaClient", FakeOllamaClient)

    config = {
        "texts_dir": str(texts_dir),
        "chroma_dir": str(chroma_dir),
        "collection_name": "test_collection",
        "chunk_size": 1200,
        "chunk_overlap": 200,
        "ollama_base_url": "http://localhost:11434",
        "chat_model": "qwen3:14b",
        "embedding_model": "bge-m3",
    }

    result = build_index(config, reset=True)

    assert result["processed_files"] == 1
    assert result["written_chunks"] == 1
    assert result["collection_count"] == 1
    assert result["skipped_empty_files"] == 0
    assert result["failed_files"] == []
    assert result["failed_chunks"] == []


def test_build_index_skips_empty_text_file(tmp_path, monkeypatch):
    texts_dir = tmp_path / "texts"
    chroma_dir = tmp_path / "chroma"
    texts_dir.mkdir()
    (texts_dir / "empty.txt").write_text("   \n", encoding="utf-8")
    monkeypatch.setattr(indexer, "OllamaClient", FakeOllamaClient)

    config = {
        "texts_dir": str(texts_dir),
        "chroma_dir": str(chroma_dir),
        "collection_name": "test_collection_empty",
        "chunk_size": 1200,
        "chunk_overlap": 200,
        "ollama_base_url": "http://localhost:11434",
        "chat_model": "qwen3:14b",
        "embedding_model": "bge-m3",
    }

    result = build_index(config, reset=True)

    assert result["processed_files"] == 0
    assert result["written_chunks"] == 0
    assert result["collection_count"] == 0
    assert result["skipped_empty_files"] == 1


def test_build_index_continues_when_one_chunk_embed_fails(tmp_path, monkeypatch):
    texts_dir = tmp_path / "texts"
    chroma_dir = tmp_path / "chroma"
    texts_dir.mkdir()
    (texts_dir / "paper.txt").write_text(
        "good\n\nFAIL\n\nalso good",
        encoding="utf-8",
    )
    monkeypatch.setattr(indexer, "OllamaClient", PartiallyFailingOllamaClient)

    config = {
        "texts_dir": str(texts_dir),
        "chroma_dir": str(chroma_dir),
        "collection_name": "test_collection_partial_failure",
        "chunk_size": 10,
        "chunk_overlap": 0,
        "ollama_base_url": "http://localhost:11434",
        "chat_model": "qwen3:14b",
        "embedding_model": "bge-m3",
    }

    result = build_index(config, reset=True)

    assert result["processed_files"] == 1
    assert result["written_chunks"] == 1
    assert result["collection_count"] == 1
    assert len(result["failed_chunks"]) == 1
    assert result["failed_chunks"][0]["source"] == "paper.txt"
    assert result["failed_files"] == []


def test_build_index_skips_nan_embedding_and_writes_valid_chunks(tmp_path, monkeypatch):
    texts_dir = tmp_path / "texts"
    chroma_dir = tmp_path / "chroma"
    texts_dir.mkdir()
    (texts_dir / "paper.txt").write_text(
        "GOOD\n\nBAD\n\nALSO GOOD",
        encoding="utf-8",
    )
    monkeypatch.setattr(indexer, "OllamaClient", NanEmbeddingOllamaClient)

    config = {
        "texts_dir": str(texts_dir),
        "chroma_dir": str(chroma_dir),
        "collection_name": "test_collection_nan_embedding",
        "chunk_size": 10,
        "chunk_overlap": 0,
        "ollama_base_url": "http://localhost:11434",
        "chat_model": "qwen3:14b",
        "embedding_model": "bge-m3",
    }

    result = build_index(config, reset=True)

    assert result["processed_files"] == 1
    assert result["written_chunks"] == 1
    assert result["collection_count"] == 1
    assert len(result["failed_chunks"]) == 1
    assert result["failed_chunks"][0]["source"] == "paper.txt"
    assert "invalid embedding" in result["failed_chunks"][0]["error"]
