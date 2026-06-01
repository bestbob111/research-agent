import pytest

from research_agent import rag
from research_agent.rag import LocalRAG


class FakeOllamaClient:
    last_instance = None

    def __init__(self, base_url, chat_model, embedding_model):
        self.base_url = base_url
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.prompts = []
        FakeOllamaClient.last_instance = self

    def embed(self, text):
        return [1.0, 0.0]

    def chat(self, prompt, system=None):
        self.prompts.append(prompt)
        return "回答内容 [证据 1]"


class FakeVectorStore:
    results = []
    queries = []

    def __init__(self, chroma_dir, collection_name):
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name

    def query(self, query_embedding, n_results=6):
        FakeVectorStore.queries.append((query_embedding, n_results))
        return FakeVectorStore.results


def make_config(tmp_path):
    return {
        "ollama_base_url": "http://localhost:11434",
        "chat_model": "qwen3:14b",
        "embedding_model": "bge-m3",
        "chroma_dir": str(tmp_path / "chroma"),
        "collection_name": "test_collection",
    }


def setup_fakes(monkeypatch, results):
    FakeVectorStore.results = results
    FakeVectorStore.queries = []
    FakeOllamaClient.last_instance = None
    monkeypatch.setattr(rag, "OllamaClient", FakeOllamaClient)
    monkeypatch.setattr(rag, "VectorStore", FakeVectorStore)


def test_retrieve_returns_results(tmp_path, monkeypatch):
    results = [{"id": "id1", "text": "text", "metadata": {}, "distance": 0.1}]
    setup_fakes(monkeypatch, results)

    local_rag = LocalRAG(make_config(tmp_path))

    assert local_rag.retrieve("问题", n_results=3) == results
    assert FakeVectorStore.queries == [([1.0, 0.0], 3)]


def test_answer_without_results_returns_no_evidence_message(tmp_path, monkeypatch):
    setup_fakes(monkeypatch, [])

    local_rag = LocalRAG(make_config(tmp_path))
    result = local_rag.answer("问题")

    assert "未在当前文献库中检索到足够相关的证据" in result["answer"]
    assert result["sources"] == []


def test_answer_prompt_contains_evidence_marker(tmp_path, monkeypatch):
    setup_fakes(
        monkeypatch,
        [
            {
                "id": "doc::chunk_0",
                "text": "证据文本",
                "metadata": {"source": "paper.txt", "chunk_id": "doc::chunk_0"},
                "distance": 0.2,
            }
        ],
    )

    local_rag = LocalRAG(make_config(tmp_path))
    result = local_rag.answer("问题")

    assert result["answer"] == "回答内容 [证据 1]"
    prompt = FakeOllamaClient.last_instance.prompts[0]
    assert "[证据 1 | source=paper.txt | chunk_id=doc::chunk_0]" in prompt
    assert "只根据给定证据回答" in prompt


def test_empty_question_raises_value_error(tmp_path, monkeypatch):
    setup_fakes(monkeypatch, [])
    local_rag = LocalRAG(make_config(tmp_path))

    with pytest.raises(ValueError):
        local_rag.retrieve("  ")
