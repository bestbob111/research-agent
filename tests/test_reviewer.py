from research_agent import reviewer
from research_agent.reviewer import EvidenceReviewer


class FakeLocalRAG:
    results = []

    def __init__(self, config):
        self.config = config

    def retrieve(self, question, n_results=12):
        return FakeLocalRAG.results


class FakeDeepSeekClient:
    last_prompt = None

    def __init__(self, api_key, base_url, model):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def chat(self, prompt, system=None, temperature=0.2):
        FakeDeepSeekClient.last_prompt = prompt
        return "结构化综述 [证据 1]"


def make_config(tmp_path):
    return {
        "ollama_base_url": "http://localhost:11434",
        "chat_model": "qwen3:14b",
        "embedding_model": "bge-m3",
        "chroma_dir": str(tmp_path / "chroma"),
        "collection_name": "test_collection",
        "deepseek_base_url": "https://api.deepseek.com",
        "deepseek_model": "deepseek-chat",
    }


def setup_fakes(monkeypatch, results):
    FakeLocalRAG.results = results
    FakeDeepSeekClient.last_prompt = None
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(reviewer, "LocalRAG", FakeLocalRAG)
    monkeypatch.setattr(reviewer, "DeepSeekClient", FakeDeepSeekClient)


def test_review_without_evidence_returns_message(tmp_path, monkeypatch):
    setup_fakes(monkeypatch, [])
    evidence_reviewer = EvidenceReviewer(make_config(tmp_path))

    result = evidence_reviewer.review_with_deepseek("问题")

    assert "未在当前文献库中检索到足够相关的证据" in result["review"]
    assert result["sources"] == []


def test_review_prompt_contains_evidence_marker(tmp_path, monkeypatch):
    setup_fakes(
        monkeypatch,
        [
            {
                "id": "doc::chunk_0",
                "text": "证据文本",
                "metadata": {"source": "paper.txt", "chunk_id": "doc::chunk_0"},
                "distance": 0.3,
            }
        ],
    )
    evidence_reviewer = EvidenceReviewer(make_config(tmp_path))

    result = evidence_reviewer.review_with_deepseek("问题")

    assert result["review"] == "结构化综述 [证据 1]"
    assert "[证据 1 | source=paper.txt | chunk_id=doc::chunk_0]" in (
        FakeDeepSeekClient.last_prompt
    )
    assert "核心结论" in FakeDeepSeekClient.last_prompt


def test_review_returns_sources(tmp_path, monkeypatch):
    setup_fakes(
        monkeypatch,
        [
            {
                "id": "doc::chunk_0",
                "text": "证据文本",
                "metadata": {"source": "paper.txt", "chunk_id": "doc::chunk_0"},
                "distance": 0.3,
            }
        ],
    )
    evidence_reviewer = EvidenceReviewer(make_config(tmp_path))

    result = evidence_reviewer.review_with_deepseek("问题")

    assert result["sources"] == [
        {
            "evidence_id": 1,
            "source": "paper.txt",
            "chunk_id": "doc::chunk_0",
            "distance": 0.3,
            "id": "doc::chunk_0",
        }
    ]
