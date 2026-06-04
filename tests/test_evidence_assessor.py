import json

from research_agent import evidence_assessor
from research_agent.evidence_assessor import assess_evidence


class FakeOllamaClient:
    calls = []

    def __init__(self, base_url, chat_model, embedding_model):
        self.base_url = base_url
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def chat(self, prompt):
        FakeOllamaClient.calls.append(prompt)
        return json.dumps(
            {
                "overall_strength": "medium",
                "supported_claims": [
                    {"claim": "结论 A", "evidence_ids": [1], "strength": "medium"}
                ],
                "weak_claims": [],
                "recommended_reading": [
                    {
                        "title": "Paper A",
                        "source": "paper.txt",
                        "reason": "相关",
                        "priority": "high",
                    }
                ],
                "possibly_irrelevant_sources": [],
            },
            ensure_ascii=False,
        )


class BadJsonOllamaClient(FakeOllamaClient):
    def chat(self, prompt):
        return "not json"


class FakeDeepSeekClient:
    calls = []

    def __init__(self, api_key, base_url, model):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def chat(self, prompt):
        FakeDeepSeekClient.calls.append(prompt)
        return json.dumps({"overall_strength": "strong"}, ensure_ascii=False)


def make_config():
    return {
        "ollama_base_url": "http://localhost:11434",
        "chat_model": "qwen3:14b",
        "embedding_model": "bge-m3",
        "deepseek_base_url": "https://api.deepseek.com",
        "deepseek_model": "deepseek-chat",
    }


def test_assess_evidence_parses_overall_strength(monkeypatch):
    FakeOllamaClient.calls = []
    monkeypatch.setattr(evidence_assessor, "OllamaClient", FakeOllamaClient)

    result = assess_evidence(
        "问题",
        "回答",
        [{"source": "paper.txt", "chunk_id": "c1", "text": "证据"}],
        make_config(),
    )

    assert result["overall_strength"] == "medium"
    assert result["recommended_reading"][0]["title"] == "Paper A"
    assert FakeOllamaClient.calls


def test_assess_evidence_bad_json_fallback(monkeypatch):
    monkeypatch.setattr(evidence_assessor, "OllamaClient", BadJsonOllamaClient)

    result = assess_evidence("问题", "回答", [], make_config())

    assert result["overall_strength"] == "unknown"
    assert result["supported_claims"] == []
    assert result["recommended_reading"] == []


def test_assess_evidence_deepseek(monkeypatch):
    FakeDeepSeekClient.calls = []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(evidence_assessor, "DeepSeekClient", FakeDeepSeekClient)

    result = assess_evidence(
        "问题",
        "回答",
        [{"source": "paper.txt", "chunk_id": "c1", "text": "证据"}],
        make_config(),
        use_deepseek=True,
    )

    assert result["overall_strength"] == "strong"
    assert FakeDeepSeekClient.calls
