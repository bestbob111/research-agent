import requests
import pytest

from research_agent.ollama_client import OllamaClient


class FakeResponse:
    def __init__(self, data, status_code=200, text=""):
        self.data = data
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self.data


def test_embed_request_url_and_payload(monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append((url, json, timeout))
        return FakeResponse({"embeddings": [[0.1, 0.2]]})

    monkeypatch.setattr("requests.post", fake_post)
    client = OllamaClient("http://localhost:11434/", "qwen3:14b", "bge-m3")

    client.embed("hello")

    assert calls == [
        (
            "http://localhost:11434/api/embed",
            {"model": "bge-m3", "input": "hello"},
            120,
        )
    ]


def test_embed_returns_vector(monkeypatch):
    def fake_post(url, json, timeout):
        return FakeResponse({"embeddings": [[1.0, 2.0, 3.0]]})

    monkeypatch.setattr("requests.post", fake_post)
    client = OllamaClient("http://localhost:11434", "qwen3:14b", "bge-m3")

    assert client.embed("hello") == [1.0, 2.0, 3.0]


def test_embed_empty_input_raises_value_error():
    client = OllamaClient("http://localhost:11434", "qwen3:14b", "bge-m3")

    with pytest.raises(ValueError):
        client.embed("   ")


def test_embed_invalid_response_raises_runtime_error(monkeypatch):
    def fake_post(url, json, timeout):
        return FakeResponse({"embedding": [1.0]})

    monkeypatch.setattr("requests.post", fake_post)
    client = OllamaClient("http://localhost:11434", "qwen3:14b", "bge-m3")

    with pytest.raises(RuntimeError):
        client.embed("hello")


def test_embed_nan_vector_raises_invalid_embedding_runtime_error(monkeypatch):
    def fake_post(url, json, timeout):
        return FakeResponse({"embeddings": [[0.1, float("nan")]]})

    monkeypatch.setattr("requests.post", fake_post)
    client = OllamaClient("http://localhost:11434", "qwen3:14b", "bge-m3")

    with pytest.raises(RuntimeError, match="invalid embedding"):
        client.embed("hello")


def test_embed_inf_vector_raises_invalid_embedding_runtime_error(monkeypatch):
    def fake_post(url, json, timeout):
        return FakeResponse({"embeddings": [[0.1, float("inf")]]})

    monkeypatch.setattr("requests.post", fake_post)
    client = OllamaClient("http://localhost:11434", "qwen3:14b", "bge-m3")

    with pytest.raises(RuntimeError, match="invalid embedding"):
        client.embed("hello")


def test_chat_request_messages(monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append((url, json, timeout))
        return FakeResponse({"message": {"content": "answer"}})

    monkeypatch.setattr("requests.post", fake_post)
    client = OllamaClient("http://localhost:11434/", "qwen3:14b", "bge-m3")

    client.chat("hello", system="be concise")

    assert calls == [
        (
            "http://localhost:11434/api/chat",
            {
                "model": "qwen3:14b",
                "messages": [
                    {"role": "system", "content": "be concise"},
                    {"role": "user", "content": "hello"},
                ],
                "stream": False,
            },
            120,
        )
    ]


def test_chat_returns_content(monkeypatch):
    def fake_post(url, json, timeout):
        return FakeResponse({"message": {"content": "answer"}})

    monkeypatch.setattr("requests.post", fake_post)
    client = OllamaClient("http://localhost:11434", "qwen3:14b", "bge-m3")

    assert client.chat("hello") == "answer"


def test_chat_empty_input_raises_value_error():
    client = OllamaClient("http://localhost:11434", "qwen3:14b", "bge-m3")

    with pytest.raises(ValueError):
        client.chat("\n ")


def test_http_exception_becomes_runtime_error(monkeypatch):
    def fake_post(url, json, timeout):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr("requests.post", fake_post)
    client = OllamaClient("http://localhost:11434", "qwen3:14b", "bge-m3")

    with pytest.raises(RuntimeError, match="Ollama"):
        client.chat("hello")


def test_http_500_runtime_error_includes_response_text(monkeypatch):
    def fake_post(url, json, timeout):
        return FakeResponse(
            {"error": "server"},
            status_code=500,
            text="model crashed while embedding",
        )

    monkeypatch.setattr("requests.post", fake_post)
    client = OllamaClient("http://localhost:11434", "qwen3:14b", "bge-m3")

    with pytest.raises(RuntimeError, match="model crashed while embedding"):
        client.embed("hello")
