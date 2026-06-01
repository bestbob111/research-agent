import pytest

from research_agent import deepseek_client
from research_agent.deepseek_client import DeepSeekClient


class FakeMessage:
    content = "deepseek answer"


class FakeChoice:
    message = FakeMessage()


class FakeResponse:
    choices = [FakeChoice()]


class FakeCompletions:
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.should_fail:
            raise RuntimeError("api failed")
        return FakeResponse()


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeOpenAI:
    completions = FakeCompletions()

    def __init__(self, api_key, base_url, timeout):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.chat = FakeChat(FakeOpenAI.completions)


def test_deepseek_client_empty_api_key_raises_value_error():
    with pytest.raises(ValueError):
        DeepSeekClient("", "https://api.deepseek.com", "deepseek-chat")


def test_deepseek_chat_empty_prompt_raises_value_error(monkeypatch):
    monkeypatch.setattr(deepseek_client, "OpenAI", FakeOpenAI)
    client = DeepSeekClient("key", "https://api.deepseek.com", "deepseek-chat")

    with pytest.raises(ValueError):
        client.chat("  ")


def test_deepseek_chat_returns_content(monkeypatch):
    FakeOpenAI.completions = FakeCompletions()
    monkeypatch.setattr(deepseek_client, "OpenAI", FakeOpenAI)
    client = DeepSeekClient("key", "https://api.deepseek.com/", "deepseek-chat")

    assert client.chat("prompt", system="system") == "deepseek answer"
    assert client.base_url == "https://api.deepseek.com"
    assert FakeOpenAI.completions.calls[0]["model"] == "deepseek-chat"
    assert FakeOpenAI.completions.calls[0]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "prompt"},
    ]


def test_deepseek_api_exception_becomes_runtime_error(monkeypatch):
    FakeOpenAI.completions = FakeCompletions(should_fail=True)
    monkeypatch.setattr(deepseek_client, "OpenAI", FakeOpenAI)
    client = DeepSeekClient("key", "https://api.deepseek.com", "deepseek-chat")

    with pytest.raises(RuntimeError, match="DeepSeek"):
        client.chat("prompt")
