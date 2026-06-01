import math
from typing import Any

import requests


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        chat_model: str,
        embedding_model: str,
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.timeout = timeout

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("text must not be empty")

        payload = {"model": self.embedding_model, "input": text}
        response = self._post("/api/embed", payload)
        embeddings = response.get("embeddings")

        if (
            not isinstance(embeddings, list)
            or not embeddings
            or not isinstance(embeddings[0], list)
        ):
            raise RuntimeError("Ollama embed response missing valid embeddings")

        embedding = embeddings[0]
        if not _is_valid_embedding(embedding):
            raise RuntimeError("Ollama embed response contains invalid embedding")

        return embedding

    def chat(self, prompt: str, system: str | None = None) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.chat_model,
            "messages": messages,
            "stream": False,
        }
        response = self._post("/api/chat", payload)

        try:
            content = response["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Ollama chat response missing message content") from exc

        if not isinstance(content, str):
            raise RuntimeError("Ollama chat response content is not a string")

        return content

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            if not 200 <= response.status_code < 300:
                body = getattr(response, "text", "")[:300]
                raise RuntimeError(
                    f"Ollama request failed: HTTP {response.status_code} for {url}: {body}"
                )
            data = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError("Ollama response is not valid JSON") from exc

        if not isinstance(data, dict):
            raise RuntimeError("Ollama response JSON must be an object")

        return data


def _is_valid_embedding(embedding: Any) -> bool:
    if not isinstance(embedding, list) or not embedding:
        return False
    return all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        for value in embedding
    )
