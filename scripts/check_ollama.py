from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from research_agent.config import load_config
from research_agent.ollama_client import OllamaClient


def main() -> None:
    config = load_config()
    client = OllamaClient(
        base_url=config["ollama_base_url"],
        chat_model=config["chat_model"],
        embedding_model=config["embedding_model"],
    )

    try:
        embedding = client.embed("test embedding")
        print(f"embedding dimension: {len(embedding)}")

        answer = client.chat("用一句话说明 SERF 原子磁强计是什么。")
        print(answer)
    except RuntimeError as exc:
        print(f"Ollama check failed: {exc}")
        print("请确认 Ollama 已启动，并且 qwen3:14b 与 bge-m3 模型已拉取。")


if __name__ == "__main__":
    main()
