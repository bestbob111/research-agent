import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from research_agent.config import load_config
from research_agent.ollama_client import OllamaClient
from research_agent.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the local ChromaDB index.")
    parser.add_argument("question")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    config = load_config()
    client = OllamaClient(
        base_url=config["ollama_base_url"],
        chat_model=config["chat_model"],
        embedding_model=config["embedding_model"],
    )
    vector_store = VectorStore(
        chroma_dir=config["chroma_dir"],
        collection_name=config["collection_name"],
    )

    query_embedding = client.embed(args.question)
    results = vector_store.query(query_embedding, n_results=args.n)

    for index, result in enumerate(results, start=1):
        metadata = result["metadata"] or {}
        text = result["text"] or ""
        print(f"[{index}] source: {metadata.get('source', '')}")
        print(f"chunk_id: {metadata.get('chunk_id', '')}")
        print(f"distance: {result['distance']}")
        print(text[:500])
        print()


if __name__ == "__main__":
    main()
