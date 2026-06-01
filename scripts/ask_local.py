import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from research_agent.config import load_config
from research_agent.history import append_jsonl
from research_agent.rag import LocalRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question using local RAG.")
    parser.add_argument("question")
    parser.add_argument("--n", type=int, default=6)
    args = parser.parse_args()

    question = args.question.strip()
    if not question:
        print("问题不能为空。")
        return

    config = load_config()
    rag = LocalRAG(config)
    result = rag.answer(question, n_results=args.n)

    print(f"问题: {question}")
    print("回答:")
    print(result["answer"])
    print("证据来源:")
    for source in result["sources"]:
        print(
            f"- source={source.get('source', '')}, "
            f"chunk_id={source.get('chunk_id', '')}, "
            f"distance={source.get('distance')}"
        )

    history_path = Path(config["metadata_dir"]) / "qa_history.jsonl"
    append_jsonl(
        history_path,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "model": config["chat_model"],
        },
    )


if __name__ == "__main__":
    main()
