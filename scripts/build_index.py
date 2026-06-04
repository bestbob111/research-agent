import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from research_agent.config import load_config
from research_agent.indexer import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB index from text files.")
    parser.add_argument("--reset", action="store_true", help="Recreate the collection.")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only index new or changed txt files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index selected txt files even if unchanged.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process first N txt files.")
    parser.add_argument(
        "--max-chars-per-embed",
        type=int,
        default=1800,
        help="Maximum characters sent to one embedding request.",
    )
    args = parser.parse_args()

    config = load_config()
    texts_dir = Path(config["texts_dir"])
    text_files = sorted(texts_dir.glob("*.txt")) if texts_dir.exists() else []
    if args.limit is not None:
        text_files = text_files[: args.limit]

    print(f"texts_dir: {texts_dir}")
    print(f"chroma_dir: {config['chroma_dir']}")
    print(f"collection_name: {config['collection_name']}")

    if not text_files:
        print("No txt files found. Run python scripts/ingest_pdfs.py first.")

    result = build_index(
        config,
        reset=args.reset,
        limit=args.limit,
        max_chars_per_embed=args.max_chars_per_embed,
        incremental=args.incremental,
        force=args.force,
    )

    print(f"processed txt: {result['processed_files']}")
    print(f"written chunks: {result['written_chunks']}")
    print(f"collection count: {result['collection_count']}")
    print(f"skipped empty files: {result['skipped_empty_files']}")
    print(f"skipped unchanged files: {result['skipped_unchanged_files']}")
    print(f"failed files: {len(result['failed_files'])}")
    print(f"failed chunks: {len(result['failed_chunks'])}")

    if result["failed_files"] or result["failed_chunks"]:
        metadata_dir = Path(config["metadata_dir"])
        metadata_dir.mkdir(parents=True, exist_ok=True)
        report_path = metadata_dir / "index_failures.json"
        report_path.write_text(
            json.dumps(
                {
                    "failed_files": result["failed_files"],
                    "failed_chunks": result["failed_chunks"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"failure report: {report_path}")


if __name__ == "__main__":
    main()
