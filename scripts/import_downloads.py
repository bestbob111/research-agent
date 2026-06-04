import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from research_agent.config import load_config
from research_agent.download_importer import import_downloaded_pdfs


def main() -> None:
    parser = argparse.ArgumentParser(description="Import downloaded PDFs.")
    parser.add_argument("--source", choices=["all", "arxiv", "import"], default="all")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--copy", action="store_true", default=True)
    group.add_argument("--move", action="store_true")
    args = parser.parse_args()

    config = load_config()
    result = import_downloaded_pdfs(
        config,
        source=args.source,
        mode="move" if args.move else "copy",
    )
    print(f"found: {result['found']}")
    print(f"imported: {result['imported']}")
    print(f"failed: {len(result['failed'])}")
    for path in result["papers"]:
        print(f"paper: {path}")
    for path in result["texts"]:
        print(f"text: {path}")
    for failure in result["failed"]:
        print(f"failed: {failure['file']}: {failure['error']}")


if __name__ == "__main__":
    main()
