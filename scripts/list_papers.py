import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from research_agent.config import load_config
from research_agent.cli import _non_negative_int, _positive_int, _slice_papers
from research_agent.metadata_db import init_db, list_papers


def main() -> None:
    parser = argparse.ArgumentParser(description="List paper metadata.")
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument("--offset", type=_non_negative_int, default=0)
    args = parser.parse_args()

    config = load_config()
    db_path = Path(config["metadata_dir"]) / "papers.sqlite"
    init_db(db_path)
    papers = list_papers(db_path)
    visible_papers = _slice_papers(papers, args.offset, args.limit)

    print(f"metadata_db: {db_path}")
    print(f"papers: {len(papers)}")
    for paper in visible_papers:
        print(
            f"{paper['id']}\t{paper.get('title') or ''}\t"
            f"{paper.get('file_path') or ''}\t{paper.get('text_path') or ''}"
        )


if __name__ == "__main__":
    main()
