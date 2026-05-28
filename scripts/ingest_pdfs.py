from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from research_agent.config import load_config
from research_agent.pdf_parser import extract_pdfs_to_texts


def main() -> None:
    config = load_config()
    papers_dir = Path(config["papers_dir"])
    texts_dir = Path(config["texts_dir"])

    output_paths = extract_pdfs_to_texts(papers_dir, texts_dir)

    print(f"papers_dir: {papers_dir}")
    print(f"texts_dir: {texts_dir}")
    print(f"generated: {len(output_paths)} txt")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
