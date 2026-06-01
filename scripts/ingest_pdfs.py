from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from research_agent.config import load_config
from research_agent.ingest_metadata import record_ingested_metadata
from research_agent.pdf_parser import extract_pdfs_to_texts_with_report


def print_progress(index: int, total: int, pdf_path: Path) -> None:
    print(f"[{index}/{total}] processing: {pdf_path.name}")


def main() -> None:
    config = load_config()
    papers_dir = Path(config["papers_dir"])
    texts_dir = Path(config["texts_dir"])

    print(f"papers_dir: {papers_dir}")
    print(f"texts_dir: {texts_dir}")
    pdf_count = len(sorted(papers_dir.glob("*.pdf"))) if papers_dir.exists() else 0
    print(f"found PDFs: {pdf_count}")

    report = extract_pdfs_to_texts_with_report(
        papers_dir,
        texts_dir,
        progress_callback=print_progress,
    )

    print(f"generated txt: {len(report['success'])}")
    for path in report["success"]:
        print(path)
    metadata_count = record_ingested_metadata(config, report["success"])
    print(f"metadata records updated: {metadata_count}")
    print(f"failed PDFs: {len(report['failures'])}")
    for failure in report["failures"]:
        print(f"{failure['pdf']}: {failure['error']}")
    print(f"output txt dir: {texts_dir}")


if __name__ == "__main__":
    main()
