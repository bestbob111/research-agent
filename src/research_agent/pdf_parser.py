from pathlib import Path
from typing import Callable

import fitz

from research_agent.text_cleaner import clean_text


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF file with page markers."""
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if not path.is_file():
        raise ValueError(f"PDF path is not a file: {path}")

    try:
        with fitz.open(path) as document:
            pages = []
            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text()
                pages.append(f"--- Page {page_number} ---\n{page_text}")
    except Exception as exc:
        raise RuntimeError(f"Failed to open or parse PDF: {path}") from exc

    return clean_text("\n\n".join(pages))


def extract_pdfs_to_texts(papers_dir: Path, texts_dir: Path) -> list[Path]:
    """Extract all PDFs in papers_dir to UTF-8 text files in texts_dir."""
    report = extract_pdfs_to_texts_with_report(papers_dir, texts_dir)
    for failure in report["failures"]:
        print(f"Failed to extract {failure['pdf']}: {failure['error']}")
    return report["success"]


def extract_pdfs_to_texts_with_report(
    papers_dir: Path,
    texts_dir: Path,
    progress_callback: Callable[[int, int, Path], None] | None = None,
) -> dict:
    """Extract PDFs and return success/failure details for observability."""
    papers_path = Path(papers_dir)
    texts_path = Path(texts_dir)

    if not papers_path.exists():
        papers_path.mkdir(parents=True, exist_ok=True)
        return {"success": [], "failures": [], "total": 0}

    texts_path.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(papers_path.glob("*.pdf"))
    output_paths = []
    failures = []

    for index, pdf_path in enumerate(pdf_paths, start=1):
        if progress_callback:
            progress_callback(index, len(pdf_paths), pdf_path)
        try:
            text = extract_pdf_text(pdf_path)
            output_path = texts_path / f"{pdf_path.stem}.txt"
            output_path.write_text(text, encoding="utf-8")
            output_paths.append(output_path)
        except Exception as exc:
            failures.append({"pdf": str(pdf_path), "error": str(exc)})

    return {"success": output_paths, "failures": failures, "total": len(pdf_paths)}
