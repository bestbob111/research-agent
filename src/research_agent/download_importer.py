import re
import shutil
from pathlib import Path

from research_agent.metadata_db import upsert_paper
from research_agent.pdf_parser import extract_pdf_text


def import_downloaded_pdfs(
    config: dict,
    source: str = "all",
    mode: str = "copy",
) -> dict:
    if source not in {"all", "arxiv", "import"}:
        raise ValueError("source must be one of: all, arxiv, import")
    if mode not in {"copy", "move"}:
        raise ValueError("mode must be one of: copy, move")

    downloads_dir = Path(config["downloads_dir"])
    papers_dir = Path(config["papers_dir"])
    texts_dir = Path(config["texts_dir"])
    db_path = Path(config["metadata_dir"]) / "papers.sqlite"
    papers_dir.mkdir(parents=True, exist_ok=True)
    texts_dir.mkdir(parents=True, exist_ok=True)

    pdfs = _find_pdfs(downloads_dir, source)
    result = {"found": len(pdfs), "imported": 0, "failed": [], "papers": [], "texts": []}

    for source_name, pdf_path in pdfs:
        try:
            target_pdf = _unique_target_path(papers_dir, _safe_pdf_name(pdf_path.name))
            if mode == "move":
                shutil.move(str(pdf_path), target_pdf)
            else:
                shutil.copy2(pdf_path, target_pdf)

            text = extract_pdf_text(target_pdf)
            text_path = texts_dir / f"{target_pdf.stem}.txt"
            text_path.write_text(text, encoding="utf-8")
            upsert_paper(
                db_path,
                {
                    "title": target_pdf.stem,
                    "file_path": str(target_pdf.resolve()),
                    "text_path": str(text_path.resolve()),
                    "source": "arxiv_pdf" if source_name == "arxiv" else "manual_pdf",
                },
            )
            result["imported"] += 1
            result["papers"].append(str(target_pdf))
            result["texts"].append(str(text_path))
        except Exception as exc:
            result["failed"].append({"file": str(pdf_path), "error": str(exc)})

    return result


def _find_pdfs(downloads_dir: Path, source: str) -> list[tuple[str, Path]]:
    sources = ["arxiv", "import"] if source == "all" else [source]
    pdfs = []
    for source_name in sources:
        directory = downloads_dir / source_name
        if directory.exists():
            pdfs.extend((source_name, path) for path in sorted(directory.glob("*.pdf")))
    return pdfs


def _safe_pdf_name(name: str) -> str:
    stem = Path(name).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "paper"
    return f"{safe_stem[:160]}.pdf"


def _unique_target_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    index = 1
    while True:
        candidate = directory / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1
