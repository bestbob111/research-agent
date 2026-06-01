from pathlib import Path

from research_agent.metadata_db import upsert_paper


def record_ingested_metadata(config: dict, txt_paths: list[Path]) -> int:
    papers_dir = Path(config["papers_dir"])
    db_path = Path(config["metadata_dir"]) / "papers.sqlite"
    pdf_by_stem = {pdf.stem: pdf for pdf in papers_dir.glob("*.pdf")}
    count = 0

    for txt_path in txt_paths:
        pdf_path = pdf_by_stem.get(txt_path.stem)
        if not pdf_path:
            continue
        upsert_paper(
            db_path,
            {
                "title": pdf_path.stem,
                "file_path": str(pdf_path.resolve()),
                "text_path": str(txt_path.resolve()),
                "source": "local_pdf",
            },
        )
        count += 1
    return count
