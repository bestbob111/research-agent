from pathlib import Path

import pytest

from research_agent import pdf_parser
from research_agent.pdf_parser import (
    extract_pdf_text,
    extract_pdfs_to_texts,
    extract_pdfs_to_texts_with_report,
)
from research_agent.metadata_db import list_papers
from research_agent.ingest_metadata import record_ingested_metadata


def test_extract_pdfs_to_texts_creates_missing_papers_dir(tmp_path):
    papers_dir = tmp_path / "papers"
    texts_dir = tmp_path / "texts"

    output_paths = extract_pdfs_to_texts(papers_dir, texts_dir)

    assert output_paths == []
    assert papers_dir.exists()


def test_extract_pdf_text_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_pdf_text(tmp_path / "missing.pdf")


def test_extract_pdf_text_directory_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        extract_pdf_text(tmp_path)


def test_extract_pdfs_to_texts_writes_txt_files(tmp_path, monkeypatch):
    papers_dir = tmp_path / "papers"
    texts_dir = tmp_path / "texts"
    papers_dir.mkdir()
    pdf_path = papers_dir / "paper.pdf"
    pdf_path.write_bytes(b"%PDF test")

    def fake_extract_pdf_text(path: Path) -> str:
        assert path == pdf_path
        return "--- Page 1 ---\ncontent"

    monkeypatch.setattr(pdf_parser, "extract_pdf_text", fake_extract_pdf_text)

    output_paths = extract_pdfs_to_texts(papers_dir, texts_dir)

    assert output_paths == [texts_dir / "paper.txt"]
    assert (texts_dir / "paper.txt").read_text(encoding="utf-8") == (
        "--- Page 1 ---\ncontent"
    )


def test_extract_pdfs_to_texts_with_report_records_success_and_failure(
    tmp_path,
    monkeypatch,
):
    papers_dir = tmp_path / "papers"
    texts_dir = tmp_path / "texts"
    papers_dir.mkdir()
    good_pdf = papers_dir / "good.pdf"
    bad_pdf = papers_dir / "bad.pdf"
    good_pdf.write_bytes(b"%PDF good")
    bad_pdf.write_bytes(b"%PDF bad")

    def fake_extract_pdf_text(path: Path) -> str:
        if path == bad_pdf:
            raise RuntimeError("broken pdf")
        return "--- Page 1 ---\ncontent"

    monkeypatch.setattr(pdf_parser, "extract_pdf_text", fake_extract_pdf_text)

    report = extract_pdfs_to_texts_with_report(papers_dir, texts_dir)

    assert report["total"] == 2
    assert report["success"] == [texts_dir / "good.txt"]
    assert report["failures"] == [{"pdf": str(bad_pdf), "error": "broken pdf"}]


def test_record_ingested_metadata_writes_sqlite(tmp_path):
    papers_dir = tmp_path / "papers"
    texts_dir = tmp_path / "texts"
    metadata_dir = tmp_path / "metadata"
    papers_dir.mkdir()
    texts_dir.mkdir()
    pdf_path = papers_dir / "paper.pdf"
    txt_path = texts_dir / "paper.txt"
    pdf_path.write_bytes(b"%PDF")
    txt_path.write_text("content", encoding="utf-8")
    config = {
        "papers_dir": str(papers_dir),
        "metadata_dir": str(metadata_dir),
    }

    count = record_ingested_metadata(config, [txt_path])

    papers = list_papers(metadata_dir / "papers.sqlite")
    assert count == 1
    assert papers[0]["title"] == "paper"
    assert papers[0]["source"] == "local_pdf"
    assert papers[0]["file_path"] == str(pdf_path.resolve())
    assert papers[0]["text_path"] == str(txt_path.resolve())
