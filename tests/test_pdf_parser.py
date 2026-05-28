from pathlib import Path

import pytest

from research_agent import pdf_parser
from research_agent.pdf_parser import extract_pdf_text, extract_pdfs_to_texts


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
