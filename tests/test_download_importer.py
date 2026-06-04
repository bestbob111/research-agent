from pathlib import Path

from research_agent import download_importer
from research_agent.download_importer import import_downloaded_pdfs
from research_agent.metadata_db import list_papers


def _config(tmp_path: Path) -> dict:
    return {
        "downloads_dir": str(tmp_path / "downloads"),
        "papers_dir": str(tmp_path / "papers"),
        "texts_dir": str(tmp_path / "texts"),
        "metadata_dir": str(tmp_path / "metadata"),
    }


def test_import_arxiv_pdf_copies_to_papers_and_writes_text(tmp_path, monkeypatch):
    config = _config(tmp_path)
    source_pdf = tmp_path / "downloads" / "arxiv" / "paper.pdf"
    source_pdf.parent.mkdir(parents=True)
    source_pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(
        download_importer,
        "extract_pdf_text",
        lambda pdf_path: f"text from {Path(pdf_path).name}",
    )

    result = import_downloaded_pdfs(config, source="arxiv")

    assert result["found"] == 1
    assert result["imported"] == 1
    assert source_pdf.exists()
    assert Path(result["papers"][0]).exists()
    text_path = Path(result["texts"][0])
    assert text_path.exists()
    assert text_path.read_text(encoding="utf-8") == "text from paper.pdf"


def test_import_downloaded_pdf_writes_metadata(tmp_path, monkeypatch):
    config = _config(tmp_path)
    source_pdf = tmp_path / "downloads" / "import" / "manual.pdf"
    source_pdf.parent.mkdir(parents=True)
    source_pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(download_importer, "extract_pdf_text", lambda pdf_path: "text")

    import_downloaded_pdfs(config, source="import")

    papers = list_papers(tmp_path / "metadata" / "papers.sqlite")
    assert len(papers) == 1
    assert papers[0]["title"] == "manual"
    assert papers[0]["source"] == "manual_pdf"
    assert papers[0]["file_path"].endswith("manual.pdf")
    assert papers[0]["text_path"].endswith("manual.txt")


def test_import_downloaded_pdf_avoids_overwriting_existing_file(tmp_path, monkeypatch):
    config = _config(tmp_path)
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    existing_pdf = papers_dir / "paper.pdf"
    existing_pdf.write_bytes(b"existing")
    source_pdf = tmp_path / "downloads" / "arxiv" / "paper.pdf"
    source_pdf.parent.mkdir(parents=True)
    source_pdf.write_bytes(b"new")
    monkeypatch.setattr(download_importer, "extract_pdf_text", lambda pdf_path: "text")

    result = import_downloaded_pdfs(config, source="arxiv")

    assert existing_pdf.read_bytes() == b"existing"
    assert Path(result["papers"][0]).name == "paper_1.pdf"
    assert (papers_dir / "paper_1.pdf").read_bytes() == b"new"


def test_import_downloaded_pdf_single_failure_does_not_stop_others(tmp_path, monkeypatch):
    config = _config(tmp_path)
    downloads_dir = tmp_path / "downloads" / "arxiv"
    downloads_dir.mkdir(parents=True)
    (downloads_dir / "bad.pdf").write_bytes(b"%PDF bad")
    (downloads_dir / "good.pdf").write_bytes(b"%PDF good")

    def fake_extract(pdf_path):
        if Path(pdf_path).name == "bad.pdf":
            raise RuntimeError("broken pdf")
        return "good text"

    monkeypatch.setattr(download_importer, "extract_pdf_text", fake_extract)

    result = import_downloaded_pdfs(config, source="arxiv")

    assert result["found"] == 2
    assert result["imported"] == 1
    assert len(result["failed"]) == 1
    assert result["failed"][0]["file"].endswith("bad.pdf")
    assert "broken pdf" in result["failed"][0]["error"]
    assert Path(result["texts"][0]).read_text(encoding="utf-8") == "good text"


def test_import_downloaded_pdf_move_removes_original(tmp_path, monkeypatch):
    config = _config(tmp_path)
    source_pdf = tmp_path / "downloads" / "arxiv" / "paper.pdf"
    source_pdf.parent.mkdir(parents=True)
    source_pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(download_importer, "extract_pdf_text", lambda pdf_path: "text")

    result = import_downloaded_pdfs(config, source="arxiv", mode="move")

    assert result["imported"] == 1
    assert not source_pdf.exists()
    assert Path(result["papers"][0]).exists()
