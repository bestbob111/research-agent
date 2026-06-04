import csv

import pytest
import requests

from research_agent.arxiv_searcher import (
    ArxivResult,
    download_arxiv_pdfs,
    import_arxiv_results_to_db,
    save_arxiv_results,
    search_arxiv,
)
from research_agent.metadata_db import list_papers


ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <updated>2024-01-02T00:00:00Z</updated>
    <published>2024-01-01T00:00:00Z</published>
    <title> SERF atomic magnetometer noise </title>
    <summary> Summary text. </summary>
    <author><name>Alice Author</name></author>
    <author><name>Bob Author</name></author>
    <link href="http://arxiv.org/abs/1234.5678v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/1234.5678v1" rel="related" type="application/pdf"/>
  </entry>
</feed>
"""


class FakeResponse:
    def __init__(self, text="", content=b"", status_code=200):
        self.text = text
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


def test_search_arxiv_parses_atom(monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params, timeout))
        return FakeResponse(text=ATOM_XML)

    monkeypatch.setattr("requests.get", fake_get)

    results = search_arxiv("SERF atomic magnetometer noise", max_results=1)

    assert len(results) == 1
    assert results[0].title == "SERF atomic magnetometer noise"
    assert results[0].authors == "Alice Author, Bob Author"
    assert results[0].year == "2024"
    assert results[0].pdf_url == "http://arxiv.org/pdf/1234.5678v1"
    assert calls[0][1]["max_results"] == 1


def test_search_arxiv_empty_query_raises_value_error():
    with pytest.raises(ValueError):
        search_arxiv(" ")


def test_search_arxiv_max_results_must_be_positive():
    with pytest.raises(ValueError):
        search_arxiv("query", max_results=0)


def test_search_arxiv_http_error_becomes_runtime_error(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return FakeResponse(status_code=500)

    monkeypatch.setattr("requests.get", fake_get)

    with pytest.raises(RuntimeError):
        search_arxiv("query")


def test_save_arxiv_results_writes_csv(tmp_path):
    output_csv = tmp_path / "metadata" / "arxiv.csv"
    result = ArxivResult(
        title="Title",
        authors="Author",
        year="2024",
        published="2024-01-01",
        summary="Summary",
        entry_url="http://example.com/abs",
        pdf_url="http://example.com/pdf",
    )

    save_arxiv_results([result], output_csv)

    with output_csv.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["title"] == "Title"
    assert rows[0]["source"] == "arXiv"


def test_import_arxiv_results_to_db(tmp_path):
    db_path = tmp_path / "papers.sqlite"
    result = ArxivResult(
        title="Title",
        authors="Author",
        year="2024",
        published="2024-01-01",
        summary="Summary",
        entry_url="http://example.com/abs",
        pdf_url="http://example.com/pdf",
    )

    count = import_arxiv_results_to_db([result], db_path)

    papers = list_papers(db_path)
    assert count == 1
    assert papers[0]["title"] == "Title"
    assert papers[0]["source"] == "arXiv"
    assert papers[0]["notes"] == "imported_from_arxiv"


def test_download_arxiv_pdfs(monkeypatch, tmp_path):
    def fake_get(url, timeout=None):
        return FakeResponse(content=b"%PDF")

    monkeypatch.setattr("requests.get", fake_get)
    result = ArxivResult(
        title="Title / With Unsafe Chars",
        authors="Author",
        year="2024",
        published="2024-01-01",
        summary="Summary",
        entry_url="http://example.com/abs",
        pdf_url="http://example.com/pdf",
    )

    paths = download_arxiv_pdfs([result], tmp_path / "downloads")

    assert len(paths) == 1
    assert paths[0].read_bytes() == b"%PDF"
