import csv
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from research_agent.metadata_db import upsert_paper


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class ArxivResult:
    title: str
    authors: str
    year: str
    published: str
    summary: str
    entry_url: str
    pdf_url: str
    source: str = "arXiv"


def search_arxiv(query: str, max_results: int = 20) -> list[ArxivResult]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if max_results <= 0:
        raise ValueError("max_results must be greater than 0")

    try:
        response = requests.get(
            ARXIV_API_URL,
            params={"search_query": query, "start": 0, "max_results": max_results},
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"arXiv request failed: {exc}") from exc

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as exc:
        raise RuntimeError(f"arXiv XML parse failed: {exc}") from exc

    results = []
    for entry in root.findall("atom:entry", ATOM_NS):
        published = _text(entry, "atom:published")
        results.append(
            ArxivResult(
                title=_normalize_space(_text(entry, "atom:title")),
                authors=", ".join(
                    _normalize_space(_text(author, "atom:name"))
                    for author in entry.findall("atom:author", ATOM_NS)
                ),
                year=published[:4],
                published=published,
                summary=_normalize_space(_text(entry, "atom:summary")),
                entry_url=_entry_url(entry),
                pdf_url=_pdf_url(entry),
            )
        )
    return results


def save_arxiv_results(results: list[ArxivResult], output_csv: Path) -> Path:
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "title",
        "authors",
        "year",
        "published",
        "summary",
        "entry_url",
        "pdf_url",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    return path


def import_arxiv_results_to_db(results: list[ArxivResult], db_path: Path) -> int:
    count = 0
    for result in results:
        upsert_paper(
            db_path,
            {
                "title": result.title,
                "authors": result.authors,
                "year": result.year,
                "abstract": result.summary,
                "url": result.entry_url,
                "source": result.source,
                "notes": "imported_from_arxiv",
            },
        )
        count += 1
    return count


def download_arxiv_pdfs(
    results: list[ArxivResult],
    output_dir: Path,
    limit: int | None = None,
) -> list[Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    selected = results[:limit] if limit is not None else results
    downloaded = []
    for result in selected:
        if not result.pdf_url:
            continue
        try:
            response = requests.get(result.pdf_url, timeout=60)
            response.raise_for_status()
            path = target_dir / f"{_safe_filename(result.title)}.pdf"
            path.write_bytes(response.content)
            downloaded.append(path)
        except requests.RequestException:
            continue
    return downloaded


def _text(element: ET.Element, selector: str) -> str:
    found = element.find(selector, ATOM_NS)
    return found.text if found is not None and found.text else ""


def _entry_url(entry: ET.Element) -> str:
    for link in entry.findall("atom:link", ATOM_NS):
        if link.attrib.get("rel") == "alternate":
            return link.attrib.get("href", "")
    return _text(entry, "atom:id")


def _pdf_url(entry: ET.Element) -> str:
    for link in entry.findall("atom:link", ATOM_NS):
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            return link.attrib.get("href", "")
    return ""


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _safe_filename(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return name[:160] or "arxiv_paper"
