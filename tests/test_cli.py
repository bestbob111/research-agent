import argparse

import pytest

from research_agent import cli


class FakeVectorStore:
    def __init__(self, chroma_dir, collection_name):
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name

    def count(self):
        return 7


def test_cli_help_exits_successfully(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])

    assert exc.value.code == 0
    assert "research-agent" in capsys.readouterr().out


def test_cli_status_runs_with_mocked_config(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    papers_dir = data_dir / "papers"
    texts_dir = data_dir / "texts"
    chroma_dir = data_dir / "chroma"
    metadata_dir = data_dir / "metadata"
    reports_dir = data_dir / "reports"
    for path in (papers_dir, texts_dir, chroma_dir, metadata_dir, reports_dir):
        path.mkdir(parents=True)
    (papers_dir / "a.pdf").write_bytes(b"%PDF")
    (texts_dir / "a.txt").write_text("text", encoding="utf-8")
    (metadata_dir / "papers.sqlite").write_text("", encoding="utf-8")

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(cli, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda: {
            "data_dir": str(data_dir),
            "papers_dir": str(papers_dir),
            "texts_dir": str(texts_dir),
            "chroma_dir": str(chroma_dir),
            "metadata_dir": str(metadata_dir),
            "reports_dir": str(reports_dir),
            "collection_name": "test_collection",
        },
    )

    cli.main(["status"])

    output = capsys.readouterr().out
    assert "pdf_count: 1" in output
    assert "txt_count: 1" in output
    assert "metadata_db_exists: yes" in output
    assert "deepseek_api_key_configured: yes" in output
    assert "chroma_collection_count: 7" in output


def test_cli_missing_required_argument_returns_error():
    with pytest.raises(SystemExit) as exc:
        cli.main(["query"])

    assert exc.value.code != 0


def test_cli_papers_list_limit_runs(tmp_path, monkeypatch, capsys):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda: {"metadata_dir": str(metadata_dir)},
    )
    monkeypatch.setattr(cli, "init_db", lambda db_path: None)
    monkeypatch.setattr(
        cli,
        "list_papers",
        lambda db_path: [
            {"id": 1, "title": "A", "file_path": "/a.pdf", "text_path": "/a.txt"},
            {"id": 2, "title": "B", "file_path": "/b.pdf", "text_path": "/b.txt"},
        ],
    )

    cli.main(["papers", "list", "--limit", "1"])

    output = capsys.readouterr().out
    assert "papers: 2" in output
    assert "A" in output
    assert "B" not in output


def test_cli_papers_list_limit_zero_errors():
    with pytest.raises(SystemExit) as exc:
        cli.main(["papers", "list", "--limit", "0"])

    assert exc.value.code != 0


def test_cli_papers_list_negative_offset_errors():
    with pytest.raises(SystemExit) as exc:
        cli.main(["papers", "list", "--offset", "-1"])

    assert exc.value.code != 0


def test_cli_broken_pipe_is_handled(monkeypatch):
    parser = cli.build_parser()
    status_parser = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ).choices["status"]
    status_parser.set_defaults(func=lambda args: (_ for _ in ()).throw(BrokenPipeError()))
    monkeypatch.setattr(cli, "build_parser", lambda: parser)

    cli.main(["status"])
