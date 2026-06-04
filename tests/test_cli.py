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
    (metadata_dir / "index_state.sqlite").write_text("", encoding="utf-8")

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(cli, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(cli, "count_indexed_files", lambda db_path: 5)
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
    assert "indexed_files: 5" in output
    assert "deepseek_api_key_configured: yes" in output
    assert "chroma_collection_count: 7" in output


def test_cli_index_parses_incremental_and_force(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda: {
            "texts_dir": "/tmp/texts",
            "chroma_dir": "/tmp/chroma",
            "collection_name": "test_collection",
            "metadata_dir": "/tmp/metadata",
        },
    )
    monkeypatch.setattr(
        cli,
        "build_index",
        lambda config, reset=False, limit=None, max_chars_per_embed=1800, incremental=False, force=False: calls.append(
            (reset, limit, max_chars_per_embed, incremental, force)
        )
        or {
            "processed_files": 1,
            "written_chunks": 2,
            "collection_count": 2,
            "skipped_empty_files": 0,
            "skipped_unchanged_files": 3,
            "failed_files": [],
            "failed_chunks": [],
        },
    )

    cli.main(
        [
            "index",
            "--incremental",
            "--force",
            "--limit",
            "5",
            "--max-chars-per-embed",
            "1000",
        ]
    )

    output = capsys.readouterr().out
    assert "skipped unchanged files: 3" in output
    assert calls == [(False, 5, 1000, True, True)]


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


def test_cli_topic_create_parses_args(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda: {"metadata_dir": "/tmp/metadata"})
    monkeypatch.setattr(
        cli,
        "create_research_topic",
        lambda config, name, description="": {
            "id": 1,
            "name": name,
            "description": description,
            "markdown_path": "/tmp/metadata/topics/topic_1.md",
        },
    )

    cli.main(["topic", "create", "主题名称", "--description", "描述"])

    output = capsys.readouterr().out
    assert "topic id: 1" in output
    assert "topic_1.md" in output


def test_cli_topic_list_parses_args(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda: {"metadata_dir": "/tmp/metadata"})
    monkeypatch.setattr(
        cli,
        "list_research_topics",
        lambda config: [{"id": 1, "name": "主题名称", "created_at": "now"}],
    )

    cli.main(["topic", "list"])

    output = capsys.readouterr().out
    assert "topics: 1" in output
    assert "主题名称" in output


def test_cli_topic_show_parses_args(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda: {"metadata_dir": "/tmp/metadata"})
    monkeypatch.setattr(
        cli,
        "show_research_topic",
        lambda config, topic_id: {"id": topic_id, "name": "主题名称"},
    )

    cli.main(["topic", "show", "1"])

    output = capsys.readouterr().out
    assert "id: 1" in output
    assert "name: 主题名称" in output


def test_cli_topic_plan_parses_args(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "load_config", lambda: {"metadata_dir": "/tmp/metadata"})
    monkeypatch.setattr(
        cli,
        "generate_topic_plan",
        lambda config, topic_id, use_deepseek=False, force=False, n_evidence=10: calls.append(
            (topic_id, use_deepseek, force, n_evidence)
        )
        or {
            "topic": {"id": topic_id},
            "plan_path": "/tmp/metadata/topics/topic_1_plan.md",
            "plan": "plan",
        },
    )

    cli.main(["topic", "plan", "1", "--force", "--n-evidence", "15", "--deepseek"])

    output = capsys.readouterr().out
    assert "plan generated:" in output
    assert "topic_1_plan.md" in output
    assert calls == [(1, True, True, 15)]


def test_cli_topic_report_parses_args(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "load_config", lambda: {"metadata_dir": "/tmp/metadata"})
    monkeypatch.setattr(
        cli,
        "generate_topic_report",
        lambda config, topic_id, use_deepseek=True, n_evidence=20, force=False: calls.append(
            (topic_id, use_deepseek, n_evidence, force)
        )
        or {
            "topic": {"id": topic_id, "name": "主题名称"},
            "plan_path": "/tmp/metadata/topics/topic_1_plan.md",
            "report_path": "/tmp/reports/topic_1_report_20260101_000000.md",
            "evidence_count": 3,
            "unique_sources_count": 2,
        },
    )

    cli.main(["topic", "report", "1", "--local", "--force", "--n-evidence", "30"])

    output = capsys.readouterr().out
    assert "topic id: 1" in output
    assert "report_path:" in output
    assert "evidence_count: 3" in output
    assert calls == [(1, False, 30, True)]


def test_cli_explore_parses_args(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "load_config", lambda: {"metadata_dir": "/tmp/metadata"})
    monkeypatch.setattr(
        cli,
        "explore_question",
        lambda config, question, rounds=2, n_evidence=8, use_deepseek=True, topic_id=None, assess_with_deepseek=False: calls.append(
            (question, rounds, n_evidence, use_deepseek, topic_id, assess_with_deepseek)
        )
        or {
            "answer": "answer",
            "subquestions": ["sub"],
            "search_queries": ["query"],
            "possible_gaps": [],
            "sources": [{"source": "paper.txt"}],
            "report_path": "/tmp/report.md",
            "json_path": "/tmp/explore.json",
        },
    )

    cli.main(
        [
            "explore",
            "问题",
            "--local",
            "--rounds",
            "3",
            "--n-evidence",
            "10",
            "--topic-id",
            "1",
            "--assess-with-deepseek",
        ]
    )

    output = capsys.readouterr().out
    assert "用户问题: 问题" in output
    assert "report_path: /tmp/report.md" in output
    assert calls == [("问题", 3, 10, False, 1, True)]


def test_cli_explore_list_parses_args(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda: {"metadata_dir": "/tmp/metadata"})
    monkeypatch.setattr(
        cli,
        "list_explorations",
        lambda config: [
            {
                "id": "explore_1",
                "created_at": "now",
                "topic_id": None,
                "sources_count": 2,
                "question": "问题",
                "report_path": "/tmp/report.md",
            }
        ],
    )

    cli.main(["explore", "list"])

    output = capsys.readouterr().out
    assert "explore_1" in output
    assert "sources=2" in output


def test_cli_explore_show_parses_args(monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda: {"metadata_dir": "/tmp/metadata"})
    monkeypatch.setattr(
        cli,
        "load_exploration",
        lambda config, explore_id: {
            "question": "问题",
            "topic_id": 1,
            "created_at": "now",
            "subquestions": ["子问题"],
            "search_queries": ["查询"],
            "possible_gaps": ["缺口"],
            "answer": "答案",
            "report_path": "/tmp/report.md",
            "json_path": "/tmp/explore.json",
        },
    )

    cli.main(["explore", "show", "explore_1"])

    output = capsys.readouterr().out
    assert "question: 问题" in output
    assert "json_path: /tmp/explore.json" in output


def test_cli_explore_continue_parses_args(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "load_config", lambda: {"metadata_dir": "/tmp/metadata"})
    monkeypatch.setattr(
        cli,
        "continue_exploration",
        lambda config, explore_id, rounds=1, n_evidence=8, use_deepseek=True, assess_with_deepseek=False: calls.append(
            (explore_id, rounds, n_evidence, use_deepseek, assess_with_deepseek)
        )
        or {
            "answer": "answer",
            "sources": [{"source": "paper.txt"}],
            "report_path": "/tmp/report.md",
            "json_path": "/tmp/explore.json",
        },
    )

    cli.main(
        [
            "explore",
            "continue",
            "explore_1",
            "--local",
            "--rounds",
            "2",
            "--assess-with-deepseek",
        ]
    )

    output = capsys.readouterr().out
    assert "continued_from: explore_1" in output
    assert calls == [("explore_1", 2, 8, False, True)]


def test_cli_arxiv_search_parses_args(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda: {
            "metadata_dir": "/tmp/metadata",
            "downloads_dir": "/tmp/downloads",
        },
    )
    monkeypatch.setattr(
        cli,
        "search_arxiv",
        lambda query, max_results=20: calls.append(("search", query, max_results))
        or [],
    )
    monkeypatch.setattr(
        cli,
        "save_arxiv_results",
        lambda results, output_csv: calls.append(("save", str(output_csv))) or output_csv,
    )
    monkeypatch.setattr(
        cli,
        "import_arxiv_results_to_db",
        lambda results, db_path: calls.append(("import", str(db_path))) or 0,
    )
    monkeypatch.setattr(
        cli,
        "download_arxiv_pdfs",
        lambda results, output_dir, limit=None: calls.append(
            ("download", str(output_dir), limit)
        )
        or [],
    )

    cli.main(
        [
            "arxiv",
            "search",
            "SERF atomic magnetometer noise",
            "--max-results",
            "20",
            "--download",
            "--download-limit",
            "3",
        ]
    )

    output = capsys.readouterr().out
    assert "results: 0" in output
    assert ("search", "SERF atomic magnetometer noise", 20) in calls
    assert ("download", "/tmp/downloads/arxiv", 3) in calls


def test_cli_import_downloads_parses_args(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        cli,
        "load_config",
        lambda: {
            "downloads_dir": "/tmp/downloads",
            "papers_dir": "/tmp/papers",
            "texts_dir": "/tmp/texts",
            "metadata_dir": "/tmp/metadata",
        },
    )
    monkeypatch.setattr(
        cli,
        "import_downloaded_pdfs",
        lambda config, source="all", mode="copy": calls.append((source, mode))
        or {
            "found": 1,
            "imported": 1,
            "failed": [],
            "papers": ["/tmp/papers/paper.pdf"],
            "texts": ["/tmp/texts/paper.txt"],
        },
    )

    cli.main(["import-downloads", "--source", "arxiv", "--move"])

    output = capsys.readouterr().out
    assert "found: 1" in output
    assert "imported: 1" in output
    assert "paper: /tmp/papers/paper.pdf" in output
    assert calls == [("arxiv", "move")]
