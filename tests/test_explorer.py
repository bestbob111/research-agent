import json

import pytest

from research_agent import explorer
from research_agent.explorer import (
    continue_exploration,
    explore_question,
    list_explorations,
    load_exploration,
)
from research_agent.metadata_db import create_topic, init_db


class FakeLocalRAG:
    calls = []
    evidence = [
        {
            "text": "证据 A",
            "metadata": {"source": "paper.txt", "chunk_id": "c1"},
            "distance": 0.1,
        },
        {
            "text": "证据 A duplicate",
            "metadata": {"source": "paper.txt", "chunk_id": "c1"},
            "distance": 0.2,
        },
    ]

    def __init__(self, config):
        self.config = config

    def retrieve(self, question, n_results=8):
        FakeLocalRAG.calls.append((question, n_results))
        return FakeLocalRAG.evidence


class FakeDeepSeekClient:
    calls = []

    def __init__(self, api_key, base_url, model):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def chat(self, prompt):
        FakeDeepSeekClient.calls.append(prompt)
        if "请把科研问题拆解为 JSON" in prompt:
            return json.dumps(
                {
                    "subquestions": ["子问题 1"],
                    "search_queries": ["扩展检索 1"],
                    "possible_gaps": ["缺口 1"],
                },
                ensure_ascii=False,
            )
        return "最终回答 [证据 1]"


class FakeOllamaClient:
    calls = []

    def __init__(self, base_url, chat_model, embedding_model):
        self.base_url = base_url
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def chat(self, prompt):
        FakeOllamaClient.calls.append(prompt)
        if "请把科研问题拆解为 JSON" in prompt:
            return "not json"
        return "本地最终回答 [证据 1]"


def make_config(tmp_path):
    return {
        "metadata_dir": str(tmp_path / "metadata"),
        "reports_dir": str(tmp_path / "reports"),
        "deepseek_base_url": "https://api.deepseek.com",
        "deepseek_model": "deepseek-chat",
        "ollama_base_url": "http://localhost:11434",
        "chat_model": "qwen3:14b",
        "embedding_model": "bge-m3",
    }


def setup_fakes(monkeypatch):
    FakeLocalRAG.calls = []
    FakeDeepSeekClient.calls = []
    FakeOllamaClient.calls = []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(explorer, "LocalRAG", FakeLocalRAG)
    monkeypatch.setattr(explorer, "DeepSeekClient", FakeDeepSeekClient)
    monkeypatch.setattr(explorer, "OllamaClient", FakeOllamaClient)
    monkeypatch.setattr(
        explorer,
        "assess_evidence",
        lambda question, answer, sources, config, use_deepseek=False: {
            "overall_strength": "medium",
            "supported_claims": [
                {"claim": "结论", "evidence_ids": [1], "strength": "medium"}
            ],
            "weak_claims": [],
            "recommended_reading": [
                {
                    "title": "Paper A",
                    "source": "paper.txt",
                    "reason": "相关",
                    "priority": "high",
                }
            ],
            "possibly_irrelevant_sources": [],
        },
    )


def test_explore_empty_question_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        explore_question(make_config(tmp_path), " ")


def test_explore_deepseek_generates_report_and_json(tmp_path, monkeypatch):
    setup_fakes(monkeypatch)

    result = explore_question(make_config(tmp_path), "问题", use_deepseek=True)

    assert result["answer"] == "最终回答 [证据 1]"
    assert result["subquestions"] == ["子问题 1"]
    assert result["search_queries"] == ["扩展检索 1"]
    assert result["possible_gaps"] == ["缺口 1"]
    assert __import__("pathlib").Path(result["report_path"]).exists()
    assert __import__("pathlib").Path(result["json_path"]).exists()
    assert result["evidence_assessment"]["overall_strength"] == "medium"
    saved = json.loads(
        __import__("pathlib").Path(result["json_path"]).read_text(encoding="utf-8")
    )
    assert saved["evidence_assessment"]["overall_strength"] == "medium"
    report = __import__("pathlib").Path(result["report_path"]).read_text(
        encoding="utf-8"
    )
    assert "证据质量评估" in report
    assert "建议精读文献" in report


def test_explore_local_uses_ollama_and_fallback_json(tmp_path, monkeypatch):
    setup_fakes(monkeypatch)

    result = explore_question(make_config(tmp_path), "问题", use_deepseek=False)

    assert result["answer"] == "本地最终回答 [证据 1]"
    assert result["subquestions"] == ["问题"]
    assert result["search_queries"] == ["问题"]
    assert result["possible_gaps"] == []
    assert FakeOllamaClient.calls


def test_explore_deduplicates_evidence(tmp_path, monkeypatch):
    setup_fakes(monkeypatch)

    result = explore_question(make_config(tmp_path), "问题", use_deepseek=True)

    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "paper.txt"
    assert result["sources"][0]["chunk_id"] == "c1"


def test_explore_missing_topic_raises_value_error(tmp_path, monkeypatch):
    setup_fakes(monkeypatch)
    config = make_config(tmp_path)
    init_db(__import__("pathlib").Path(config["metadata_dir"]) / "papers.sqlite")

    with pytest.raises(ValueError):
        explore_question(config, "问题", topic_id=999)


def test_explore_with_topic_context(tmp_path, monkeypatch):
    setup_fakes(monkeypatch)
    config = make_config(tmp_path)
    db_path = __import__("pathlib").Path(config["metadata_dir"]) / "papers.sqlite"
    topic_id = create_topic(db_path, {"name": "主题", "description": "描述"})

    result = explore_question(config, "问题", topic_id=topic_id)

    assert result["answer"] == "最终回答 [证据 1]"


def test_list_explorations_returns_summaries(tmp_path):
    config = make_config(tmp_path)
    explorations_dir = __import__("pathlib").Path(config["metadata_dir"]) / "explorations"
    explorations_dir.mkdir(parents=True)
    (explorations_dir / "explore_20260601_123456.json").write_text(
        json.dumps(
            {
                "question": "问题",
                "topic_id": 1,
                "created_at": "2026-06-01T12:34:56",
                "report_path": "/tmp/report.md",
                "sources": [{"source": "paper.txt"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    records = list_explorations(config)

    assert records == [
        {
            "id": "explore_20260601_123456",
            "question": "问题",
            "topic_id": 1,
            "created_at": "2026-06-01T12:34:56",
            "report_path": "/tmp/report.md",
            "sources_count": 1,
        }
    ]


def test_load_exploration_missing_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_exploration(make_config(tmp_path), "missing")


def test_continue_exploration_reads_old_json_and_sets_continued_from(
    tmp_path,
    monkeypatch,
):
    setup_fakes(monkeypatch)
    config = make_config(tmp_path)
    explorations_dir = __import__("pathlib").Path(config["metadata_dir"]) / "explorations"
    explorations_dir.mkdir(parents=True)
    (explorations_dir / "explore_old.json").write_text(
        json.dumps(
            {
                "question": "原问题",
                "topic_id": None,
                "possible_gaps": ["缺口"],
                "search_queries": ["旧查询"],
                "sources": [
                    {
                        "text": "旧证据",
                        "source": "paper.txt",
                        "chunk_id": "c1",
                        "distance": 0.1,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = continue_exploration(config, "explore_old", use_deepseek=True)
    saved = json.loads(
        __import__("pathlib").Path(result["json_path"]).read_text(encoding="utf-8")
    )

    assert saved["continued_from"] == "explore_old"
    assert "继续探索以下问题" in saved["question"]


def test_continue_exploration_deduplicates_old_and_new_sources(tmp_path, monkeypatch):
    setup_fakes(monkeypatch)
    config = make_config(tmp_path)
    explorations_dir = __import__("pathlib").Path(config["metadata_dir"]) / "explorations"
    explorations_dir.mkdir(parents=True)
    (explorations_dir / "explore_old.json").write_text(
        json.dumps(
            {
                "question": "原问题",
                "topic_id": None,
                "possible_gaps": [],
                "search_queries": ["旧查询"],
                "sources": [
                    {
                        "text": "旧证据",
                        "source": "paper.txt",
                        "chunk_id": "c1",
                        "distance": 0.1,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = continue_exploration(config, "explore_old", use_deepseek=True)

    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "paper.txt"
    assert result["sources"][0]["chunk_id"] == "c1"
