import pytest

from research_agent.topic_manager import (
    create_research_topic,
    generate_topic_plan,
    generate_topic_report,
    list_research_topics,
    show_research_topic,
)
from research_agent.metadata_db import get_topic


PLAN_MARKDOWN = """# 研究主题计划：主题 A

## 1. 主题描述
描述 A

## 2. 研究目标
目标内容

## 3. 核心科学问题
- 问题 1

## 4. 技术路线拆解
- 物理机制

## 5. 中文关键词
- 中文关键词 1
- 中文关键词 2

## 6. 英文关键词
- English keyword 1
- English keyword 2

## 7. 排除词和容易误检方向
- 排除词

## 8. 优先阅读文献类型
- 综述

## 9. 当前本地文献库中的初步线索
- paper.txt / chunk_0

## 10. 下一步行动计划
- 行动 1

## 11. 预期输出
- 文献清单
"""

REPORT_MARKDOWN = """# 主题 A 调研报告

## 1. 研究背景与问题定义
背景

## 2. 核心结论摘要
结论 [证据 1]

## 3. 文献证据概览

## 4. 技术路线分类

## 5. 主要噪声来源/关键问题

## 6. 方法与抑制策略对比

## 7. 代表性文献与贡献

## 8. 当前证据不足与不确定点

## 9. 后续检索关键词

## 10. 下一步研究建议
"""


class FakeOllamaClient:
    calls = 0

    def __init__(self, base_url, chat_model, embedding_model):
        self.base_url = base_url
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def chat(self, prompt):
        FakeOllamaClient.calls += 1
        if "正式 Markdown 科研调研报告" in prompt:
            return REPORT_MARKDOWN
        return PLAN_MARKDOWN


class FakeDeepSeekClient:
    calls = 0

    def __init__(self, api_key, base_url, model):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def chat(self, prompt):
        FakeDeepSeekClient.calls += 1
        return REPORT_MARKDOWN


class FakeLocalRAG:
    calls = []
    results = [
        {
            "text": "本地证据片段",
            "metadata": {"source": "paper.txt", "chunk_id": "chunk_0"},
            "distance": 0.2,
        }
    ]

    def __init__(self, config):
        self.config = config

    def retrieve(self, question, n_results=10):
        FakeLocalRAG.calls.append((question, n_results))
        return FakeLocalRAG.results


def make_config(tmp_path):
    return {
        "metadata_dir": str(tmp_path / "metadata"),
        "ollama_base_url": "http://localhost:11434",
        "chat_model": "qwen3:14b",
        "embedding_model": "bge-m3",
        "deepseek_base_url": "https://api.deepseek.com",
        "deepseek_model": "deepseek-chat",
        "reports_dir": str(tmp_path / "reports"),
    }


def test_create_topic_writes_database(tmp_path):
    topic = create_research_topic(
        make_config(tmp_path),
        "SERF 原子磁强计噪声抑制",
        "调研噪声来源和抑制方法",
    )

    topics = list_research_topics(make_config(tmp_path))

    assert topic["id"] == 1
    assert topics[0]["name"] == "SERF 原子磁强计噪声抑制"


def test_create_topic_writes_markdown_file(tmp_path):
    topic = create_research_topic(make_config(tmp_path), "主题 A", "描述 A")

    markdown_path = tmp_path / "metadata" / "topics" / "topic_1.md"
    text = markdown_path.read_text(encoding="utf-8")

    assert topic["markdown_path"] == str(markdown_path)
    assert "# 主题 A" in text
    assert "## 1. 主题描述" in text
    assert "## 2. 当前状态" in text
    assert "- 状态：planning" in text
    assert "## 6. 关联文件" in text
    assert "## 8. 下一步" in text


def test_list_and_show_topics(tmp_path):
    config = make_config(tmp_path)
    create_research_topic(config, "主题 A")

    topics = list_research_topics(config)
    topic = show_research_topic(config, 1)

    assert len(topics) == 1
    assert topic is not None
    assert topic["name"] == "主题 A"


def test_show_missing_topic_returns_none(tmp_path):
    assert show_research_topic(make_config(tmp_path), 999) is None


def test_generate_topic_plan_creates_markdown_and_updates_keywords(
    tmp_path,
    monkeypatch,
):
    from research_agent import topic_manager

    config = make_config(tmp_path)
    topic = create_research_topic(config, "主题 A", "描述 A")
    FakeOllamaClient.calls = 0
    FakeLocalRAG.calls = []
    monkeypatch.setattr(topic_manager, "OllamaClient", FakeOllamaClient)
    monkeypatch.setattr(topic_manager, "LocalRAG", FakeLocalRAG)

    result = generate_topic_plan(config, topic["id"])

    plan_path = tmp_path / "metadata" / "topics" / "topic_1_plan.md"
    db_topic = get_topic(tmp_path / "metadata" / "papers.sqlite", topic["id"])
    assert result["plan_path"] == str(plan_path)
    assert plan_path.exists()
    assert "# 研究主题计划：主题 A" in plan_path.read_text(encoding="utf-8")
    assert "中文关键词 1" in db_topic["keywords_cn"]
    assert "English keyword 1" in db_topic["keywords_en"]
    assert FakeOllamaClient.calls == 1
    assert FakeLocalRAG.calls[0][1] == 10


def test_generate_topic_plan_missing_topic_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        generate_topic_plan(make_config(tmp_path), 999)


def test_generate_topic_plan_existing_without_force_does_not_call_ollama(
    tmp_path,
    monkeypatch,
):
    from research_agent import topic_manager

    config = make_config(tmp_path)
    topic = create_research_topic(config, "主题 A")
    plan_path = tmp_path / "metadata" / "topics" / "topic_1_plan.md"
    plan_path.write_text("existing plan", encoding="utf-8")
    FakeOllamaClient.calls = 0
    FakeLocalRAG.calls = []
    monkeypatch.setattr(topic_manager, "OllamaClient", FakeOllamaClient)
    monkeypatch.setattr(topic_manager, "LocalRAG", FakeLocalRAG)

    result = generate_topic_plan(config, topic["id"], force=False)

    assert result["plan"] == "existing plan"
    assert FakeOllamaClient.calls == 0
    assert FakeLocalRAG.calls == []


def test_generate_topic_report_creates_report_file(tmp_path, monkeypatch):
    from research_agent import topic_manager

    config = make_config(tmp_path)
    topic = create_research_topic(config, "主题 A", "描述 A")
    plan_path = tmp_path / "metadata" / "topics" / "topic_1_plan.md"
    plan_path.write_text(PLAN_MARKDOWN, encoding="utf-8")
    FakeDeepSeekClient.calls = 0
    FakeLocalRAG.calls = []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(topic_manager, "DeepSeekClient", FakeDeepSeekClient)
    monkeypatch.setattr(topic_manager, "LocalRAG", FakeLocalRAG)

    result = generate_topic_report(config, topic["id"])

    report_path = result["report_path"]
    text = __import__("pathlib").Path(report_path).read_text(encoding="utf-8")
    assert __import__("pathlib").Path(report_path).exists()
    assert "# 主题 A 调研报告" in text
    assert "## 证据来源" in text
    assert result["evidence_count"] == 1
    assert result["unique_sources_count"] == 1
    assert FakeDeepSeekClient.calls == 1


def test_generate_topic_report_auto_generates_plan_when_missing(tmp_path, monkeypatch):
    from research_agent import topic_manager

    config = make_config(tmp_path)
    topic = create_research_topic(config, "主题 A", "描述 A")
    FakeOllamaClient.calls = 0
    FakeLocalRAG.calls = []
    monkeypatch.setattr(topic_manager, "OllamaClient", FakeOllamaClient)
    monkeypatch.setattr(topic_manager, "LocalRAG", FakeLocalRAG)

    result = generate_topic_report(config, topic["id"], use_deepseek=False)

    assert (tmp_path / "metadata" / "topics" / "topic_1_plan.md").exists()
    assert __import__("pathlib").Path(result["report_path"]).exists()
    assert FakeOllamaClient.calls >= 2


def test_generate_topic_report_updates_topic_markdown_links(tmp_path, monkeypatch):
    from research_agent import topic_manager

    config = make_config(tmp_path)
    topic = create_research_topic(config, "主题 A", "描述 A")
    (tmp_path / "metadata" / "topics" / "topic_1_plan.md").write_text(
        PLAN_MARKDOWN,
        encoding="utf-8",
    )
    monkeypatch.setattr(topic_manager, "OllamaClient", FakeOllamaClient)
    monkeypatch.setattr(topic_manager, "LocalRAG", FakeLocalRAG)

    result = generate_topic_report(config, topic["id"], use_deepseek=False)

    home = (tmp_path / "metadata" / "topics" / "topic_1.md").read_text(
        encoding="utf-8"
    )
    assert "检索计划：topic_1_plan.md" in home
    assert result["report_path"] in home


def test_generate_topic_report_missing_topic_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        generate_topic_report(make_config(tmp_path), 999)
