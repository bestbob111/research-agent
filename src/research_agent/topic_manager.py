import os
from datetime import datetime
from pathlib import Path

from research_agent.config import load_project_dotenv
from research_agent.deepseek_client import DeepSeekClient
from research_agent.metadata_db import create_topic, get_topic, list_topics, update_topic
from research_agent.ollama_client import OllamaClient
from research_agent.rag import LocalRAG


def create_research_topic(
    config: dict,
    name: str,
    description: str = "",
) -> dict:
    if not name.strip():
        raise ValueError("topic name must not be empty")

    db_path = _db_path(config)
    topic_id = create_topic(
        db_path,
        {
            "name": name.strip(),
            "description": description,
            "keywords_cn": "",
            "keywords_en": "",
            "notes": "",
        },
    )
    topic = get_topic(db_path, topic_id)
    markdown_path = _topic_markdown_path(config, topic_id)
    _write_topic_markdown(markdown_path, topic or {})
    result = dict(topic or {})
    result["markdown_path"] = str(markdown_path)
    return result


def list_research_topics(config: dict) -> list[dict]:
    return list_topics(_db_path(config))


def show_research_topic(config: dict, topic_id: int) -> dict | None:
    topic = get_topic(_db_path(config), topic_id)
    if not topic:
        return None
    topic["markdown_path"] = str(_topic_markdown_path(config, topic_id))
    return topic


def generate_topic_plan(
    config: dict,
    topic_id: int,
    use_deepseek: bool = False,
    force: bool = False,
    n_evidence: int = 10,
) -> dict:
    topic = get_topic(_db_path(config), topic_id)
    if not topic:
        raise ValueError(f"topic not found: {topic_id}")

    plan_path = _topic_plan_path(config, topic_id)
    if plan_path.exists() and not force:
        return {
            "topic": topic,
            "plan_path": str(plan_path),
            "plan": plan_path.read_text(encoding="utf-8"),
            "message": "plan already exists",
        }

    evidence = _retrieve_topic_evidence(config, topic, n_evidence)
    client = _create_plan_client(config, use_deepseek)
    plan = client.chat(_build_plan_prompt(topic, evidence))
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(plan, encoding="utf-8")

    update_topic(
        _db_path(config),
        topic_id,
        {
            "keywords_cn": _extract_section(plan, "## 5. 中文关键词"),
            "keywords_en": _extract_section(plan, "## 6. 英文关键词"),
        },
    )
    updated_topic = get_topic(_db_path(config), topic_id) or topic
    return {"topic": updated_topic, "plan_path": str(plan_path), "plan": plan}


def generate_topic_report(
    config: dict,
    topic_id: int,
    use_deepseek: bool = True,
    n_evidence: int = 20,
    force: bool = False,
) -> dict:
    topic = get_topic(_db_path(config), topic_id)
    if not topic:
        raise ValueError(f"topic not found: {topic_id}")

    plan_path = _topic_plan_path(config, topic_id)
    if not plan_path.exists():
        generate_topic_plan(config, topic_id, force=False, n_evidence=10)
    plan = plan_path.read_text(encoding="utf-8") if plan_path.exists() else ""

    evidence = _retrieve_report_evidence(config, topic, plan, n_evidence)
    client = _create_plan_client(config, use_deepseek)
    report_body = client.chat(_build_report_prompt(topic, plan, evidence))
    report_path = _topic_report_path(config, topic_id)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _append_evidence_table(report_body, evidence),
        encoding="utf-8",
    )

    _write_topic_markdown(
        _topic_markdown_path(config, topic_id),
        topic,
        plan_path=plan_path,
        report_path=report_path,
    )
    return {
        "topic": topic,
        "plan_path": str(plan_path),
        "report_path": str(report_path),
        "evidence_count": len(evidence),
        "unique_sources_count": _count_unique_sources(evidence),
    }


def _db_path(config: dict) -> Path:
    return Path(config["metadata_dir"]) / "papers.sqlite"


def _topic_markdown_path(config: dict, topic_id: int) -> Path:
    return Path(config["metadata_dir"]) / "topics" / f"topic_{topic_id}.md"


def _topic_plan_path(config: dict, topic_id: int) -> Path:
    return Path(config["metadata_dir"]) / "topics" / f"topic_{topic_id}_plan.md"


def _topic_report_path(config: dict, topic_id: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(config["reports_dir"]) / f"topic_{topic_id}_report_{timestamp}.md"


def _write_topic_markdown(
    path: Path,
    topic: dict,
    plan_path: Path | None = None,
    report_path: Path | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            f"# {topic.get('name') or ''}",
            "",
            "## 1. 主题描述",
            "",
            topic.get("description") or "",
            "",
            "## 2. 当前状态",
            "",
            "- 状态：planning",
            f"- 创建时间：{topic.get('created_at') or ''}",
            f"- 更新时间：{topic.get('updated_at') or ''}",
            "",
            "## 3. 研究目标",
            "",
            "",
            "## 4. 中文关键词",
            "",
            topic.get("keywords_cn") or "",
            "",
            "## 5. 英文关键词",
            "",
            topic.get("keywords_en") or "",
            "",
            "## 6. 关联文件",
            "",
            f"- 检索计划：{plan_path.name if plan_path else ''}",
            f"- 调研报告：{str(report_path) if report_path else ''}",
            "- 文献清单：",
            "",
            "## 7. 笔记",
            "",
            topic.get("notes") or "",
            "",
            "## 8. 下一步",
            "",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _retrieve_topic_evidence(config: dict, topic: dict, n_evidence: int) -> list[dict]:
    question = (
        f"围绕 {topic.get('name') or ''}，调研 {topic.get('description') or ''}，"
        "需要哪些关键词、核心问题和代表文献？"
    )
    try:
        return LocalRAG(config).retrieve(question, n_results=n_evidence)
    except Exception:
        return []


def _retrieve_report_evidence(
    config: dict,
    topic: dict,
    plan: str,
    n_evidence: int,
) -> list[dict]:
    query = "\n".join(
        [
            topic.get("name") or "",
            topic.get("description") or "",
            topic.get("keywords_cn") or "",
            topic.get("keywords_en") or "",
            plan[:3000],
        ]
    )
    try:
        return LocalRAG(config).retrieve(query, n_results=n_evidence)
    except Exception:
        return []


def _create_plan_client(config: dict, use_deepseek: bool):
    if use_deepseek:
        load_project_dotenv()
        return DeepSeekClient(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=config["deepseek_base_url"],
            model=config["deepseek_model"],
        )
    return OllamaClient(
        base_url=config["ollama_base_url"],
        chat_model=config["chat_model"],
        embedding_model=config["embedding_model"],
    )


def _build_plan_prompt(topic: dict, evidence: list[dict]) -> str:
    evidence_text = _format_evidence(evidence)
    return (
        "请根据研究主题和当前本地文献库线索，生成一份 Markdown 科研检索计划和研究路线规划。"
        "不要编造不存在的文献；如果本地线索不足，请生成通用计划并明确说明。\n\n"
        f"主题名称：{topic.get('name') or ''}\n"
        f"主题描述：{topic.get('description') or ''}\n\n"
        f"当前本地文献库线索：\n{evidence_text}\n\n"
        "请严格输出以下结构：\n\n"
        f"# 研究主题计划：{topic.get('name') or ''}\n\n"
        "## 1. 主题描述\n"
        "写入 topic description。\n\n"
        "## 2. 研究目标\n"
        "说明该主题要解决什么科研问题。\n\n"
        "## 3. 核心科学问题\n"
        "列出 5-10 个问题。\n\n"
        "## 4. 技术路线拆解\n"
        "至少包括：物理机制、主要噪声来源、噪声抑制方法、实验系统设计、性能评价指标、典型应用场景。\n\n"
        "## 5. 中文关键词\n"
        "不少于 15 组中文检索关键词。\n\n"
        "## 6. 英文关键词\n"
        "不少于 15 组英文检索关键词。\n\n"
        "## 7. 排除词和容易误检方向\n"
        "说明哪些方向容易误检，例如无关的医学、图像处理、宏观磁传感器等。\n\n"
        "## 8. 优先阅读文献类型\n"
        "例如经典理论论文、综述论文、实验系统论文、噪声建模论文、仪器设计论文、应用验证论文。\n\n"
        "## 9. 当前本地文献库中的初步线索\n"
        "根据 LocalRAG.retrieve 返回的 sources，列出若干可能相关的文献标题/文件名/chunk_id。"
        "如果没有证据，写“当前本地文献库未检索到足够相关片段”。\n\n"
        "## 10. 下一步行动计划\n"
        "给出 7-10 个具体行动步骤，例如精读哪些类型文献、补充检索哪些关键词、"
        "建立哪些对比表、总结哪些公式/模型、查找哪些实验系统细节。\n\n"
        "## 11. 预期输出\n"
        "列出文献清单、技术路线图、噪声来源分类表、抑制方法对比表、综述报告、可选创新点。\n"
    )


def _build_report_prompt(topic: dict, plan: str, evidence: list[dict]) -> str:
    evidence_text = _format_evidence(evidence)
    return (
        "请基于当前本地文献库证据和研究计划，生成正式 Markdown 科研调研报告。"
        "只基于给定证据和 topic plan；不要编造不存在的文献；关键结论后标注 [证据 1]、[证据 2]；"
        "区分文献直接支持和模型推断；如果证据不足，明确说明。\n\n"
        f"报告问题：请基于当前本地文献库和研究计划，围绕主题《{topic.get('name') or ''}》"
        f"生成一份科研调研报告。主题描述：{topic.get('description') or ''}。\n\n"
        f"研究计划：\n{plan[:5000]}\n\n"
        f"证据：\n{evidence_text}\n\n"
        "请严格输出以下结构：\n\n"
        f"# {topic.get('name') or ''} 调研报告\n\n"
        "## 1. 研究背景与问题定义\n"
        "## 2. 核心结论摘要\n"
        "## 3. 文献证据概览\n"
        "## 4. 技术路线分类\n"
        "## 5. 主要噪声来源/关键问题\n"
        "## 6. 方法与抑制策略对比\n"
        "## 7. 代表性文献与贡献\n"
        "## 8. 当前证据不足与不确定点\n"
        "## 9. 后续检索关键词\n"
        "## 10. 下一步研究建议\n"
    )


def _format_evidence(evidence: list[dict]) -> str:
    if not evidence:
        return "当前本地文献库未检索到足够相关片段。"
    blocks = []
    for index, item in enumerate(evidence, start=1):
        metadata = item.get("metadata") or {}
        source = metadata.get("source") or item.get("source") or ""
        chunk_id = metadata.get("chunk_id") or item.get("chunk_id") or ""
        text = item.get("text") or ""
        blocks.append(f"[线索 {index} | source={source} | chunk_id={chunk_id}]\n{text[:1000]}")
    return "\n\n".join(blocks)


def _append_evidence_table(report_body: str, evidence: list[dict]) -> str:
    lines = [
        report_body.rstrip(),
        "",
        "## 证据来源",
        "",
        "| 编号 | source | chunk_id | distance |",
        "| --- | --- | --- | --- |",
    ]
    for index, item in enumerate(evidence, start=1):
        metadata = item.get("metadata") or {}
        source = metadata.get("source") or item.get("source") or ""
        chunk_id = metadata.get("chunk_id") or item.get("chunk_id") or ""
        distance = item.get("distance")
        lines.append(f"| {index} | {_cell(source)} | {_cell(chunk_id)} | {_cell(distance)} |")
    lines.append("")
    return "\n".join(lines)


def _cell(value) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _count_unique_sources(evidence: list[dict]) -> int:
    values = set()
    for item in evidence:
        metadata = item.get("metadata") or {}
        source = metadata.get("source") or item.get("source") or item.get("id")
        if source:
            values.add(source)
    return len(values)


def _extract_section(markdown: str, heading: str) -> str:
    lines = markdown.splitlines()
    capture = False
    section_lines = []
    for line in lines:
        if line.strip() == heading:
            capture = True
            continue
        if capture and line.startswith("## "):
            break
        if capture:
            section_lines.append(line)
    return "\n".join(section_lines).strip()
