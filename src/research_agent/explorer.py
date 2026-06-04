import json
import os
from datetime import datetime
from pathlib import Path

from research_agent.config import load_project_dotenv
from research_agent.deepseek_client import DeepSeekClient
from research_agent.evidence_assessor import assess_evidence
from research_agent.metadata_db import get_topic
from research_agent.ollama_client import OllamaClient
from research_agent.rag import LocalRAG


def explore_question(
    config: dict,
    question: str,
    rounds: int = 2,
    n_evidence: int = 8,
    use_deepseek: bool = True,
    topic_id: int | None = None,
    initial_sources: list[dict] | None = None,
    continued_from: str | None = None,
    extra_queries: list[str] | None = None,
    assess_with_deepseek: bool = False,
) -> dict:
    if not question.strip():
        raise ValueError("question must not be empty")

    topic = _load_topic(config, topic_id)
    rag = LocalRAG(config)
    client = _create_client(config, use_deepseek)
    model = config["deepseek_model"] if use_deepseek else config["chat_model"]

    initial_evidence = rag.retrieve(question, n_results=n_evidence)
    exploration = _parse_exploration_json(
        client.chat(_build_exploration_prompt(question, topic, initial_evidence)),
        question,
    )
    all_evidence = list(initial_sources or [])
    all_evidence.extend(initial_evidence)
    queries = list(extra_queries or exploration["search_queries"])

    for _round in range(max(rounds - 1, 0)):
        next_evidence = []
        for query in queries:
            next_evidence.extend(rag.retrieve(query, n_results=n_evidence))
        all_evidence.extend(next_evidence)
        if rounds > 2 and next_evidence:
            queries = exploration["search_queries"]

    sources = _dedupe_evidence(all_evidence)
    answer = client.chat(_build_answer_prompt(question, topic, sources))
    evidence_assessment = assess_evidence(
        question=question,
        answer=answer,
        sources=sources,
        config=config,
        use_deepseek=assess_with_deepseek,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    created_at = datetime.now().isoformat(timespec="seconds")
    report_path = Path(config["reports_dir"]) / f"explore_{timestamp}.md"
    json_path = Path(config["metadata_dir"]) / "explorations" / f"explore_{timestamp}.json"

    _write_markdown_report(
        report_path,
        question=question,
        topic_id=topic_id,
        continued_from=continued_from,
        rounds=rounds,
        n_evidence=n_evidence,
        model=model,
        subquestions=exploration["subquestions"],
        search_queries=exploration["search_queries"],
        possible_gaps=exploration["possible_gaps"],
        answer=answer,
        sources=sources,
        evidence_assessment=evidence_assessment,
    )
    record = {
        "question": question,
        "topic_id": topic_id,
        "rounds": rounds,
        "n_evidence": n_evidence,
        "model": model,
        "subquestions": exploration["subquestions"],
        "search_queries": exploration["search_queries"],
        "possible_gaps": exploration["possible_gaps"],
        "answer": answer,
        "sources": sources,
        "evidence_assessment": evidence_assessment,
        "report_path": str(report_path),
        "json_path": str(json_path),
        "created_at": created_at,
    }
    if continued_from:
        record["continued_from"] = continued_from
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "answer": answer,
        "subquestions": exploration["subquestions"],
        "search_queries": exploration["search_queries"],
        "possible_gaps": exploration["possible_gaps"],
        "sources": sources,
        "evidence_assessment": evidence_assessment,
        "report_path": str(report_path),
        "json_path": str(json_path),
    }


def list_explorations(config: dict) -> list[dict]:
    records = []
    for path in sorted(_explorations_dir(config).glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        records.append(
            {
                "id": path.stem,
                "question": data.get("question", ""),
                "topic_id": data.get("topic_id"),
                "created_at": data.get("created_at", ""),
                "report_path": data.get("report_path", ""),
                "sources_count": len(data.get("sources") or []),
            }
        )
    return sorted(records, key=lambda item: item.get("created_at") or item["id"], reverse=True)


def load_exploration(config: dict, explore_id: str) -> dict:
    path = _explorations_dir(config) / f"{explore_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"exploration not found: {explore_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["json_path"] = str(path)
    return data


def continue_exploration(
    config: dict,
    explore_id: str,
    rounds: int = 1,
    n_evidence: int = 8,
    use_deepseek: bool = True,
    assess_with_deepseek: bool = False,
) -> dict:
    previous = load_exploration(config, explore_id)
    question = previous["question"]
    possible_gaps = previous.get("possible_gaps") or []
    search_queries = previous.get("search_queries") or []
    continued_question = (
        f"继续探索以下问题：{question}。"
        f"重点补充这些证据缺口：{'; '.join(possible_gaps)}。"
        f"继续检索这些方向：{'; '.join(search_queries)}。"
    )
    return explore_question(
        config,
        continued_question,
        rounds=rounds,
        n_evidence=n_evidence,
        use_deepseek=use_deepseek,
        topic_id=previous.get("topic_id"),
        initial_sources=previous.get("sources") or [],
        continued_from=explore_id,
        extra_queries=search_queries or [question],
        assess_with_deepseek=assess_with_deepseek,
    )


def _load_topic(config: dict, topic_id: int | None) -> dict | None:
    if topic_id is None:
        return None
    topic = get_topic(Path(config["metadata_dir"]) / "papers.sqlite", topic_id)
    if topic is None:
        raise ValueError(f"topic not found: {topic_id}")
    return topic


def _explorations_dir(config: dict) -> Path:
    return Path(config["metadata_dir"]) / "explorations"


def _create_client(config: dict, use_deepseek: bool):
    if use_deepseek:
        load_project_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError("DeepSeek 不可用，可使用 --local，或运行 research-agent check-deepseek --ping 检查。")
        return DeepSeekClient(
            api_key=api_key,
            base_url=config["deepseek_base_url"],
            model=config["deepseek_model"],
        )
    return OllamaClient(
        base_url=config["ollama_base_url"],
        chat_model=config["chat_model"],
        embedding_model=config["embedding_model"],
    )


def _build_exploration_prompt(question: str, topic: dict | None, evidence: list[dict]) -> str:
    return (
        "请把科研问题拆解为 JSON。只输出 JSON，不要输出解释文字。\n"
        '格式：{"subquestions":["..."],"search_queries":["..."],"possible_gaps":["..."]}\n\n'
        f"原始问题：{question}\n\n"
        f"topic 信息：{_format_topic(topic)}\n\n"
        f"初始 evidence 摘要：\n{_format_evidence(evidence)}\n"
    )


def _build_answer_prompt(question: str, topic: dict | None, sources: list[dict]) -> str:
    return (
        "请只基于给定证据回答，不要编造未提供的文献信息。"
        "每个关键结论标注 [证据 1]、[证据 2]；"
        "明确区分“文献直接支持”和“基于证据的推断”；如果证据不足，明确说明。\n\n"
        f"问题：{question}\n\n"
        f"topic 信息：{_format_topic(topic)}\n\n"
        f"证据：\n{_format_evidence(sources)}\n\n"
        "请按以下结构输出：\n"
        "1. 直接回答\n"
        "2. 分项分析\n"
        "3. 证据支持\n"
        "4. 证据不足或不确定点\n"
        "5. 下一步建议检索关键词\n"
        "6. 建议精读文献\n"
    )


def _parse_exploration_json(text: str, question: str) -> dict:
    try:
        data = json.loads(text)
        return {
            "subquestions": _string_list(data.get("subquestions")) or [question],
            "search_queries": _string_list(data.get("search_queries")) or [question],
            "possible_gaps": _string_list(data.get("possible_gaps")),
        }
    except (json.JSONDecodeError, TypeError):
        return {
            "subquestions": [question],
            "search_queries": [question],
            "possible_gaps": [],
        }


def _string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _dedupe_evidence(evidence: list[dict]) -> list[dict]:
    seen = set()
    results = []
    for item in evidence:
        metadata = item.get("metadata") or {}
        source = metadata.get("source") or item.get("source") or ""
        chunk_id = metadata.get("chunk_id") or item.get("chunk_id") or item.get("id") or ""
        key = (source, chunk_id)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {
                "evidence_id": len(results) + 1,
                "title": metadata.get("title") or item.get("title") or "",
                "authors": metadata.get("authors") or item.get("authors") or "",
                "year": metadata.get("year") or item.get("year") or "",
                "source": source,
                "chunk_id": chunk_id,
                "distance": item.get("distance"),
                "text": item.get("text") or "",
            }
        )
    return results


def _format_topic(topic: dict | None) -> str:
    if not topic:
        return ""
    return "\n".join(
        [
            f"name: {topic.get('name') or ''}",
            f"description: {topic.get('description') or ''}",
            f"keywords_cn: {topic.get('keywords_cn') or ''}",
            f"keywords_en: {topic.get('keywords_en') or ''}",
        ]
    )


def _format_evidence(evidence: list[dict]) -> str:
    if not evidence:
        return "无"
    blocks = []
    for index, item in enumerate(evidence, start=1):
        metadata = item.get("metadata") or {}
        source = metadata.get("source") or item.get("source") or ""
        chunk_id = metadata.get("chunk_id") or item.get("chunk_id") or ""
        text = item.get("text") or ""
        blocks.append(f"[证据 {index} | source={source} | chunk_id={chunk_id}]\n{text[:1000]}")
    return "\n\n".join(blocks)


def _write_markdown_report(
    path: Path,
    question: str,
    topic_id: int | None,
    continued_from: str | None,
    rounds: int,
    n_evidence: int,
    model: str,
    subquestions: list[str],
    search_queries: list[str],
    possible_gaps: list[str],
    answer: str,
    sources: list[dict],
    evidence_assessment: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 问题探索报告",
        "",
        "## 1. 用户问题",
        "",
        question,
        "",
        "## 2. 探索设置",
        "",
        f"- rounds: {rounds}",
        f"- n_evidence: {n_evidence}",
        f"- model: {model}",
        f"- topic_id: {topic_id if topic_id is not None else ''}",
        f"- continued_from: {continued_from or ''}",
        "",
        "## 3. 拆解出的子问题",
        "",
        *[f"- {item}" for item in subquestions],
        "",
        "## 4. 扩展检索词",
        "",
        *[f"- {item}" for item in search_queries],
        "",
        "## 5. 最终回答",
        "",
        answer,
        "",
        "## 6. 证据来源表",
        "",
        "| 编号 | title | authors | year | source | chunk_id | distance |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for source in sources:
        lines.append(
            "| "
            f"{source.get('evidence_id')} | {_cell(source.get('title'))} | "
            f"{_cell(source.get('authors'))} | {_cell(source.get('year'))} | "
            f"{_cell(source.get('source'))} | {_cell(source.get('chunk_id'))} | "
            f"{_cell(source.get('distance'))} |"
        )
    lines.extend(
        [
            "",
            "## 7. 证据不足与后续检索建议",
            "",
            *[f"- {item}" for item in possible_gaps],
            "",
            "## 8. 证据质量评估",
            "",
            f"- overall_strength: {evidence_assessment.get('overall_strength') or 'unknown'}",
            "",
            "### supported_claims",
            "",
            *[
                f"- {item.get('claim', '')} | evidence_ids={item.get('evidence_ids', [])} | strength={item.get('strength', '')}"
                for item in evidence_assessment.get("supported_claims", [])
                if isinstance(item, dict)
            ],
            "",
            "### weak_claims",
            "",
            *[
                f"- {item.get('claim', '')} | reason={item.get('reason', '')} | needed_evidence={item.get('needed_evidence', '')}"
                for item in evidence_assessment.get("weak_claims", [])
                if isinstance(item, dict)
            ],
            "",
            "### possibly_irrelevant_sources",
            "",
            *[
                f"- {item.get('source', '')} | reason={item.get('reason', '')}"
                for item in evidence_assessment.get("possibly_irrelevant_sources", [])
                if isinstance(item, dict)
            ],
            "",
            "## 9. 建议精读文献",
            "",
            *[
                f"- {item.get('title', '')} | source={item.get('source', '')} | priority={item.get('priority', '')} | reason={item.get('reason', '')}"
                for item in evidence_assessment.get("recommended_reading", [])
                if isinstance(item, dict)
            ],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _cell(value) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")
