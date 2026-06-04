import json
import os

from research_agent.config import load_project_dotenv
from research_agent.deepseek_client import DeepSeekClient
from research_agent.ollama_client import OllamaClient


FALLBACK_ASSESSMENT = {
    "overall_strength": "unknown",
    "supported_claims": [],
    "weak_claims": [],
    "recommended_reading": [],
    "possibly_irrelevant_sources": [],
}


def assess_evidence(
    question: str,
    answer: str,
    sources: list[dict],
    config: dict,
    use_deepseek: bool = False,
) -> dict:
    client = _create_client(config, use_deepseek)
    response = client.chat(_build_prompt(question, answer, sources))
    try:
        data = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return dict(FALLBACK_ASSESSMENT)
    return {
        "overall_strength": data.get("overall_strength") or "unknown",
        "supported_claims": _list_or_empty(data.get("supported_claims")),
        "weak_claims": _list_or_empty(data.get("weak_claims")),
        "recommended_reading": _list_or_empty(data.get("recommended_reading")),
        "possibly_irrelevant_sources": _list_or_empty(
            data.get("possibly_irrelevant_sources")
        ),
    }


def _create_client(config: dict, use_deepseek: bool):
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


def _build_prompt(question: str, answer: str, sources: list[dict]) -> str:
    return (
        "请评估回答所依据的证据质量。只输出 JSON，不要输出解释文字。\n"
        "JSON 格式：\n"
        "{\n"
        '  "overall_strength": "strong|medium|weak|unknown",\n'
        '  "supported_claims": [{"claim": "...", "evidence_ids": [1, 3], "strength": "strong|medium|weak"}],\n'
        '  "weak_claims": [{"claim": "...", "reason": "...", "needed_evidence": "..."}],\n'
        '  "recommended_reading": [{"title": "...", "source": "...", "reason": "...", "priority": "high|medium|low"}],\n'
        '  "possibly_irrelevant_sources": [{"source": "...", "reason": "..."}]\n'
        "}\n\n"
        f"问题：{question}\n\n"
        f"回答：{answer}\n\n"
        f"证据：\n{_format_sources(sources)}\n"
    )


def _format_sources(sources: list[dict]) -> str:
    blocks = []
    for index, source in enumerate(sources, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[证据 {index}]",
                    f"title: {source.get('title') or ''}",
                    f"authors: {source.get('authors') or ''}",
                    f"year: {source.get('year') or ''}",
                    f"source: {source.get('source') or ''}",
                    f"chunk_id: {source.get('chunk_id') or ''}",
                    f"text: {(source.get('text') or '')[:1000]}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _list_or_empty(value) -> list:
    return value if isinstance(value, list) else []
