import os

from research_agent.config import load_project_dotenv
from research_agent.deepseek_client import DeepSeekClient
from research_agent.rag import LocalRAG, NO_EVIDENCE_ANSWER


class EvidenceReviewer:
    def __init__(self, config: dict) -> None:
        load_project_dotenv()
        self.config = config
        self.rag = LocalRAG(config)
        self.deepseek = DeepSeekClient(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=config["deepseek_base_url"],
            model=config["deepseek_model"],
        )

    def retrieve_evidence(self, question: str, n_results: int = 12) -> list[dict]:
        if not question.strip():
            raise ValueError("question must not be empty")
        return self.rag.retrieve(question, n_results=n_results)

    def review_with_deepseek(self, question: str, n_results: int = 12) -> dict:
        evidence = self.retrieve_evidence(question, n_results=n_results)
        if not evidence:
            return {
                "review": NO_EVIDENCE_ANSWER,
                "sources": [],
                "metadata": {
                    "model": self.config["deepseek_model"],
                    "n_results": n_results,
                    "evidence_count": 0,
                    "unique_sources_count": 0,
                },
            }

        evidence_blocks = []
        sources = []
        for index, result in enumerate(evidence, start=1):
            metadata = result.get("metadata") or {}
            source = metadata.get("source", "")
            chunk_id = metadata.get("chunk_id", "")
            text = result.get("text") or ""
            evidence_blocks.append(
                f"[证据 {index} | source={source} | chunk_id={chunk_id}]\n{text}"
            )
            sources.append(
                {
                    "evidence_id": index,
                    "source": source,
                    "chunk_id": chunk_id,
                    "distance": result.get("distance"),
                    "id": result.get("id"),
                }
            )

        prompt = _build_review_prompt(question, evidence_blocks)
        review = self.deepseek.chat(prompt)
        return {
            "review": review,
            "sources": sources,
            "metadata": {
                "model": self.config["deepseek_model"],
                "n_results": n_results,
                "evidence_count": len(sources),
                "unique_sources_count": _count_unique_sources(sources),
            },
        }


def _build_review_prompt(question: str, evidence_blocks: list[str]) -> str:
    evidence_text = "\n\n".join(evidence_blocks)
    return (
        "你是科研综述助手。请严格遵守：\n"
        "1. 只基于给定证据回答。\n"
        "2. 区分“文献直接支持的结论”和“基于证据的推断”。\n"
        "3. 不要编造未提供的文献、作者、年份或实验结果。\n"
        "4. 关键论点后标注 [证据 1]、[证据 2] 等证据编号。\n"
        "5. 按以下结构输出：\n"
        "   1. 核心结论\n"
        "   2. 技术路线\n"
        "   3. 代表性证据\n"
        "   4. 不确定点\n"
        "   5. 下一步检索建议\n\n"
        f"综述问题：{question}\n\n"
        f"证据片段：\n{evidence_text}\n\n"
        "请生成结构化科研综述："
    )


def _count_unique_sources(sources: list[dict]) -> int:
    values = {
        source.get("source") or source.get("id")
        for source in sources
        if source.get("source") or source.get("id")
    }
    return len(values)
