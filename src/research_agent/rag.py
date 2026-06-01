from pathlib import Path

from research_agent.metadata_db import get_paper_by_text_path
from research_agent.ollama_client import OllamaClient
from research_agent.vector_store import VectorStore


NO_EVIDENCE_ANSWER = "未在当前文献库中检索到足够相关的证据。"


class LocalRAG:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.client = OllamaClient(
            base_url=config["ollama_base_url"],
            chat_model=config["chat_model"],
            embedding_model=config["embedding_model"],
        )
        self.vector_store = VectorStore(
            chroma_dir=config["chroma_dir"],
            collection_name=config["collection_name"],
        )

    def retrieve(self, question: str, n_results: int = 6) -> list[dict]:
        if not question.strip():
            raise ValueError("question must not be empty")
        query_embedding = self.client.embed(question)
        return self.vector_store.query(query_embedding, n_results=n_results)

    def answer(self, question: str, n_results: int = 6) -> dict:
        results = self.retrieve(question, n_results=n_results)
        if not results:
            return {"answer": NO_EVIDENCE_ANSWER, "sources": []}

        evidence_blocks = []
        sources = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata") or {}
            source = metadata.get("source", "")
            chunk_id = metadata.get("chunk_id", "")
            text_path = metadata.get("text_path", "")
            text = result.get("text") or ""
            evidence_blocks.append(
                f"[证据 {index} | source={source} | chunk_id={chunk_id}]\n{text}"
            )
            source_info = {
                "evidence_id": index,
                "source": source,
                "chunk_id": chunk_id,
                "distance": result.get("distance"),
                "id": result.get("id"),
                "text_path": text_path,
            }
            paper = self._get_paper_metadata(text_path)
            if paper:
                source_info.update(
                    {
                        "title": paper.get("title"),
                        "authors": paper.get("authors"),
                        "year": paper.get("year"),
                    }
                )
            sources.append(source_info)

        prompt = _build_prompt(question, evidence_blocks)
        answer = self.client.chat(prompt)
        return {"answer": answer, "sources": sources}

    def _get_paper_metadata(self, text_path: str) -> dict | None:
        if not text_path or "metadata_dir" not in self.config:
            return None
        db_path = Path(self.config["metadata_dir"]) / "papers.sqlite"
        return get_paper_by_text_path(db_path, text_path)


def _build_prompt(question: str, evidence_blocks: list[str]) -> str:
    evidence_text = "\n\n".join(evidence_blocks)
    return (
        "你是一个本地科研文献助手。请严格遵守：\n"
        "1. 只根据给定证据回答问题。\n"
        "2. 不要编造文献中没有的信息。\n"
        "3. 如果证据不足，明确说“根据当前文献片段无法确定”。\n"
        "4. 关键结论后标注 [证据 1]、[证据 2] 等证据编号。\n"
        "5. 最后列出“不确定点/需要继续检索的内容”。\n\n"
        f"问题：{question}\n\n"
        f"证据：\n{evidence_text}\n\n"
        "请给出回答："
    )
