from pathlib import Path
from typing import Any

import chromadb


class VectorStore:
    def __init__(self, chroma_dir: str, collection_name: str) -> None:
        self.chroma_dir = str(chroma_dir)
        self.collection_name = collection_name
        Path(self.chroma_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.chroma_dir)
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def add_chunks(
        self,
        doc_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> int:
        if not doc_id:
            raise ValueError("doc_id must not be empty")
        if not (len(chunks) == len(embeddings) == len(metadatas)):
            raise ValueError("chunks, embeddings, and metadatas must have same length")
        if not chunks:
            return 0

        ids = []
        stored_metadatas = []
        for chunk_index, metadata in enumerate(metadatas):
            chunk_id = f"{doc_id}::chunk_{chunk_index}"
            ids.append(chunk_id)
            stored_metadata = dict(metadata)
            stored_metadata["doc_id"] = doc_id
            stored_metadata["chunk_id"] = chunk_id
            stored_metadatas.append(stored_metadata)

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=stored_metadatas,
        )
        return len(chunks)

    def query(self, query_embedding: list[float], n_results: int = 6) -> list[dict]:
        if not query_embedding:
            raise ValueError("query_embedding must not be empty")
        if n_results <= 0:
            raise ValueError("n_results must be greater than 0")

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        ids = _first_result_list(result.get("ids"))
        documents = _first_result_list(result.get("documents"))
        metadatas = _first_result_list(result.get("metadatas"))
        distances = _first_result_list(result.get("distances"))

        rows = []
        for index, item_id in enumerate(ids):
            rows.append(
                {
                    "id": item_id,
                    "text": documents[index] if index < len(documents) else "",
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                    "distance": distances[index] if index < len(distances) else None,
                }
            )
        return rows

    def count(self) -> int:
        return self.collection.count()

    def delete_doc(self, doc_id: str) -> None:
        if not doc_id:
            raise ValueError("doc_id must not be empty")
        self.collection.delete(where={"doc_id": doc_id})

    def reset_collection(self, confirm: bool = False) -> None:
        if confirm is not True:
            raise ValueError("reset_collection requires confirm=True")
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(self.collection_name)


def _first_result_list(value: Any) -> list:
    if not value or not isinstance(value, list):
        return []
    first = value[0]
    return first if isinstance(first, list) else []
