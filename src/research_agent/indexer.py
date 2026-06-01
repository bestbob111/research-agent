import math
from pathlib import Path
from typing import Any

from research_agent.chunking import chunk_text
from research_agent.ollama_client import OllamaClient
from research_agent.text_cleaner import prepare_text_for_embedding
from research_agent.vector_store import VectorStore


def build_index(
    config: dict,
    reset: bool = False,
    limit: int | None = None,
    max_chars_per_embed: int = 1800,
) -> dict:
    texts_dir = Path(config["texts_dir"])
    vector_store = VectorStore(
        chroma_dir=config["chroma_dir"],
        collection_name=config["collection_name"],
    )
    if reset:
        vector_store.reset_collection(confirm=True)

    text_files = sorted(texts_dir.glob("*.txt")) if texts_dir.exists() else []
    if limit is not None:
        text_files = text_files[:limit]

    client = OllamaClient(
        base_url=config["ollama_base_url"],
        chat_model=config["chat_model"],
        embedding_model=config["embedding_model"],
    )

    processed_files = 0
    written_chunks = 0
    skipped_empty_files = 0
    failed_files = []
    failed_chunks = []

    for text_path in text_files:
        print(f"indexing txt: {text_path.name}")
        try:
            text = text_path.read_text(encoding="utf-8")
            if not text.strip():
                print(f"skip empty txt: {text_path.name}")
                skipped_empty_files += 1
                continue

            chunks = chunk_text(
                text,
                chunk_size=int(config["chunk_size"]),
                overlap=int(config["chunk_overlap"]),
            )
            print(f"chunks for {text_path.name}: {len(chunks)}")
            if not chunks:
                print(f"skip txt with 0 chunks: {text_path.name}")
                skipped_empty_files += 1
                continue

            valid_chunks = []
            embeddings = []
            metadatas = []

            for chunk_index, chunk in enumerate(chunks):
                prepared_chunk = prepare_text_for_embedding(
                    chunk,
                    max_chars=max_chars_per_embed,
                )
                if not prepared_chunk:
                    print(f"skip empty chunk: {text_path.name} #{chunk_index}")
                    continue
                try:
                    embedding = client.embed(prepared_chunk)
                except Exception as exc:
                    error = str(exc)[:300]
                    failed_chunks.append(
                        {
                            "source": text_path.name,
                            "chunk_id": chunk_index,
                            "error": error,
                        }
                    )
                    print(f"failed chunk: {text_path.name} #{chunk_index}: {error}")
                    continue
                if not _is_valid_embedding(embedding):
                    error = "invalid embedding: empty or contains non-finite values"
                    failed_chunks.append(
                        {
                            "source": text_path.name,
                            "chunk_id": chunk_index,
                            "error": error,
                        }
                    )
                    print(f"failed chunk: {text_path.name} #{chunk_index}: {error}")
                    continue

                valid_chunks.append(prepared_chunk)
                embeddings.append(embedding)
                metadatas.append(
                    {
                        "source": text_path.name,
                        "text_path": str(text_path),
                        "chunk_id": chunk_index,
                    }
                )

            if not valid_chunks:
                failed_files.append(
                    {
                        "source": text_path.name,
                        "error": "no chunks were embedded successfully",
                    }
                )
                print(f"no chunks embedded successfully: {text_path.name}")
                continue

            written = vector_store.add_chunks(
                doc_id=text_path.stem,
                chunks=valid_chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            written_chunks += written
            processed_files += 1
            print(f"cumulative written chunks: {written_chunks}")
        except Exception as exc:
            error = str(exc)
            failed_files.append({"source": text_path.name, "error": error})
            print(f"Failed to index {text_path}: {error}")

    return {
        "processed_files": processed_files,
        "written_chunks": written_chunks,
        "collection_count": vector_store.count(),
        "skipped_empty_files": skipped_empty_files,
        "failed_files": failed_files,
        "failed_chunks": failed_chunks,
    }


def _is_valid_embedding(embedding: Any) -> bool:
    if not isinstance(embedding, list) or not embedding:
        return False
    return all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        for value in embedding
    )
