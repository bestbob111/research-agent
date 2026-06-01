import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from research_agent.config import DIRECTORY_KEYS, load_config, load_project_dotenv
from research_agent.history import append_jsonl
from research_agent.indexer import build_index
from research_agent.ingest_metadata import record_ingested_metadata
from research_agent.metadata_db import init_db, list_papers
from research_agent.ollama_client import OllamaClient
from research_agent.pdf_parser import extract_pdfs_to_texts_with_report
from research_agent.rag import LocalRAG
from research_agent.report_generator import generate_review_report
from research_agent.reviewer import EvidenceReviewer
from research_agent.vector_store import VectorStore


def main(argv: list[str] | None = None) -> None:
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        args.func(args)
    except BrokenPipeError:
        if argv is None:
            _redirect_stdout_to_devnull()
        return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status")
    status_parser.set_defaults(func=cmd_status)

    check_paths_parser = subparsers.add_parser("check-paths")
    check_paths_parser.set_defaults(func=cmd_check_paths)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.set_defaults(func=cmd_ingest)

    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("--reset", action="store_true")
    index_parser.add_argument("--limit", type=int, default=None)
    index_parser.add_argument("--max-chars-per-embed", type=int, default=1800)
    index_parser.set_defaults(func=cmd_index)

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("question")
    query_parser.add_argument("--n", type=int, default=5)
    query_parser.set_defaults(func=cmd_query)

    ask_parser = subparsers.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--n", type=int, default=6)
    ask_parser.set_defaults(func=cmd_ask)

    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("question")
    review_parser.add_argument("--n", type=int, default=12)
    review_parser.add_argument("--output", default=None)
    review_parser.set_defaults(func=cmd_review)

    papers_parser = subparsers.add_parser("papers")
    papers_subparsers = papers_parser.add_subparsers(dest="papers_command", required=True)
    papers_list_parser = papers_subparsers.add_parser("list")
    papers_list_parser.add_argument("--limit", type=_positive_int, default=None)
    papers_list_parser.add_argument("--offset", type=_non_negative_int, default=0)
    papers_list_parser.set_defaults(func=cmd_papers_list)

    return parser


def cmd_status(_args: argparse.Namespace) -> None:
    config = load_config()
    load_project_dotenv()
    keys = (
        "data_dir",
        "papers_dir",
        "texts_dir",
        "chroma_dir",
        "metadata_dir",
        "reports_dir",
    )
    for key in keys:
        print(f"{key}: {config[key]}")

    papers_dir = Path(config["papers_dir"])
    texts_dir = Path(config["texts_dir"])
    metadata_db = Path(config["metadata_dir"]) / "papers.sqlite"
    print(f"pdf_count: {_count_files(papers_dir, '*.pdf')}")
    print(f"txt_count: {_count_files(texts_dir, '*.txt')}")
    print(f"metadata_db_exists: {'yes' if metadata_db.exists() else 'no'}")
    print(f"deepseek_api_key_configured: {'yes' if os.getenv('DEEPSEEK_API_KEY') else 'no'}")

    try:
        vector_store = VectorStore(config["chroma_dir"], config["collection_name"])
        print(f"chroma_collection_count: {vector_store.count()}")
    except Exception as exc:
        print(f"chroma_collection_count: unavailable ({exc})")


def cmd_check_paths(_args: argparse.Namespace) -> None:
    config = load_config()
    for key in DIRECTORY_KEYS:
        path = Path(config[key])
        status = "OK" if path.exists() else "MISSING"
        print(f"{key}: {path} [{status}]")

    data_dir = Path(config["data_dir"])
    test_file = data_dir / ".write_test"
    try:
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        print(f"writable: {data_dir} [OK]")
    except OSError as exc:
        print(f"writable: {data_dir} [FAILED: {exc}]")


def cmd_ingest(_args: argparse.Namespace) -> None:
    config = load_config()
    papers_dir = Path(config["papers_dir"])
    texts_dir = Path(config["texts_dir"])

    print(f"papers_dir: {papers_dir}")
    print(f"texts_dir: {texts_dir}")
    pdf_count = _count_files(papers_dir, "*.pdf")
    print(f"found PDFs: {pdf_count}")

    report = extract_pdfs_to_texts_with_report(
        papers_dir,
        texts_dir,
        progress_callback=lambda index, total, path: print(
            f"[{index}/{total}] processing: {path.name}"
        ),
    )

    print(f"generated txt: {len(report['success'])}")
    for path in report["success"]:
        print(path)
    metadata_count = record_ingested_metadata(config, report["success"])
    print(f"metadata records updated: {metadata_count}")
    print(f"failed PDFs: {len(report['failures'])}")
    for failure in report["failures"]:
        print(f"{failure['pdf']}: {failure['error']}")
    print(f"output txt dir: {texts_dir}")


def cmd_index(args: argparse.Namespace) -> None:
    config = load_config()
    texts_dir = Path(config["texts_dir"])
    text_files = sorted(texts_dir.glob("*.txt")) if texts_dir.exists() else []
    if args.limit is not None:
        text_files = text_files[: args.limit]

    print(f"texts_dir: {texts_dir}")
    print(f"chroma_dir: {config['chroma_dir']}")
    print(f"collection_name: {config['collection_name']}")
    if not text_files:
        print("No txt files found. Run research-agent ingest first.")

    result = build_index(
        config,
        reset=args.reset,
        limit=args.limit,
        max_chars_per_embed=args.max_chars_per_embed,
    )
    _print_index_result(config, result)


def cmd_query(args: argparse.Namespace) -> None:
    config = load_config()
    client = OllamaClient(
        base_url=config["ollama_base_url"],
        chat_model=config["chat_model"],
        embedding_model=config["embedding_model"],
    )
    vector_store = VectorStore(config["chroma_dir"], config["collection_name"])
    results = vector_store.query(client.embed(args.question), n_results=args.n)
    for index, result in enumerate(results, start=1):
        metadata = result["metadata"] or {}
        text = result["text"] or ""
        print(f"[{index}] source: {metadata.get('source', '')}")
        print(f"chunk_id: {metadata.get('chunk_id', '')}")
        print(f"distance: {result['distance']}")
        print(text[:500])
        print()


def cmd_ask(args: argparse.Namespace) -> None:
    question = args.question.strip()
    if not question:
        print("问题不能为空。")
        return

    config = load_config()
    rag = LocalRAG(config)
    result = rag.answer(question, n_results=args.n)

    print(f"问题: {question}")
    print("回答:")
    print(result["answer"])
    print("证据来源:")
    for source in result["sources"]:
        title = source.get("title") or source.get("source", "")
        print(
            f"- title={title}, "
            f"authors={source.get('authors') or ''}, "
            f"year={source.get('year') or ''}, "
            f"source={source.get('source', '')}, "
            f"chunk_id={source.get('chunk_id', '')}, "
            f"distance={source.get('distance')}"
        )

    append_jsonl(
        Path(config["metadata_dir"]) / "qa_history.jsonl",
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "model": config["chat_model"],
        },
    )


def cmd_review(args: argparse.Namespace) -> None:
    question = args.question.strip()
    if not question:
        print("问题不能为空。")
        return

    config = load_config()
    try:
        reviewer = EvidenceReviewer(config)
    except ValueError as exc:
        print(f"DeepSeek 配置错误: {exc}")
        print("请在 .env 或环境变量中配置 DEEPSEEK_API_KEY。")
        return

    result = reviewer.review_with_deepseek(question, n_results=args.n)
    print(f"问题: {question}")
    print("DeepSeek 综述:")
    print(result["review"])
    print("证据来源:")
    for source in result["sources"]:
        print(
            f"- source={source.get('source', '')}, "
            f"chunk_id={source.get('chunk_id', '')}, "
            f"distance={source.get('distance')}"
        )

    reports_dir = Path(config["reports_dir"])
    report_path = _resolve_report_path(reports_dir, args.output)
    generate_review_report(
        question=question,
        review=result["review"],
        sources=result["sources"],
        output_path=report_path,
        metadata=result.get("metadata"),
    )
    print(f"报告已保存: {report_path}")


def cmd_papers_list(args: argparse.Namespace) -> None:
    config = load_config()
    db_path = Path(config["metadata_dir"]) / "papers.sqlite"
    init_db(db_path)
    papers = list_papers(db_path)
    visible_papers = _slice_papers(papers, args.offset, args.limit)
    print(f"metadata_db: {db_path}")
    print(f"papers: {len(papers)}")
    for paper in visible_papers:
        print(
            f"{paper['id']}\t{paper.get('title') or ''}\t"
            f"{paper.get('file_path') or ''}\t{paper.get('text_path') or ''}"
        )


def _count_files(path: Path, pattern: str) -> int:
    return len(sorted(path.glob(pattern))) if path.exists() else 0


def _redirect_stdout_to_devnull() -> None:
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull_fd, sys.stdout.fileno())
        finally:
            os.close(devnull_fd)
    except (OSError, ValueError):
        pass


def _positive_int(value: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return integer


def _non_negative_int(value: str) -> int:
    integer = int(value)
    if integer < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return integer


def _slice_papers(papers: list[dict], offset: int = 0, limit: int | None = None) -> list[dict]:
    if limit is None:
        return papers[offset:]
    return papers[offset : offset + limit]


def _print_index_result(config: dict, result: dict) -> None:
    print(f"processed txt: {result['processed_files']}")
    print(f"written chunks: {result['written_chunks']}")
    print(f"collection count: {result['collection_count']}")
    print(f"skipped empty files: {result['skipped_empty_files']}")
    print(f"failed files: {len(result['failed_files'])}")
    print(f"failed chunks: {len(result['failed_chunks'])}")

    if result["failed_files"] or result["failed_chunks"]:
        metadata_dir = Path(config["metadata_dir"])
        metadata_dir.mkdir(parents=True, exist_ok=True)
        report_path = metadata_dir / "index_failures.json"
        report_path.write_text(
            json.dumps(
                {
                    "failed_files": result["failed_files"],
                    "failed_chunks": result["failed_chunks"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"failure report: {report_path}")


def _resolve_report_path(reports_dir: Path, output: str | None) -> Path:
    if output:
        path = Path(output)
        if path.suffix != ".md":
            path = path.with_suffix(".md")
        return path if path.is_absolute() else reports_dir / path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return reports_dir / f"review_{timestamp}.md"
