import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from research_agent.arxiv_searcher import (
    download_arxiv_pdfs,
    import_arxiv_results_to_db,
    save_arxiv_results,
    search_arxiv,
)
from research_agent.config import DIRECTORY_KEYS, load_config, load_project_dotenv
from research_agent.download_importer import import_downloaded_pdfs
from research_agent.explorer import (
    continue_exploration,
    explore_question,
    list_explorations,
    load_exploration,
)
from research_agent.history import append_jsonl
from research_agent.indexer import build_index
from research_agent.ingest_metadata import record_ingested_metadata
from research_agent.metadata_db import init_db, list_papers
from research_agent.ollama_client import OllamaClient
from research_agent.pdf_parser import extract_pdfs_to_texts_with_report
from research_agent.rag import LocalRAG
from research_agent.report_generator import generate_review_report
from research_agent.reviewer import EvidenceReviewer
from research_agent.topic_manager import (
    create_research_topic,
    generate_topic_plan,
    generate_topic_report,
    list_research_topics,
    show_research_topic,
)
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

    arxiv_parser = subparsers.add_parser("arxiv")
    arxiv_subparsers = arxiv_parser.add_subparsers(dest="arxiv_command", required=True)
    arxiv_search_parser = arxiv_subparsers.add_parser("search")
    arxiv_search_parser.add_argument("query")
    arxiv_search_parser.add_argument("--max-results", type=_positive_int, default=20)
    arxiv_search_parser.add_argument("--download", action="store_true")
    arxiv_search_parser.add_argument("--download-limit", type=_positive_int, default=None)
    arxiv_search_parser.add_argument("--output", default=None)
    arxiv_search_parser.set_defaults(func=cmd_arxiv_search)

    import_downloads_parser = subparsers.add_parser("import-downloads")
    import_downloads_parser.add_argument(
        "--source",
        choices=["all", "arxiv", "import"],
        default="all",
    )
    import_mode_group = import_downloads_parser.add_mutually_exclusive_group()
    import_mode_group.add_argument("--copy", action="store_true", default=True)
    import_mode_group.add_argument("--move", action="store_true")
    import_downloads_parser.set_defaults(func=cmd_import_downloads)

    explore_parser = subparsers.add_parser("explore")
    explore_parser.add_argument("explore_args", nargs=argparse.REMAINDER)
    explore_parser.set_defaults(func=cmd_explore)

    papers_parser = subparsers.add_parser("papers")
    papers_subparsers = papers_parser.add_subparsers(dest="papers_command", required=True)
    papers_list_parser = papers_subparsers.add_parser("list")
    papers_list_parser.add_argument("--limit", type=_positive_int, default=None)
    papers_list_parser.add_argument("--offset", type=_non_negative_int, default=0)
    papers_list_parser.set_defaults(func=cmd_papers_list)

    topic_parser = subparsers.add_parser("topic")
    topic_subparsers = topic_parser.add_subparsers(dest="topic_command", required=True)
    topic_create_parser = topic_subparsers.add_parser("create")
    topic_create_parser.add_argument("name")
    topic_create_parser.add_argument("--description", default="")
    topic_create_parser.set_defaults(func=cmd_topic_create)

    topic_list_parser = topic_subparsers.add_parser("list")
    topic_list_parser.set_defaults(func=cmd_topic_list)

    topic_show_parser = topic_subparsers.add_parser("show")
    topic_show_parser.add_argument("topic_id", type=int)
    topic_show_parser.set_defaults(func=cmd_topic_show)

    topic_plan_parser = topic_subparsers.add_parser("plan")
    topic_plan_parser.add_argument("topic_id", type=int)
    topic_plan_parser.add_argument("--force", action="store_true")
    topic_plan_parser.add_argument("--n-evidence", type=_positive_int, default=10)
    topic_plan_parser.add_argument("--deepseek", action="store_true")
    topic_plan_parser.set_defaults(func=cmd_topic_plan)

    topic_report_parser = topic_subparsers.add_parser("report")
    topic_report_parser.add_argument("topic_id", type=int)
    topic_report_parser.add_argument("--n-evidence", type=_positive_int, default=20)
    topic_report_parser.add_argument("--local", action="store_true")
    topic_report_parser.add_argument("--force", action="store_true")
    topic_report_parser.set_defaults(func=cmd_topic_report)

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


def cmd_arxiv_search(args: argparse.Namespace) -> None:
    config = load_config()
    results = search_arxiv(args.query, max_results=args.max_results)
    output_csv = _resolve_arxiv_csv_path(Path(config["metadata_dir"]), args.output)
    csv_path = save_arxiv_results(results, output_csv)
    imported_count = import_arxiv_results_to_db(
        results,
        Path(config["metadata_dir"]) / "papers.sqlite",
    )

    downloaded_paths = []
    if args.download:
        downloaded_paths = download_arxiv_pdfs(
            results,
            Path(config["downloads_dir"]) / "arxiv",
            limit=args.download_limit,
        )

    print(f"results: {len(results)}")
    print(f"csv_path: {csv_path}")
    print(f"imported: {imported_count}")
    print(f"downloaded: {len(downloaded_paths)}")


def cmd_import_downloads(args: argparse.Namespace) -> None:
    config = load_config()
    result = import_downloaded_pdfs(
        config,
        source=args.source,
        mode="move" if args.move else "copy",
    )

    print(f"found: {result['found']}")
    print(f"imported: {result['imported']}")
    print(f"failed: {len(result['failed'])}")
    for path in result["papers"]:
        print(f"paper: {path}")
    for path in result["texts"]:
        print(f"text: {path}")
    for failure in result["failed"]:
        print(f"failed: {failure['file']}: {failure['error']}")


def cmd_explore(args: argparse.Namespace) -> None:
    if not args.explore_args:
        raise SystemExit("research-agent explore requires a question or subcommand")
    first = args.explore_args[0]
    if first == "list":
        return _cmd_explore_list(args.explore_args[1:])
    if first == "show":
        return _cmd_explore_show(args.explore_args[1:])
    if first == "continue":
        return _cmd_explore_continue(args.explore_args[1:])
    return _cmd_explore_question(args.explore_args)


def _cmd_explore_question(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="research-agent explore")
    parser.add_argument("question")
    parser.add_argument("--rounds", type=_positive_int, default=2)
    parser.add_argument("--n-evidence", type=_positive_int, default=8)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--topic-id", type=int, default=None)
    parser.add_argument("--assess-with-deepseek", action="store_true")
    parsed = parser.parse_args(argv)
    config = load_config()
    try:
        result = explore_question(
            config,
            question=parsed.question,
            rounds=parsed.rounds,
            n_evidence=parsed.n_evidence,
            use_deepseek=not parsed.local,
            topic_id=parsed.topic_id,
            assess_with_deepseek=parsed.assess_with_deepseek,
        )
    except ValueError as exc:
        message = str(exc)
        if "DeepSeek" in message:
            print(message)
            return
        raise

    print(f"用户问题: {parsed.question}")
    print("子问题:")
    for item in result["subquestions"]:
        print(f"- {item}")
    print("最终回答:")
    print(result["answer"])
    print(f"证据数量: {len(result['sources'])}")
    print(f"report_path: {result['report_path']}")
    print(f"json_path: {result['json_path']}")


def _cmd_explore_list(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="research-agent explore list")
    parser.parse_args(argv)
    config = load_config()
    explorations = list_explorations(config)
    for item in explorations:
        question = (item.get("question") or "")[:80]
        print(
            f"{item['id']}\t{item.get('created_at') or ''}\t"
            f"topic_id={item.get('topic_id')}\t"
            f"sources={item.get('sources_count')}\t"
            f"{question}\t{item.get('report_path') or ''}"
        )


def _cmd_explore_show(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="research-agent explore show")
    parser.add_argument("explore_id")
    parsed = parser.parse_args(argv)
    config = load_config()
    data = load_exploration(config, parsed.explore_id)
    print(f"question: {data.get('question') or ''}")
    print(f"topic_id: {data.get('topic_id')}")
    print(f"created_at: {data.get('created_at') or ''}")
    print("subquestions:")
    for item in data.get("subquestions") or []:
        print(f"- {item}")
    print("search_queries:")
    for item in data.get("search_queries") or []:
        print(f"- {item}")
    print("possible_gaps:")
    for item in data.get("possible_gaps") or []:
        print(f"- {item}")
    print("answer:")
    print((data.get("answer") or "")[:1000])
    print(f"report_path: {data.get('report_path') or ''}")
    print(f"json_path: {data.get('json_path') or ''}")


def _cmd_explore_continue(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="research-agent explore continue")
    parser.add_argument("explore_id")
    parser.add_argument("--rounds", type=_positive_int, default=1)
    parser.add_argument("--n-evidence", type=_positive_int, default=8)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--assess-with-deepseek", action="store_true")
    parsed = parser.parse_args(argv)
    config = load_config()
    try:
        result = continue_exploration(
            config,
            parsed.explore_id,
            rounds=parsed.rounds,
            n_evidence=parsed.n_evidence,
            use_deepseek=not parsed.local,
            assess_with_deepseek=parsed.assess_with_deepseek,
        )
    except ValueError as exc:
        message = str(exc)
        if "DeepSeek" in message:
            print(message)
            return
        raise
    print(f"continued_from: {parsed.explore_id}")
    print("最终回答:")
    print(result["answer"])
    print(f"证据数量: {len(result['sources'])}")
    print(f"report_path: {result['report_path']}")
    print(f"json_path: {result['json_path']}")


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


def cmd_topic_create(args: argparse.Namespace) -> None:
    config = load_config()
    topic = create_research_topic(
        config,
        name=args.name,
        description=args.description,
    )
    print(f"topic id: {topic['id']}")
    print(f"markdown: {topic['markdown_path']}")


def cmd_topic_list(_args: argparse.Namespace) -> None:
    config = load_config()
    topics = list_research_topics(config)
    print(f"topics: {len(topics)}")
    for topic in topics:
        print(f"{topic['id']}\t{topic.get('name') or ''}\t{topic.get('created_at') or ''}")


def cmd_topic_show(args: argparse.Namespace) -> None:
    config = load_config()
    topic = show_research_topic(config, args.topic_id)
    if topic is None:
        print(f"topic not found: {args.topic_id}")
        return
    for key, value in topic.items():
        print(f"{key}: {value or ''}")


def cmd_topic_plan(args: argparse.Namespace) -> None:
    config = load_config()
    plan_path = Path(config["metadata_dir"]) / "topics" / f"topic_{args.topic_id}_plan.md"
    existed = plan_path.exists()
    result = generate_topic_plan(
        config,
        args.topic_id,
        use_deepseek=args.deepseek,
        force=args.force,
        n_evidence=args.n_evidence,
    )
    if existed and not args.force:
        print(f"plan exists: {result['plan_path']}")
    else:
        print(f"plan generated: {result['plan_path']}")


def cmd_topic_report(args: argparse.Namespace) -> None:
    config = load_config()
    result = generate_topic_report(
        config,
        args.topic_id,
        use_deepseek=not args.local,
        n_evidence=args.n_evidence,
        force=args.force,
    )
    topic = result["topic"]
    print(f"topic id: {topic['id']}")
    print(f"topic name: {topic.get('name') or ''}")
    print(f"plan_path: {result['plan_path']}")
    print(f"report_path: {result['report_path']}")
    print(f"evidence_count: {result['evidence_count']}")
    print(f"unique_sources_count: {result['unique_sources_count']}")


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


def _resolve_arxiv_csv_path(metadata_dir: Path, output: str | None) -> Path:
    if output:
        path = Path(output)
        if path.suffix != ".csv":
            path = path.with_suffix(".csv")
        return path if path.is_absolute() else metadata_dir / path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return metadata_dir / f"arxiv_results_{timestamp}.csv"
