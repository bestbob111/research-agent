import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from research_agent.config import load_config
from research_agent.reviewer import EvidenceReviewer


def main() -> None:
    parser = argparse.ArgumentParser(description="Review local evidence with DeepSeek.")
    parser.add_argument("question")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--output", default=None, help="Optional report filename.")
    args = parser.parse_args()

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
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = _resolve_report_path(reports_dir, args.output)
    report_path.write_text(
        _format_report(question, result["review"], result["sources"]),
        encoding="utf-8",
    )
    print(f"报告已保存: {report_path}")


def _resolve_report_path(reports_dir: Path, output: str | None) -> Path:
    if output:
        path = Path(output)
        if path.suffix != ".md":
            path = path.with_suffix(".md")
        return reports_dir / path.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return reports_dir / f"review_{timestamp}.md"


def _format_report(question: str, review: str, sources: list[dict]) -> str:
    lines = [
        "# DeepSeek 综述报告",
        "",
        "## 问题",
        "",
        question,
        "",
        "## 回答",
        "",
        review,
        "",
        "## 证据来源",
        "",
    ]
    for source in sources:
        lines.append(
            "- "
            f"source={source.get('source', '')}, "
            f"chunk_id={source.get('chunk_id', '')}, "
            f"distance={source.get('distance')}"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
