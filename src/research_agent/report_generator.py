from datetime import datetime
from pathlib import Path


def generate_review_report(
    question: str,
    review: str,
    sources: list[dict],
    output_path: Path,
    metadata: dict | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = metadata or {}
    lines = [
        "# 科研文献调研报告",
        "",
        "## 1. 研究问题",
        "",
        question,
        "",
        "## 2. 生成信息",
        "",
        f"- 生成时间: {info.get('generated_at') or datetime.now().isoformat(timespec='seconds')}",
        f"- 使用模型: {info.get('model') or ''}",
        f"- 证据片段数量: {info.get('evidence_count', len(sources))}",
        f"- 文献来源数量: {info.get('unique_sources_count', _count_unique_sources(sources))}",
        "",
        "## 3. 综合分析",
        "",
        review,
        "",
        "## 4. 证据来源",
        "",
        "| 编号 | title | authors | year | source | chunk_id | distance |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, source in enumerate(sources, start=1):
        lines.append(
            "| "
            f"{source.get('evidence_id') or index} | "
            f"{_cell(source.get('title'))} | "
            f"{_cell(source.get('authors'))} | "
            f"{_cell(source.get('year'))} | "
            f"{_cell(source.get('source'))} | "
            f"{_cell(source.get('chunk_id'))} | "
            f"{_cell(source.get('distance'))} |"
        )

    lines.extend(
        [
            "",
            "## 5. 使用说明与注意事项",
            "",
            "- 本报告由本地文献库检索片段和大模型生成；",
            "- 结论应回到原文献核查；",
            "- 若证据不足，应继续补充检索；",
            "- 不应将模型推断直接等同于文献事实。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _cell(value) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _count_unique_sources(sources: list[dict]) -> int:
    values = {
        source.get("source") or source.get("title") or source.get("id")
        for source in sources
        if source.get("source") or source.get("title") or source.get("id")
    }
    return len(values)
