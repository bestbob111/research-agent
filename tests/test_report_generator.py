from research_agent.report_generator import generate_review_report


def test_generate_review_report_writes_markdown(tmp_path):
    output_path = tmp_path / "reports" / "review.md"
    sources = [
        {
            "evidence_id": 1,
            "title": "Paper Title",
            "authors": "Author A",
            "year": "2024",
            "source": "paper.txt",
            "chunk_id": "doc::chunk_0",
            "distance": 0.12,
        }
    ]

    result_path = generate_review_report(
        question="研究问题",
        review="综合分析正文",
        sources=sources,
        output_path=output_path,
        metadata={"model": "deepseek-chat", "evidence_count": 1, "unique_sources_count": 1},
    )

    text = output_path.read_text(encoding="utf-8")
    assert result_path == output_path
    assert output_path.exists()
    assert "# 科研文献调研报告" in text
    assert "## 1. 研究问题" in text
    assert "## 4. 证据来源" in text
    assert "Paper Title" in text
    assert "paper.txt" in text
    assert "doc::chunk_0" in text
