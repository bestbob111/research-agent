import pytest

from research_agent.chunking import chunk_text


def test_chunk_text_empty_returns_empty_list():
    assert chunk_text("") == []


def test_chunk_text_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=10, overlap=10)


def test_chunk_text_chunk_size_must_be_positive():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=0)


def test_chunk_text_mixed_chinese_english_returns_non_empty_chunks():
    text = "这是第一段 mixed English text.\n\n这是第二段 with numbers 123."

    chunks = chunk_text(text, chunk_size=30, overlap=5)

    assert chunks
    assert all(chunk for chunk in chunks)


def test_chunk_text_long_paragraph_is_split():
    text = "a" * 100

    chunks = chunk_text(text, chunk_size=30, overlap=5)

    assert len(chunks) > 1
    assert all(len(chunk) <= 30 for chunk in chunks)


def test_chunk_text_chunk_lengths_are_controlled():
    text = "\n\n".join(["段落一" * 10, "paragraph two " * 5, "段落三" * 10])

    chunks = chunk_text(text, chunk_size=50, overlap=10)

    assert chunks
    assert all(len(chunk) <= 50 for chunk in chunks)
