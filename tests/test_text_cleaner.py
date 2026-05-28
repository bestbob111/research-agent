from research_agent.text_cleaner import clean_text


def test_clean_text_empty_string():
    assert clean_text("") == ""


def test_clean_text_normalizes_windows_newlines():
    assert clean_text("a\r\nb\rc") == "a\nb\nc"


def test_clean_text_compresses_multiple_blank_lines():
    assert clean_text("a\n\n\n\nb") == "a\n\nb"


def test_clean_text_preserves_page_marker():
    assert "--- Page 1 ---" in clean_text("--- Page 1 ---\ncontent")


def test_clean_text_preserves_chinese_and_formula_symbols():
    text = "中文 English 123 E=mc^2 α + β ≥ γ"
    assert clean_text(text) == text
