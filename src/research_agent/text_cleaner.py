import re
import unicodedata


def clean_text(text: str) -> str:
    """Normalize extracted text while preserving content."""
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized


def prepare_text_for_embedding(text: str, max_chars: int = 1800) -> str:
    """Prepare a text chunk for embedding without removing visible content."""
    if text is None or not str(text).strip():
        return ""
    if max_chars <= 0:
        return ""

    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = "".join(
        char
        for char in normalized
        if char in ("\n", "\t") or not unicodedata.category(char).startswith("C")
    )
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = normalized.strip()
    return normalized[:max_chars]
