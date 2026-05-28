def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """Split text into chunks, preferring paragraph boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text:
        return []

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if paragraph == "":
            separator = "\n\n" if current else ""
            candidate = f"{current}{separator}\n\n"
        else:
            candidate = f"{current}\n\n{paragraph}" if current else paragraph

        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_text(paragraph, chunk_size, overlap))
        elif len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = _with_overlap(chunks[-1], paragraph, overlap) if chunks else paragraph
            if len(current) > chunk_size:
                chunks.extend(_split_long_text(current, chunk_size, overlap))
                current = ""

    if current:
        chunks.append(current)

    return [chunk for chunk in chunks if chunk]


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunk = text[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def _with_overlap(previous: str, text: str, overlap: int) -> str:
    if overlap == 0 or not previous:
        return text
    return previous[-overlap:] + text
