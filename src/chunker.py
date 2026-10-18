
def chunk_pages(
    pages: list[dict],
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[dict]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk_size.")

    chunks = []

    for page in pages:
        text = page["text"]
        page_number = page["page"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page_number,
                    "chunk_id": len(chunks),
                })

            if end == len(text):
                break

            start = end - overlap

    return chunks
