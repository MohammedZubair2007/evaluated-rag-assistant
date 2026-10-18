
from src.embedder import create_embeddings
from src.vector_store import search_index


def retrieve_chunks(
    question: str,
    model,
    index,
    chunks: list[dict],
    top_k: int = 5,
    min_score: float = 0.20,
) -> list[dict]:
    """Retrieve relevant chunks and filter weak matches."""

    if not chunks or index.ntotal == 0:
        return []

    query_embedding = create_embeddings([question], model)

    # Search more chunks first, then filter by relevance.
    candidate_k = min(max(top_k * 3, 10), index.ntotal)

    scores, ids = search_index(
        index,
        query_embedding[0],
        top_k=candidate_k,
    )

    results = []

    for score, chunk_id in zip(scores, ids):
        if int(chunk_id) < 0:
            continue

        score = float(score)

        if score < min_score:
            continue

        chunk = chunks[int(chunk_id)]

        results.append({
            "text": chunk["text"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "score": score,
        })

    results.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return results[:top_k]
