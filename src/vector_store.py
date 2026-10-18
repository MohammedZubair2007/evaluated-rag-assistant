
import faiss
import numpy as np


def build_vector_index(embeddings: np.ndarray):
    """Build a FAISS index using normalized text embeddings."""

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2D array.")

    # FAISS expects 32-bit floating-point vectors.
    vectors = np.asarray(embeddings, dtype=np.float32)

    # Inner product equals cosine similarity for normalized vectors.
    index = faiss.IndexFlatIP(vectors.shape[1])

    # Add all document vectors to the index.
    index.add(vectors)

    return index


def search_index(index, query_embedding: np.ndarray, top_k: int = 3):
    """Return the top matching vector IDs and similarity scores."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if index.ntotal == 0:
        raise ValueError("The vector index is empty.")

    query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)

    if query.shape[1] != index.d:
        raise ValueError("Query and document embeddings must have equal dimensions.")

    k = min(top_k, index.ntotal)
    scores, ids = index.search(query, k)

    return scores[0], ids[0]
