
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model():
    """Load the model used to convert text into embeddings."""
    return SentenceTransformer(MODEL_NAME)


def create_embeddings(texts: list[str], model) -> list:
    """Convert a list of text passages into numerical vectors."""
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings
