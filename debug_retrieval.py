
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pdf_loader import extract_text_from_pdf
from src.chunker import chunk_pages
from src.embedder import load_embedding_model, create_embeddings
from src.vector_store import build_vector_index, search_index
from src.retriever import retrieve_chunks
from src.generator import generate_answer


def main():
    # 1. Load the PDF and split its text into chunks.
    pdf_path = PROJECT_ROOT / "data" / "raw" / "sample.pdf"

    pages = extract_text_from_pdf(str(pdf_path))
    chunks = chunk_pages(pages)

    print(f"PDF pages: {len(pages)}")
    print(f"Text chunks: {len(chunks)}")

    # 2. Create embeddings and build the vector index.
    print("\nLoading embedding model...")
    model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(texts, model)
    index = build_vector_index(embeddings)

    # 3. Ask the question we are investigating.
    question = "Is feature filtering using correlation mandatory?"

    # 4. Search all chunks without applying the relevance threshold.
    query_embedding = create_embeddings([question], model)

    scores, ids = search_index(
        index,
        query_embedding[0],
        top_k=len(chunks),
    )

    print("\n===== RAW RETRIEVAL RESULTS =====")

    for rank, (score, chunk_id) in enumerate(
        zip(scores, ids), start=1
    ):
        chunk = chunks[int(chunk_id)]

        print(f"\nRank: {rank}")
        print(f"Similarity score: {float(score):.4f}")
        print(f"Page: {chunk['page']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print("Text:")
        print(chunk["text"][:1000])
        print("-" * 60)

    # 5. Test the normal retriever with its relevance threshold.
    retrieved = retrieve_chunks(
        question=question,
        model=model,
        index=index,
        chunks=chunks,
        top_k=5,
    )

    print("\n===== FILTERED RETRIEVAL RESULTS =====")
    print(f"Chunks returned: {len(retrieved)}")

    for chunk in retrieved:
        print(
            f"\nPage {chunk['page']}, "
            f"Chunk ID {chunk['chunk_id']}, "
            f"Score {chunk['score']:.4f}"
        )
        print(chunk["text"][:500])

    # 6. Check whether the generator answers using those chunks.
    print("\n===== GENERATOR RESULT =====")

    response = generate_answer(question, retrieved)
    print("Answer:", response["answer"])
    print("Citations:", response["citations"])


if __name__ == "__main__":
    main()
