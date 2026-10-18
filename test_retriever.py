
from src.pdf_loader import extract_text_from_pdf
from src.chunker import chunk_pages
from src.embedder import load_embedding_model, create_embeddings
from src.vector_store import build_vector_index
from src.retriever import retrieve_chunks


pages = extract_text_from_pdf("data/raw/sample.pdf")
chunks = chunk_pages(pages)

model = load_embedding_model()

texts = [chunk["text"] for chunk in chunks]
embeddings = create_embeddings(texts, model)

index = build_vector_index(embeddings)

question = input("\nEnter your question: ")

results = retrieve_chunks(
    question=question,
    model=model,
    index=index,
    chunks=chunks,
    top_k=3,
)

print(f"\nQuestion: {question}")

for rank, result in enumerate(results, start=1):
    print(f"\nResult {rank}")
    print(f"Page: {result['page']}")
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Similarity: {result['score']:.4f}")
    print(f"Text: {result['text'][:300]}")
