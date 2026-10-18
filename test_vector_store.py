
from src.pdf_loader import extract_text_from_pdf
from src.chunker import chunk_pages
from src.embedder import load_embedding_model, create_embeddings
from src.vector_store import build_vector_index, search_index


# Step 1: Load and split the PDF
pages = extract_text_from_pdf("data/raw/sample.pdf")
chunks = chunk_pages(pages)

# Step 2: Load the embedding model
model = load_embedding_model()

# Step 3: Create embeddings for every document chunk
texts = [chunk["text"] for chunk in chunks]
document_embeddings = create_embeddings(texts, model)

# Step 4: Build the FAISS index
index = build_vector_index(document_embeddings)

# Step 5: Convert the user's question into an embedding
question = "Which machine learning algorithms should be used?"
query_embedding = create_embeddings([question], model)

# Step 6: Search for the three most relevant chunks
scores, ids = search_index(index, query_embedding[0], top_k=3)

# Step 7: Display the search results
print(f"\nQuestion: {question}")

for rank, (score, chunk_id) in enumerate(zip(scores, ids), start=1):
    chunk = chunks[int(chunk_id)]

    print(f"\nResult {rank}")
    print(f"Similarity score: {score:.4f}")
    print(f"Source page: {chunk['page']}")
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Text: {chunk['text'][:300]}")
