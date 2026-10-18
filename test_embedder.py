
from src.pdf_loader import extract_text_from_pdf
from src.chunker import chunk_pages
from src.embedder import load_embedding_model, create_embeddings


# Step 1: Extract text from the PDF
pages = extract_text_from_pdf("data/raw/sample.pdf")

# Step 2: Split the extracted text into chunks
chunks = chunk_pages(pages)

# Step 3: Load the embedding model
model = load_embedding_model()

# Step 4: Convert each chunk into an embedding
texts = [chunk["text"] for chunk in chunks]
embeddings = create_embeddings(texts, model)

# Step 5: Display the results
print(f"Total chunks: {len(chunks)}")
print(f"Embedding shape: {embeddings.shape}")
print(f"First embedding (first 10 values): {embeddings[0][:10]}")
