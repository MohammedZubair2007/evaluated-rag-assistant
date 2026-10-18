
from src.pdf_loader import extract_text_from_pdf
from src.chunker import chunk_pages
from src.embedder import load_embedding_model, create_embeddings
from src.vector_store import build_vector_index
from src.retriever import retrieve_chunks
from src.generator import generate_answer


# Step 1: Load the PDF and extract its text
pdf_path = "data/raw/sample.pdf"
pages = extract_text_from_pdf(pdf_path)

# Step 2: Split the text into chunks
chunks = chunk_pages(pages)

# Step 3: Load the embedding model
model = load_embedding_model()

# Step 4: Generate embeddings for all document chunks
texts = [chunk["text"] for chunk in chunks]
document_embeddings = create_embeddings(texts, model)

# Step 5: Build the FAISS search index
index = build_vector_index(document_embeddings)

# Step 6: Ask a question
question = input("\nAsk a question about the PDF: ")

# Step 7: Retrieve relevant passages
results = retrieve_chunks(
    question=question,
    model=model,
    index=index,
    chunks=chunks,
    top_k=5,
)

# Step 8: Generate an answer using Ollama
answer = generate_answer(question, results)

# Step 9: Display the answer and citations
print("\nAnswer:")
print(answer["answer"])

print("\nCitations:")
if answer["citations"]:
    for page_number in answer["citations"]:
        print(f"- Page {page_number}")
else:
    print("No citations returned.")
