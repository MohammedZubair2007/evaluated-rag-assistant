
from src.pdf_loader import extract_text_from_pdf
from src.chunker import chunk_pages

pdf_path = "data/raw/sample.pdf"

pages = extract_text_from_pdf(pdf_path)
chunks = chunk_pages(pages)

print(f"Pages containing text: {len(pages)}")
print(f"Total chunks created: {len(chunks)}")

for chunk in chunks[:3]:
    print(f"\nChunk ID: {chunk['chunk_id']}")
    print(f"Source page: {chunk['page']}")
    print(f"Chunk length: {len(chunk['text'])} characters")
    print(f"Text preview: {chunk['text'][:200]}")
