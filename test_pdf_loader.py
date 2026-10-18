
from src.pdf_loader import extract_text_from_pdf

pdf_path = "data/raw/sample.pdf"

pages = extract_text_from_pdf(pdf_path)

print(f"Pages containing text: {len(pages)}")

for page in pages:
    print(f"\nPage {page['page']}:")
    print(page["text"][:500])
