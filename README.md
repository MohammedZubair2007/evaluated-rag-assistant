
# Evaluated RAG Knowledge Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions from PDF documents using semantic search, vector embeddings, and a locally running Large Language Model (LLM).

The project focuses on retrieving relevant document passages, generating evidence-based answers, providing page citations, and evaluating how the system handles questions that cannot be answered from the supplied document.

## Project Overview

Traditional language models can produce incorrect or unsupported answers. This project addresses that challenge by retrieving relevant passages from a source PDF and supplying them as context to a local LLM.

The assistant is designed to:
- Extract text from PDF documents.
- Split document text into manageable chunks.
- Convert chunks into numerical embeddings.
- Retrieve relevant chunks using semantic similarity.
- Generate answers using a local LLM.
- Return page citations where available.
- Abstain when sufficient evidence is unavailable.
- Evaluate answer and citation-related behavior.

## Architecture

1. **PDF Loading:** PyMuPDF extracts text and page numbers from the PDF.
2. **Text Chunking:** The extracted text is divided into overlapping chunks.
3. **Embedding Generation:** Sentence Transformers converts text into vector embeddings.
4. **Vector Search:** FAISS retrieves relevant chunks using similarity search.
5. **Answer Generation:** Ollama runs the Llama 3.2 3B model locally to generate answers from retrieved context.
6. **Evaluation:** A set of questions measures abstention decisions, answer presence, and citation page matching.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| PDF processing | PyMuPDF |
| Embeddings | Sentence Transformers |
| Embedding model | all-MiniLM-L6-v2 |
| Vector database/index | FAISS |
| Local LLM runtime | Ollama |
| Language model | Llama 3.2 3B |
| Numerical computing | NumPy |
| Evaluation output | JSON |

## Project Structure

```text
evaluated-rag-assistant/
├── data/
│   ├── raw/
│   │   └── sample.pdf
│   └── processed/
├── src/
│   ├── pdf_loader.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── generator.py
├── evaluation/
│   ├── questions.json
│   ├── run_evaluation.py
│   └── evaluation_results.json
├── tests/
├── logs/
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup and Installation

### Prerequisites

- Python 3.12
- Git
- Ollama
- The Llama 3.2 3B model
- A PDF document to process

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd evaluated-rag-assistant
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Set up Ollama

Install Ollama, then download the model:

```powershell
ollama pull llama3.2:3b
```

Ensure Ollama is running and the model is available before generating answers.

### 5. Add a PDF document

Place your PDF at:

```text
data/raw/sample.pdf
```

Use a document you are permitted to process and share.

## Running the Evaluation

From the project root, with the virtual environment activated, run:

```powershell
python evaluation/run_evaluation.py
```

The script processes the PDF, builds the vector index, answers the evaluation questions, and writes the report to:

```text
evaluation/evaluation_results.json
```

The evaluation report records the questions, generated answers, citations, retrieved pages, and evaluation checks.

## Evaluation Results

The latest recorded run used 12 evaluation questions.

| Metric | Result |
|---|---:|
| Abstention-decision accuracy | 100% (12/12) |
| Answer-presence checks | 100% (12/12) |
| Citation page-matching checks | 91.7% (11/12) |

These metrics describe the current evaluation checks; they do not establish that every generated answer is factually correct or every citation semantically supports its answer.

## Current Limitations

- The evaluation set contains only 12 questions and one sample PDF.
- Citation checking verifies page-number matching, not full semantic support.
- One answerable evaluation question currently returns no citation.
- Retrieval quality may vary with document structure, chunk size, and similarity thresholds.
- The system has not yet been validated on a large collection of diverse documents.
- Answer quality depends partly on the local language model and retrieved context.

## Future Improvements

- Improve citation coverage and evidence verification.
- Expand the evaluation dataset with more answerable and unanswerable questions.
- Add automated tests for the individual pipeline components.
- Build a user interface or API for document uploads and question answering.
- Evaluate retrieval quality using metrics such as Recall@K.
- Test the assistant across different PDFs and document types.

## Learning Outcomes

This project demonstrates practical experience with:

- Retrieval-Augmented Generation (RAG)
- Natural language processing and text embeddings
- Semantic search and vector indexing
- Local LLM integration
- PDF text extraction and chunking
- Evidence-based question answering
- Evaluation of abstention and citation behavior

## License

Add an appropriate license before distributing this project publicly.
