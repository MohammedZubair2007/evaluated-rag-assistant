
import json
import sys
from pathlib import Path

# Locate the project root so imports work when this script is run.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pdf_loader import extract_text_from_pdf
from src.chunker import chunk_pages
from src.embedder import load_embedding_model, create_embeddings
from src.vector_store import build_vector_index
from src.retriever import retrieve_chunks
from src.generator import generate_answer


def is_abstention(answer, retrieved_chunks):
    """Check whether the assistant declines to answer."""
    if not isinstance(answer, str) or not answer.strip():
        return True

    answer_text = answer.strip().lower()

    abstention_phrases = [
        "could not find",
        "couldn't find",
        "not enough evidence",
        "not enough information",
        "insufficient information",
        "cannot determine",
        "can't determine",
        "does not provide",
        "does not specify",
        "not specified",
        "unable to answer",
        "i don't know",
    ]

    return (
        not retrieved_chunks
        or any(
            phrase in answer_text
            for phrase in abstention_phrases
        )
    )


def check_citations(
    citations,
    expected_pages,
    retrieved_pages,
    expected_answerable,
):
    """
    Check page-number coverage and validity.

    For answerable questions, all expected pages must be cited.
    For unanswerable questions, citations are allowed if they
    refer to retrieved pages. An empty citation list is also valid.

    This checks page references, not semantic support.
    """
    if not isinstance(citations, list):
        return False

    if any(
        not isinstance(page, int) or isinstance(page, bool)
        for page in citations
    ):
        return False

    if len(citations) != len(set(citations)):
        return False

    if not set(citations).issubset(set(retrieved_pages)):
        return False

    if expected_answerable:
        return (
            bool(expected_pages)
            and bool(citations)
            and set(expected_pages).issubset(set(citations))
        )

    # For unanswerable questions, valid evidence citations are allowed.
    return True


def main():
    # 1. Load evaluation questions.
    questions_path = PROJECT_ROOT / "evaluation" / "questions.json"

    with open(questions_path, "r", encoding="utf-8") as file:
        questions = json.load(file)

    # 2. Load and process the PDF.
    pdf_path = PROJECT_ROOT / "data" / "raw" / "sample.pdf"

    pages = extract_text_from_pdf(str(pdf_path))
    chunks = chunk_pages(pages)

    print(f"PDF pages loaded: {len(pages)}")
    print(f"Text chunks created: {len(chunks)}")

    # 3. Create embeddings and the vector index.
    print("\nLoading embedding model...")
    model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(texts, model)
    index = build_vector_index(embeddings)

    # 4. Evaluate every question.
    results = []

    for item in questions:
        question_id = item["id"]
        question = item["question"]
        expected_answerable = item["answerable"]
        expected_pages = item.get("expected_pages", [])

        print("\n" + "=" * 60)
        print(f"Question {question_id}: {question}")

        retrieved_chunks = retrieve_chunks(
            question=question,
            model=model,
            index=index,
            chunks=chunks,
            top_k=5,
        )

        response = generate_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        answer = response.get("answer", "")
        citations = response.get("citations", [])

        abstained = is_abstention(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
        )

        expected_abstention = not expected_answerable
        abstention_correct = abstained == expected_abstention

        answer_present = (
            isinstance(answer, str)
            and bool(answer.strip())
            and not abstained
        )

        answer_presence_correct = (
            answer_present if expected_answerable else abstained
        )

        retrieved_pages = sorted({
            chunk["page"] for chunk in retrieved_chunks
        })

        citation_correct = check_citations(
            citations=citations,
            expected_pages=expected_pages,
            retrieved_pages=retrieved_pages,
            expected_answerable=expected_answerable,
        )

        result = {
            "id": question_id,
            "question": question,
            "expected_answerable": expected_answerable,
            "expected_answer": item.get("expected_answer"),
            "expected_pages": expected_pages,
            "answer": answer,
            "citations": citations,
            "retrieved_pages": retrieved_pages,
            "abstained": abstained,
            "abstention_correct": abstention_correct,
            "answer_presence_correct": answer_presence_correct,
            "citation_correct": citation_correct,
        }

        results.append(result)

        print(f"Answer: {answer}")
        print(f"Expected pages: {expected_pages}")
        print(f"Citations: {citations}")
        print(f"Retrieved pages: {retrieved_pages}")
        print(f"Abstention decision correct: {abstention_correct}")
        print(f"Answer presence correct: {answer_presence_correct}")
        print(f"Citation check passed: {citation_correct}")

    # 5. Calculate metrics.
    total = len(results)

    correct_abstention = sum(
        result["abstention_correct"] for result in results
    )

    correct_answer_presence = sum(
        result["answer_presence_correct"] for result in results
    )

    correct_citations = sum(
        result["citation_correct"] for result in results
    )

    abstention_accuracy = (
        correct_abstention / total if total else 0.0
    )

    answer_presence_accuracy = (
        correct_answer_presence / total if total else 0.0
    )

    citation_accuracy = (
        correct_citations / total if total else 0.0
    )

    report = {
        "total_questions": total,
        "correct_abstention_decisions": correct_abstention,
        "abstention_decision_accuracy": abstention_accuracy,
        "correct_answer_presence_checks": correct_answer_presence,
        "answer_presence_accuracy": answer_presence_accuracy,
        "correct_citation_checks": correct_citations,
        "citation_check_accuracy": citation_accuracy,
        "results": results,
    }

    # 6. Save the report.
    output_path = (
        PROJECT_ROOT
        / "evaluation"
        / "evaluation_results.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # 7. Print the summary.
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total questions: {total}")
    print(
        f"Abstention decisions: {correct_abstention}/{total} "
        f"({abstention_accuracy:.1%})"
    )
    print(
        f"Answer presence checks: {correct_answer_presence}/{total} "
        f"({answer_presence_accuracy:.1%})"
    )
    print(
        f"Citation checks: {correct_citations}/{total} "
        f"({citation_accuracy:.1%})"
    )
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
