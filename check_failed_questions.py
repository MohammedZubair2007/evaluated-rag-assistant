
import json

with open(
    "evaluation/evaluation_results.json",
    "r",
    encoding="utf-8",
) as file:
    data = json.load(file)

print("\n===== FAILED EVALUATION QUESTIONS =====\n")

failed_count = 0

for result in data["results"]:
    if not result["abstention_correct"]:
        failed_count += 1

        print("ID:", result["id"])
        print("Question:", result["question"])
        print("Expected answerable:", result["expected_answerable"])
        print("Assistant abstained:", result["abstained"])
        print("Actual answer:", result["answer"])
        print("-" * 50)

print("Total failed questions:", failed_count)
