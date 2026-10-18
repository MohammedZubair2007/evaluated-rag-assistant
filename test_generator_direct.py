
import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"

context = """
Page 2 of the document states:

"Features filtering using correlation is optional."

It separately states:

"Independence Testing: Perform a Pearson or Spearman Correlation
test between input features and the target KPI to test for
independence and to check potential target leakage."
"""

question = "Is feature filtering using correlation mandatory?"

prompt = f"""
You must answer the question based only on the document excerpt.

DOCUMENT:
{context}

QUESTION:
{question}

Read these two statements carefully:
- Feature filtering using correlation is optional.
- A Pearson or Spearman correlation test is specified under
  Independence Testing.

Write a complete answer with exactly these two parts:
1. Feature filtering: state whether it is optional or mandatory.
2. Correlation testing: state what the instructions say to do.

Do not answer with only "Yes" or "No".
Do not confuse feature filtering with correlation testing.
Do not add information that is absent from the excerpt.

Return valid JSON:
{{
  "answer": "your complete answer",
  "citations": [2]
}}
"""

payload = {
    "model": "llama3.2:3b",
    "messages": [
        {
            "role": "user",
            "content": prompt,
        }
    ],
    "stream": False,
    "format": "json",
    "options": {
        "temperature": 0,
    },
}

request = urllib.request.Request(
    OLLAMA_URL,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    answer = json.loads(result["message"]["content"])

    print("\n===== DIRECT GENERATOR TEST =====")
    print(json.dumps(answer, indent=2, ensure_ascii=False))

except Exception as error:
    print(f"Test failed: {error}")
