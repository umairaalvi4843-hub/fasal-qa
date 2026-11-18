import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

# 1. LOAD THE REAL EXTRACTED CORPUS

CORPUS_PATH = Path("data/corpus.json")

with open(CORPUS_PATH, "r", encoding="utf-8") as file:
    CORPUS = json.load(file)

if not CORPUS:
    raise ValueError("The corpus is empty. Check data/corpus.json.")

print("Loaded passages:", len(CORPUS))


# 2. TOKENIZER

def tokenize(text):
    """Convert text into lowercase tokens."""
    return re.findall(r"[\w]+", text.lower())


# 3. BUILD THE BM25 INDEX

passage_ids = [passage["id"] for passage in CORPUS]
passage_texts = [passage["text"] for passage in CORPUS]

tokenized_corpus = [
    tokenize(pid + " " + text)
    for pid, text in zip(passage_ids, passage_texts)
]

bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)


# 4. SEARCH FUNCTION

def bm25_search(query, k=3):
    """Return the top-k passages matching a query."""

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    sorted_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for i in sorted_indices[:k]:
        results.append({
            "id": passage_ids[i],
            "score": float(scores[i]),
            "text": passage_texts[i]
        })

    return results


# 5. TEST QUERIES

if __name__ == "__main__":

    print("\n--- BM25 REAL CORPUS BASELINE ---")

    test_queries = [
        "PM Kisan amount",
        "installment",
        "crop insurance",
        "registration fee",
        "how to get a loan",
        "eligibility for PM Kisan",
        "compensation for crop loss due to rainfall"
    ]

    all_results = []

    for query in test_queries:

        print("\nQuery:", query)

        results = bm25_search(query, k=3)

        query_results = {
            "query": query,
            "results": []
        }

        for rank, result in enumerate(results, start=1):

            print(
                f"\n  {rank}. {result['id']} "
                f"(score: {result['score']:.4f})"
            )

            print("     " + result["text"][:350].replace("\n", " "))

            query_results["results"].append(result)

        all_results.append(query_results)

    # 6. SAVE BASELINE RESULTS

    output_path = Path("data/bm25_baseline_results.json")

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(all_results, file, ensure_ascii=False, indent=2)

    print("\n--------------------------------")
    print("BM25 evaluation completed.")
    print("Results saved to:", output_path)
