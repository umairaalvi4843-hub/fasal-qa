import json
import re
import numpy as np
from rank_bm25 import BM25Okapi

CORPUS_PATH = "data/corpus_section_aware_v2.json"
EVALUATION_PATH = "data/evaluation_queries_section_aware_v2.json"

RESULTS_PATH = "data/bm25_filtered_section_aware_v2_results.json"
METRICS_PATH = "data/bm25_filtered_section_aware_v2_metrics.json"

TOP_K = 3


def tokenize(text):
    return re.findall(r"[\w]+", text.lower())


def detect_scheme(query):
    q = query.lower()

    if "kcc" in q or "kisan credit card" in q:
        return "KCC"

    if any(term in q for term in [
        "pmfby", "crop insurance", "fasal bima"
    ]):
        return "PMFBY"

    if "pm-kisan" in q or "pmkisan" in q:
        return "PMKISAN"

    return None


with open(CORPUS_PATH, encoding="utf-8") as f:
    corpus = json.load(f)

with open(EVALUATION_PATH, encoding="utf-8") as f:
    evaluation_queries = json.load(f)

all_results = []
recalls = []
reciprocal_ranks = []

for item in evaluation_queries:
    query = item["query"]
    relevant = set(item["relevant_passages"])
    scheme = detect_scheme(query)

    filtered_corpus = [
        p for p in corpus
        if scheme is None or p["id"].startswith(scheme + "_SA2")
    ]

    documents = [p["text"] for p in filtered_corpus]
    document_ids = [p["id"] for p in filtered_corpus]

    bm25 = BM25Okapi([tokenize(doc) for doc in documents])

    scores = bm25.get_scores(tokenize(query))
    ranked_indices = np.argsort(scores)[::-1][:TOP_K]

    top_results = []

    for index in ranked_indices:
        top_results.append({
            "id": document_ids[index],
            "score": float(scores[index]),
            "text": documents[index]
        })

    retrieved_ids = [r["id"] for r in top_results]

    hits = len(set(retrieved_ids) & relevant)
    recall = hits / len(relevant) if relevant else 0.0

    reciprocal_rank = 0.0
    for rank, passage_id in enumerate(retrieved_ids, start=1):
        if passage_id in relevant:
            reciprocal_rank = 1 / rank
            break

    recalls.append(recall)
    reciprocal_ranks.append(reciprocal_rank)

    all_results.append({
        "query": query,
        "scheme": scheme,
        "relevant_passages": sorted(relevant),
        "top_results": top_results,
        "recall_at_3": recall,
        "reciprocal_rank": reciprocal_rank
    })

    print("\n" + "=" * 75)
    print("QUERY:", query)
    print("SCHEME FILTER:", scheme)
    print("RETRIEVED:", retrieved_ids)
    print(f"Recall@3: {recall:.4f}")
    print(f"RR: {reciprocal_rank:.4f}")

metrics = {
    "experiment": "Scheme-filtered BM25 on section-aware v2 corpus",
    "total_queries": len(evaluation_queries),
    "recall_at_3": float(np.mean(recalls)),
    "mrr_at_3": float(np.mean(reciprocal_ranks))
}

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=4)

print("\n" + "=" * 75)
print("FILTERED BM25 RESULTS")
print("Total queries:", metrics["total_queries"])
print(f"Recall@3: {metrics['recall_at_3']:.4f}")
print(f"MRR@3: {metrics['mrr_at_3']:.4f}")
print("Saved:", RESULTS_PATH)
print("Saved:", METRICS_PATH)
