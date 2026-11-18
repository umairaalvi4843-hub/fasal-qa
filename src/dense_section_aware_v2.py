import json
import numpy as np
from sentence_transformers import SentenceTransformer

CORPUS_PATH = "data/corpus_section_aware_v2.json"
EVALUATION_PATH = "data/evaluation_queries_section_aware_v2.json"

RESULTS_PATH = "data/dense_section_aware_v2_results.json"
METRICS_PATH = "data/dense_section_aware_v2_metrics.json"

TOP_K = 3
MODEL_NAME = "intfloat/multilingual-e5-small"

with open(CORPUS_PATH, encoding="utf-8") as f:
    corpus = json.load(f)

with open(EVALUATION_PATH, encoding="utf-8") as f:
    evaluation_queries = json.load(f)

print("Loading multilingual E5 model...")
model = SentenceTransformer(MODEL_NAME)

documents = [
    "passage: " + p["text"]
    for p in corpus
]

document_ids = [p["id"] for p in corpus]

print("Encoding corpus passages...")
document_embeddings = model.encode(
    documents,
    normalize_embeddings=True,
    show_progress_bar=True
)

all_results = []
recalls = []
reciprocal_ranks = []

for item in evaluation_queries:
    query = item["query"]
    relevant = set(item["relevant_passages"])

    query_embedding = model.encode(
        ["query: " + query],
        normalize_embeddings=True
    )[0]

    scores = np.dot(document_embeddings, query_embedding)
    ranked_indices = np.argsort(scores)[::-1][:TOP_K]

    top_results = []

    for index in ranked_indices:
        top_results.append({
            "id": document_ids[index],
            "score": float(scores[index]),
            "text": corpus[index]["text"]
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
        "relevant_passages": sorted(relevant),
        "top_results": top_results,
        "recall_at_3": recall,
        "reciprocal_rank": reciprocal_rank
    })

    print("\n" + "=" * 75)
    print("QUERY:", query)
    print("RETRIEVED:", retrieved_ids)
    print(f"Recall@3: {recall:.4f}")
    print(f"RR: {reciprocal_rank:.4f}")

metrics = {
    "experiment": "Multilingual E5 on section-aware v2 corpus",
    "model": MODEL_NAME,
    "total_queries": len(evaluation_queries),
    "corpus_passages": len(corpus),
    "recall_at_3": float(np.mean(recalls)),
    "mrr_at_3": float(np.mean(reciprocal_ranks))
}

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=4)

print("\n" + "=" * 75)
print("SECTION-AWARE V2 DENSE RETRIEVAL RESULTS")
print("Total queries:", metrics["total_queries"])
print("Corpus passages:", metrics["corpus_passages"])
print(f"Recall@3: {metrics['recall_at_3']:.4f}")
print(f"MRR@3: {metrics['mrr_at_3']:.4f}")
print("Saved:", RESULTS_PATH)
print("Saved:", METRICS_PATH)
