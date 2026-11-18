import json
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CORPUS_PATH = "data/corpus_section_aware_v2.json"
EVALUATION_PATH = "data/evaluation_queries_section_aware_v2.json"

RESULTS_PATH = "data/hybrid_section_aware_v2_results.json"
METRICS_PATH = "data/hybrid_section_aware_v2_metrics.json"

MODEL_NAME = "intfloat/multilingual-e5-small"
TOP_K = 3
RRF_K = 60


def tokenize(text):
    return re.findall(r"[\w]+", text.lower())


with open(CORPUS_PATH, encoding="utf-8") as f:
    corpus = json.load(f)

with open(EVALUATION_PATH, encoding="utf-8") as f:
    evaluation_queries = json.load(f)

document_ids = [p["id"] for p in corpus]
documents = [p["text"] for p in corpus]

print("Loading multilingual E5 model...")
model = SentenceTransformer(MODEL_NAME)

print("Encoding corpus passages...")
document_embeddings = model.encode(
    ["passage: " + text for text in documents],
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Building BM25 index...")
bm25 = BM25Okapi([tokenize(text) for text in documents])

all_results = []
recalls = []
reciprocal_ranks = []

for item in evaluation_queries:
    query = item["query"]
    relevant = set(item["relevant_passages"])

    # BM25 ranking
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_ranked = np.argsort(bm25_scores)[::-1]

    # Dense E5 ranking
    query_embedding = model.encode(
        ["query: " + query],
        normalize_embeddings=True
    )[0]

    dense_scores = np.dot(document_embeddings, query_embedding)
    dense_ranked = np.argsort(dense_scores)[::-1]

    # Reciprocal Rank Fusion
    rrf_scores = {}

    for rank, index in enumerate(bm25_ranked, start=1):
        passage_id = document_ids[index]
        rrf_scores[passage_id] = (
            rrf_scores.get(passage_id, 0.0) +
            1 / (RRF_K + rank)
        )

    for rank, index in enumerate(dense_ranked, start=1):
        passage_id = document_ids[index]
        rrf_scores[passage_id] = (
            rrf_scores.get(passage_id, 0.0) +
            1 / (RRF_K + rank)
        )

    ranked_ids = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )[:TOP_K]

    top_results = []

    for passage_id in ranked_ids:
        index = document_ids.index(passage_id)
        top_results.append({
            "id": passage_id,
            "rrf_score": float(rrf_scores[passage_id]),
            "text": documents[index]
        })

    hits = len(set(ranked_ids) & relevant)
    recall = hits / len(relevant) if relevant else 0.0

    reciprocal_rank = 0.0
    for rank, passage_id in enumerate(ranked_ids, start=1):
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
    print("BM25 TOP 3:", [document_ids[i] for i in bm25_ranked[:3]])
    print("DENSE TOP 3:", [document_ids[i] for i in dense_ranked[:3]])
    print("HYBRID TOP 3:", ranked_ids)
    print(f"Recall@3: {recall:.4f}")
    print(f"RR: {reciprocal_rank:.4f}")

metrics = {
    "experiment": "Hybrid BM25 + multilingual E5 using RRF on section-aware v2 corpus",
    "model": MODEL_NAME,
    "fusion": "Reciprocal Rank Fusion",
    "rrf_k": RRF_K,
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
print("SECTION-AWARE V2 HYBRID RETRIEVAL RESULTS")
print("Total queries:", metrics["total_queries"])
print("Corpus passages:", metrics["corpus_passages"])
print(f"Recall@3: {metrics['recall_at_3']:.4f}")
print(f"MRR@3: {metrics['mrr_at_3']:.4f}")
print("Saved:", RESULTS_PATH)
print("Saved:", METRICS_PATH)
