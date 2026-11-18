import json
import re
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Load corpus and evaluation questions
with open("data/corpus.json", encoding="utf-8") as f:
    corpus = json.load(f)

with open("data/evaluation_queries.json", encoding="utf-8") as f:
    questions = json.load(f)

ids = [p["id"] for p in corpus]
texts = [p["text"] for p in corpus]

def tokenize(text):
    return re.findall(r"[\w]+", text.lower())

# BM25 setup
bm25 = BM25Okapi(
    [tokenize(text) for text in texts],
    k1=1.5,
    b=0.75
)

# Dense retrieval setup
print("Loading E5 model...")
model = SentenceTransformer(
    "intfloat/multilingual-e5-small",
    device="cpu"
)

print("Encoding corpus...")
embeddings = model.encode(
    ["passage: " + text for text in texts],
    batch_size=16,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

# Detect scheme from query
def detect_scheme(query):
    q = query.lower()

    if "kcc" in q or "kisan credit card" in q:
        return "KCC"

    if "pmfby" in q or "crop insurance" in q or "fasal bima" in q:
        return "PMFBY"

    if "pm-kisan" in q or "pmkisan" in q:
        return "PMKISAN"

    return None

# Evaluate a pair of weights
def evaluate(bm25_weight, dense_weight):
    recalls = []
    reciprocal_ranks = []
    details = []

    for item in questions:
        query = item["query"]
        relevant = set(item["relevant_passages"])
        scheme = detect_scheme(query)

        candidates = [
            i for i, pid in enumerate(ids)
            if scheme is None or pid.startswith(scheme + "_")
        ]

        bm25_scores = bm25.get_scores(tokenize(query))

        bm25_order = sorted(
            candidates,
            key=lambda i: bm25_scores[i],
            reverse=True
        )

        query_embedding = model.encode(
            "query: " + query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        dense_scores = np.dot(embeddings, query_embedding)

        dense_order = sorted(
            candidates,
            key=lambda i: dense_scores[i],
            reverse=True
        )

        bm25_rank = {
            idx: rank
            for rank, idx in enumerate(bm25_order, 1)
        }

        dense_rank = {
            idx: rank
            for rank, idx in enumerate(dense_order, 1)
        }

        # Weighted Reciprocal Rank Fusion
        scores = {}

        for idx in candidates:
            scores[idx] = (
                bm25_weight / (60 + bm25_rank[idx])
                +
                dense_weight / (60 + dense_rank[idx])
            )

        ranked = sorted(
            scores,
            key=scores.get,
            reverse=True
        )[:3]

        retrieved = [ids[i] for i in ranked]

        found = sum(pid in relevant for pid in retrieved)
        recall = found / len(relevant) if relevant else 0

        rr = 0
        for rank, pid in enumerate(retrieved, 1):
            if pid in relevant:
                rr = 1 / rank
                break

        recalls.append(recall)
        reciprocal_ranks.append(rr)

        details.append({
            "query": query,
            "scheme": scheme,
            "top_3": retrieved,
            "relevant_passages": list(relevant),
            "recall_at_3": recall,
            "reciprocal_rank": rr
        })

    return {
        "bm25_weight": bm25_weight,
        "dense_weight": dense_weight,
        "recall_at_3": float(np.mean(recalls)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "details": details
    }

# Compare three configurations
configurations = [
    (1, 1),
    (1, 2),
    (1, 3)
]

all_results = []

print("\nWEIGHTED FUSION COMPARISON")
print("=" * 65)

for bm25_weight, dense_weight in configurations:
    result = evaluate(bm25_weight, dense_weight)
    all_results.append(result)

    print(
        f"\nBM25 weight: {bm25_weight} | "
        f"E5 weight: {dense_weight}"
    )
    print(f"Recall@3: {result['recall_at_3']:.4f}")
    print(f"MRR:      {result['mrr']:.4f}")

    print("\nPer-query results:")

    for i, detail in enumerate(result["details"], 1):
        print(
            f"Q{i}: {detail['top_3']} "
            f"| Recall: {detail['recall_at_3']:.2f} "
            f"| RR: {detail['reciprocal_rank']:.2f}"
        )

# Save all configurations
with open(
    "data/weighted_fusion_results.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

print("\nSaved to data/weighted_fusion_results.json")
