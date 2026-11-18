import json
import re
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CORPUS_PATH = "data/corpus_section_aware_v2.json"
EVALUATION_PATH = "data/heldout_multilingual_queries.json"

RESULTS_PATH = "data/multilingual_heldout_results.json"
METRICS_PATH = "data/multilingual_heldout_metrics.json"

MODEL_NAME = "intfloat/multilingual-e5-small"
TOP_K = 3
RRF_K = 60


def tokenize(text):
    return re.findall(r"[\w]+", text.lower())


with open(CORPUS_PATH, encoding="utf-8") as f:
    corpus = json.load(f)

with open(EVALUATION_PATH, encoding="utf-8") as f:
    evaluation = json.load(f)

corpus_ids = {p["id"] for p in corpus}

# Validate all relevance labels before evaluating.
for item in evaluation:
    missing = set(item["relevant_passages"]) - corpus_ids
    if missing:
        raise ValueError(
            f"Unknown passage IDs for query '{item['query']}': {missing}"
        )

documents = [p["text"] for p in corpus]
document_ids = [p["id"] for p in corpus]

print("Corpus passages:", len(corpus))
print("Evaluation queries:", len(evaluation))
print("Loading multilingual E5 model...")

model = SentenceTransformer(MODEL_NAME)

print("Encoding corpus passages...")

document_embeddings = model.encode(
    ["passage: " + text for text in documents],
    normalize_embeddings=True,
    show_progress_bar=True
)

bm25 = BM25Okapi([tokenize(text) for text in documents])

all_results = []

for item in evaluation:
    query = item["query"]
    language = item["language"]
    relevant = set(item["relevant_passages"])

    # BM25 ranking
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_ranked = np.argsort(bm25_scores)[::-1]

    # Multilingual E5 ranking
    query_embedding = model.encode(
        ["query: " + query],
        normalize_embeddings=True,
        show_progress_bar=False
    )[0]

    dense_scores = np.dot(document_embeddings, query_embedding)
    dense_ranked = np.argsort(dense_scores)[::-1]

    # Reciprocal Rank Fusion
    rrf_scores = {}

    for rank, index in enumerate(bm25_ranked, start=1):
        passage_id = document_ids[index]
        rrf_scores[passage_id] = (
            rrf_scores.get(passage_id, 0.0)
            + 1 / (RRF_K + rank)
        )

    for rank, index in enumerate(dense_ranked, start=1):
        passage_id = document_ids[index]
        rrf_scores[passage_id] = (
            rrf_scores.get(passage_id, 0.0)
            + 1 / (RRF_K + rank)
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

    recall = len(set(ranked_ids) & relevant) / len(relevant)

    reciprocal_rank = 0.0
    for rank, passage_id in enumerate(ranked_ids, start=1):
        if passage_id in relevant:
            reciprocal_rank = 1 / rank
            break

    all_results.append({
        "query": query,
        "language": language,
        "relevant_passages": sorted(relevant),
        "top_results": top_results,
        "recall_at_3": recall,
        "reciprocal_rank": reciprocal_rank
    })

    print("\n" + "=" * 70)
    print("LANGUAGE:", language)
    print("QUERY:", query)
    print("RELEVANT:", sorted(relevant))
    print("RETRIEVED:", ranked_ids)
    print(f"Recall@3: {recall:.4f}")
    print(f"MRR@3: {reciprocal_rank:.4f}")


# Overall metrics
overall_recall = np.mean([
    item["recall_at_3"] for item in all_results
])

overall_mrr = np.mean([
    item["reciprocal_rank"] for item in all_results
])

# Language-wise metrics
language_metrics = {}

for language in sorted(set(item["language"] for item in evaluation)):
    subset = [
        item for item in all_results
        if item["language"] == language
    ]

    language_metrics[language] = {
        "total_queries": len(subset),
        "recall_at_3": float(np.mean([
            item["recall_at_3"] for item in subset
        ])),
        "mrr_at_3": float(np.mean([
            item["reciprocal_rank"] for item in subset
        ]))
    }

metrics = {
    "experiment": "Held-out multilingual hybrid retrieval",
    "model": MODEL_NAME,
    "fusion": "Reciprocal Rank Fusion",
    "rrf_k": RRF_K,
    "corpus_passages": len(corpus),
    "total_queries": len(evaluation),
    "recall_at_3": float(overall_recall),
    "mrr_at_3": float(overall_mrr),
    "language_metrics": language_metrics
}

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=4)

print("\n" + "=" * 70)
print("HELD-OUT MULTILINGUAL EVALUATION")
print("=" * 70)

print(f"\n{'LANGUAGE':<15} {'QUERIES':>10} {'RECALL@3':>12} {'MRR@3':>12}")
print("-" * 55)

for language, values in language_metrics.items():
    print(
        f"{language:<15} "
        f"{values['total_queries']:>10} "
        f"{values['recall_at_3']:>12.4f} "
        f"{values['mrr_at_3']:>12.4f}"
    )

print("-" * 55)
print(
    f"{'OVERALL':<15} "
    f"{len(evaluation):>10} "
    f"{overall_recall:>12.4f} "
    f"{overall_mrr:>12.4f}"
)

print("\nSaved:", RESULTS_PATH)
print("Saved:", METRICS_PATH)
