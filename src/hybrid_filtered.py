import json
import re
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Load corpus and evaluation questions
# --------------------------------------------------

with open("data/corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

with open("data/evaluation_queries.json", "r", encoding="utf-8") as f:
    evaluation_queries = json.load(f)

passages = [item["text"] for item in corpus]
passage_ids = [item["id"] for item in corpus]

print("Loaded passages:", len(corpus))


# --------------------------------------------------
# 2. Prepare BM25
# --------------------------------------------------

def tokenize(text):
    return re.findall(r"[\w]+", text.lower())


tokenized_corpus = [tokenize(text) for text in passages]

bm25 = BM25Okapi(
    tokenized_corpus,
    k1=1.5,
    b=0.75
)


# --------------------------------------------------
# 3. Generate multilingual E5 embeddings
# --------------------------------------------------

print("\nLoading multilingual E5 model...")

model = SentenceTransformer(
    "intfloat/multilingual-e5-small",
    device="cpu"
)

print("\nGenerating passage embeddings...")

passage_embeddings = model.encode(
    ["passage: " + text for text in passages],
    batch_size=16,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)


# --------------------------------------------------
# 4. Detect the scheme from the question
# --------------------------------------------------

def detect_scheme(query):
    q = query.lower()

    if "kcc" in q or "kisan credit card" in q:
        return "KCC"

    if (
        "pmfby" in q
        or "crop insurance" in q
        or "fasal bima" in q
    ):
        return "PMFBY"

    if "pm-kisan" in q or "pmkisan" in q:
        return "PMKISAN"

    return None


# --------------------------------------------------
# 5. Filtered hybrid retrieval using RRF
# --------------------------------------------------

RRF_K = 60
TOP_K = 3


def hybrid_search(query):

    scheme = detect_scheme(query)

    if scheme:
        candidate_indices = [
            i for i, pid in enumerate(passage_ids)
            if pid.startswith(scheme + "_")
        ]
    else:
        candidate_indices = list(range(len(passages)))

    # BM25 ranking within the selected scheme
    bm25_scores = bm25.get_scores(tokenize(query))

    bm25_ranking = sorted(
        candidate_indices,
        key=lambda i: bm25_scores[i],
        reverse=True
    )

    bm25_ranks = {
        index: rank
        for rank, index in enumerate(bm25_ranking, start=1)
    }

    # Dense ranking within the selected scheme
    query_embedding = model.encode(
        "query: " + query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dense_scores = np.dot(
        passage_embeddings,
        query_embedding
    )

    dense_ranking = sorted(
        candidate_indices,
        key=lambda i: dense_scores[i],
        reverse=True
    )

    dense_ranks = {
        index: rank
        for rank, index in enumerate(dense_ranking, start=1)
    }

    # Reciprocal Rank Fusion
    rrf_scores = {}

    for index in candidate_indices:
        score = (
            1 / (RRF_K + bm25_ranks[index])
            +
            1 / (RRF_K + dense_ranks[index])
        )

        rrf_scores[index] = score

    ranked_indices = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )

    return [
        {
            "id": passage_ids[index],
            "rrf_score": round(rrf_scores[index], 6),
            "bm25_rank": bm25_ranks[index],
            "dense_rank": dense_ranks[index],
            "scheme_filter": scheme
        }
        for index in ranked_indices[:TOP_K]
    ]


# --------------------------------------------------
# 6. Evaluate filtered hybrid retrieval
# --------------------------------------------------

all_results = []
recall_scores = []
reciprocal_ranks = []

print("\n" + "=" * 80)
print("SCHEME-FILTERED HYBRID RETRIEVAL EVALUATION")
print("=" * 80)

for q_num, item in enumerate(evaluation_queries, start=1):

    query = item["query"]
    relevant = set(item["relevant_passages"])

    results = hybrid_search(query)

    retrieved_ids = [r["id"] for r in results]

    relevant_found = [
        pid for pid in retrieved_ids if pid in relevant
    ]

    recall = (
        len(relevant_found) / len(relevant)
        if relevant else 0
    )

    reciprocal_rank = 0

    for rank, pid in enumerate(retrieved_ids, start=1):
        if pid in relevant:
            reciprocal_rank = 1 / rank
            break

    recall_scores.append(recall)
    reciprocal_ranks.append(reciprocal_rank)

    for result in results:
        result["relevant"] = result["id"] in relevant

    all_results.append({
        "query": query,
        "scheme_filter": detect_scheme(query),
        "relevant_passages": list(relevant),
        "top_3_results": results,
        "recall_at_3": recall,
        "reciprocal_rank": reciprocal_rank
    })

    print(f"\nQ{q_num}: {query}")
    print("Scheme filter:", detect_scheme(query))
    print("Top 3 results:")

    for rank, result in enumerate(results, start=1):
        status = "RELEVANT" if result["relevant"] else "NOT RELEVANT"

        print(
            f"  {rank}. {result['id']} "
            f"| RRF: {result['rrf_score']} "
            f"| BM25 rank: {result['bm25_rank']} "
            f"| E5 rank: {result['dense_rank']} "
            f"| {status}"
        )

    print(f"Recall@3: {recall:.4f}")
    print(f"Reciprocal Rank: {reciprocal_rank:.4f}")


# --------------------------------------------------
# 7. Save metrics and results
# --------------------------------------------------

metrics = {
    "total_queries": len(evaluation_queries),
    "average_recall_at_3": float(np.mean(recall_scores)),
    "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
    "rrf_constant": RRF_K,
    "scheme_filtering": True
}

print("\n" + "=" * 80)
print("FINAL SCHEME-FILTERED HYBRID RESULTS")
print("=" * 80)

print("Total queries:", len(evaluation_queries))
print(f"Average Recall@3: {metrics['average_recall_at_3']:.4f}")
print(f"Mean Reciprocal Rank (MRR): {metrics['mean_reciprocal_rank']:.4f}")

with open(
    "data/hybrid_filtered_results.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

with open(
    "data/hybrid_filtered_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

print("\nResults saved to data/hybrid_filtered_results.json")
print("Metrics saved to data/hybrid_filtered_metrics.json")
