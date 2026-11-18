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
print("Loaded evaluation queries:", len(evaluation_queries))


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
# 3. Generate E5 embeddings
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
# 4. Reciprocal Rank Fusion
# --------------------------------------------------

RRF_K = 60
TOP_K = 3


def hybrid_search(query):

    # BM25 ranking
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_ranking = np.argsort(bm25_scores)[::-1]

    bm25_ranks = {
        int(index): rank
        for rank, index in enumerate(bm25_ranking, start=1)
    }

    # Dense E5 ranking
    query_embedding = model.encode(
        "query: " + query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dense_scores = np.dot(
        passage_embeddings,
        query_embedding
    )

    dense_ranking = np.argsort(dense_scores)[::-1]

    dense_ranks = {
        int(index): rank
        for rank, index in enumerate(dense_ranking, start=1)
    }

    # Combine rankings using RRF
    rrf_scores = {}

    for index in range(len(passages)):

        bm25_rank = bm25_ranks[index]
        dense_rank = dense_ranks[index]

        score = (
            1 / (RRF_K + bm25_rank)
            +
            1 / (RRF_K + dense_rank)
        )

        rrf_scores[index] = score

    # Sort by combined RRF score
    ranked_indices = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )

    results = []

    for index in ranked_indices[:TOP_K]:
        results.append({
            "id": passage_ids[index],
            "rrf_score": round(rrf_scores[index], 6),
            "bm25_rank": bm25_ranks[index],
            "dense_rank": dense_ranks[index]
        })

    return results


# --------------------------------------------------
# 5. Evaluate hybrid retrieval
# --------------------------------------------------

all_results = []
recall_scores = []
reciprocal_ranks = []

print("\n" + "=" * 80)
print("HYBRID RETRIEVAL EVALUATION (BM25 + E5 + RRF)")
print("=" * 80)

for q_num, item in enumerate(evaluation_queries, start=1):

    query = item["query"]
    relevant = set(item["relevant_passages"])

    results = hybrid_search(query)

    retrieved_ids = [result["id"] for result in results]

    relevant_found = [
        pid for pid in retrieved_ids if pid in relevant
    ]

    recall = len(relevant_found) / len(relevant) if relevant else 0

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
        "relevant_passages": list(relevant),
        "top_3_results": results,
        "recall_at_3": recall,
        "reciprocal_rank": reciprocal_rank
    })

    print(f"\nQ{q_num}: {query}")
    print("Relevant passages:", ", ".join(sorted(relevant)))
    print("Top 3 hybrid results:")

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
# 6. Calculate final metrics
# --------------------------------------------------

average_recall = float(np.mean(recall_scores))
mrr = float(np.mean(reciprocal_ranks))

metrics = {
    "total_queries": len(evaluation_queries),
    "average_recall_at_3": average_recall,
    "mean_reciprocal_rank": mrr,
    "rrf_constant": RRF_K
}

print("\n" + "=" * 80)
print("FINAL HYBRID RETRIEVAL RESULTS")
print("=" * 80)

print("Total evaluation queries:", len(evaluation_queries))
print(f"Average Recall@3: {average_recall:.4f}")
print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")

with open(
    "data/hybrid_retrieval_results.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

with open(
    "data/hybrid_evaluation_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

print("\nDetailed results saved to data/hybrid_retrieval_results.json")
print("Metrics saved to data/hybrid_evaluation_metrics.json")
