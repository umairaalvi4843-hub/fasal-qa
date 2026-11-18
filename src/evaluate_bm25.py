import json
import re
from rank_bm25 import BM25Okapi

# Load the actual document corpus
with open("data/corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

# Load manually labelled evaluation questions
with open("data/evaluation_queries.json", "r", encoding="utf-8") as f:
    evaluation_data = json.load(f)

# Tokenization
def tokenize(text):
    return re.findall(r"[\w]+", text.lower())

# Prepare BM25
tokenized_corpus = [tokenize(p["text"]) for p in corpus]
bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

# Map passage IDs to corpus positions
passage_ids = [p["id"] for p in corpus]
id_to_index = {pid: i for i, pid in enumerate(passage_ids)}

recall_scores = []
reciprocal_ranks = []

print("=" * 80)
print("BM25 RETRIEVAL EVALUATION")
print("=" * 80)

for i, item in enumerate(evaluation_data, start=1):
    query = item["query"]
    relevant = set(item["relevant_passages"])

    # Check that all relevance labels exist in the corpus
    missing = relevant - set(id_to_index)
    if missing:
        raise ValueError(
            f"Query '{query}' contains missing passage IDs: {missing}"
        )

    # Retrieve top 3 passages
    scores = bm25.get_scores(tokenize(query))
    top_indices = sorted(
        range(len(scores)),
        key=lambda idx: scores[idx],
        reverse=True
    )[:3]

    top_results = [
        (passage_ids[idx], scores[idx]) for idx in top_indices
    ]

    retrieved_ids = [pid for pid, score in top_results]
    relevant_retrieved = relevant.intersection(retrieved_ids)

    # Recall@3
    recall = len(relevant_retrieved) / len(relevant)

    # Reciprocal Rank
    reciprocal_rank = 0.0
    for rank, pid in enumerate(retrieved_ids, start=1):
        if pid in relevant:
            reciprocal_rank = 1 / rank
            break

    recall_scores.append(recall)
    reciprocal_ranks.append(reciprocal_rank)

    print(f"\nQ{i}: {query}")
    print("Relevant passages:", ", ".join(sorted(relevant)))

    print("Top 3 BM25 results:")
    for rank, (pid, score) in enumerate(top_results, start=1):
        marker = "RELEVANT" if pid in relevant else "NOT RELEVANT"
        print(f"  {rank}. {pid} | Score: {score:.4f} | {marker}")

    print(f"Recall@3: {recall:.4f}")
    print(f"Reciprocal Rank: {reciprocal_rank:.4f}")

# Calculate overall metrics
mean_recall = sum(recall_scores) / len(recall_scores)
mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

print("\n" + "=" * 80)
print("FINAL EVALUATION RESULTS")
print("=" * 80)
print(f"Total evaluation queries: {len(evaluation_data)}")
print(f"Average Recall@3: {mean_recall:.4f}")
print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
print("=" * 80)

# Save results
results = {
    "total_queries": len(evaluation_data),
    "average_recall_at_3": mean_recall,
    "mean_reciprocal_rank": mrr
}

with open("data/bm25_evaluation_metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("\nMetrics saved to data/bm25_evaluation_metrics.json")
