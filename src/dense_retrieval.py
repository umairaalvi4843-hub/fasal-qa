import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load corpus and evaluation questions
with open("data/corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

with open("data/evaluation_queries.json", "r", encoding="utf-8") as f:
    evaluation_data = json.load(f)

print(f"Loaded passages: {len(corpus)}")
print(f"Evaluation questions: {len(evaluation_data)}")

# Load multilingual E5 model
model_name = "intfloat/multilingual-e5-small"

print("\nLoading multilingual E5 model...")
model = SentenceTransformer(model_name, device="cpu")

# E5 requires passage: prefix for documents
passages = ["passage: " + item["text"] for item in corpus]

print("\nGenerating passage embeddings...")
passage_embeddings = model.encode(
    passages,
    batch_size=16,
    show_progress_bar=True,
    normalize_embeddings=True
)

# Prepare passage IDs
passage_ids = [item["id"] for item in corpus]
corpus_id_set = set(passage_ids)

all_recall = []
all_reciprocal_ranks = []
all_results = []

print("\n" + "=" * 80)
print("MULTILINGUAL DENSE RETRIEVAL EVALUATION")
print("=" * 80)

for i, item in enumerate(evaluation_data, start=1):
    query = item["query"]
    relevant = set(item["relevant_passages"])

    missing = relevant - corpus_id_set
    if missing:
        raise ValueError(
            f"Query '{query}' contains missing passage IDs: {missing}"
        )

    # E5 requires query: prefix for questions
    query_embedding = model.encode(
        ["query: " + query],
        normalize_embeddings=True
    )[0]

    # Cosine similarity (embeddings are normalized)
    scores = np.dot(passage_embeddings, query_embedding)

    top_indices = np.argsort(scores)[::-1][:3]

    top_results = [
        {
            "id": passage_ids[idx],
            "score": float(scores[idx]),
            "relevant": passage_ids[idx] in relevant
        }
        for idx in top_indices
    ]

    retrieved_ids = [result["id"] for result in top_results]
    relevant_retrieved = relevant.intersection(retrieved_ids)

    recall = len(relevant_retrieved) / len(relevant)

    reciprocal_rank = 0.0
    for rank, pid in enumerate(retrieved_ids, start=1):
        if pid in relevant:
            reciprocal_rank = 1 / rank
            break

    all_recall.append(recall)
    all_reciprocal_ranks.append(reciprocal_rank)

    print(f"\nQ{i}: {query}")
    print("Relevant passages:", ", ".join(sorted(relevant)))
    print("Top 3 dense retrieval results:")

    for rank, result in enumerate(top_results, start=1):
        marker = "RELEVANT" if result["relevant"] else "NOT RELEVANT"
        print(
            f"  {rank}. {result['id']} | "
            f"Score: {result['score']:.4f} | {marker}"
        )

    print(f"Recall@3: {recall:.4f}")
    print(f"Reciprocal Rank: {reciprocal_rank:.4f}")

    all_results.append({
        "query": query,
        "relevant_passages": sorted(relevant),
        "top_3_results": top_results,
        "recall_at_3": recall,
        "reciprocal_rank": reciprocal_rank
    })

# Calculate overall metrics
mean_recall = float(np.mean(all_recall))
mrr = float(np.mean(all_reciprocal_ranks))

print("\n" + "=" * 80)
print("FINAL DENSE RETRIEVAL RESULTS")
print("=" * 80)
print(f"Total evaluation queries: {len(evaluation_data)}")
print(f"Average Recall@3: {mean_recall:.4f}")
print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
print("=" * 80)

# Save detailed results
with open("data/dense_retrieval_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

# Save evaluation metrics
metrics = {
    "model": model_name,
    "total_queries": len(evaluation_data),
    "total_passages": len(corpus),
    "average_recall_at_3": mean_recall,
    "mean_reciprocal_rank": mrr
}

with open("data/dense_evaluation_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print("\nDetailed results saved to data/dense_retrieval_results.json")
print("Metrics saved to data/dense_evaluation_metrics.json")
