import json
from pathlib import Path

evaluation_path = Path("data/evaluation_queries_section_aware_v2_audited.json")

methods = {
    "BM25": "data/bm25_section_aware_v2_results.json",
    "BM25 + Scheme Filtering": "data/bm25_filtered_section_aware_v2_results.json",
    "Dense E5": "data/dense_section_aware_v2_results.json",
    "Hybrid": "data/hybrid_section_aware_v2_results.json",
    "Hybrid + Scheme Filtering": "data/hybrid_filtered_section_aware_v2_results.json"
}

with evaluation_path.open(encoding="utf-8") as f:
    evaluation = json.load(f)

gold = {item["query"]: set(item["relevant_passages"]) for item in evaluation}

summary = {}

for method, filename in methods.items():
    with open(filename, encoding="utf-8") as f:
        old_results = json.load(f)

    results_by_query = {item["query"]: item for item in old_results}

    recalls = []
    reciprocal_ranks = []
    audited_results = []

    for query, relevant in gold.items():
        item = results_by_query[query]
        top = item.get("top_results", [])[:3]
        retrieved = [p["id"] for p in top]

        recall = len(set(retrieved) & relevant) / len(relevant)

        rr = 0.0
        for rank, passage_id in enumerate(retrieved, start=1):
            if passage_id in relevant:
                rr = 1 / rank
                break

        recalls.append(recall)
        reciprocal_ranks.append(rr)

        updated = dict(item)
        updated["relevant_passages"] = sorted(relevant)
        updated["recall_at_3"] = recall
        updated["reciprocal_rank"] = rr
        audited_results.append(updated)

    safe_name = method.lower().replace(" + ", "_").replace(" ", "_")

    with open(f"data/{safe_name}_audited_v2_results.json", "w", encoding="utf-8") as f:
        json.dump(audited_results, f, ensure_ascii=False, indent=4)

    metrics = {
        "method": method,
        "total_queries": len(evaluation),
        "recall_at_3": sum(recalls) / len(recalls),
        "mrr_at_3": sum(reciprocal_ranks) / len(reciprocal_ranks)
    }

    with open(f"data/{safe_name}_audited_v2_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    summary[method] = metrics

print("\nAUDITED RETRIEVAL EVALUATION")
print("=" * 65)
print(f"{'METHOD':<32} {'RECALL@3':>12} {'MRR@3':>12}")
print("-" * 65)

for method, metrics in summary.items():
    print(f"{method:<32} {metrics['recall_at_3']:>12.4f} {metrics['mrr_at_3']:>12.4f}")

with open("data/audited_v2_comparison.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=4)

print("\nAll audited results and metrics saved successfully.")
