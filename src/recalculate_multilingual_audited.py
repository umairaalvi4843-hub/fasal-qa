import json
import numpy as np

with open("data/heldout_multilingual_queries.json", encoding="utf-8") as f:
    evaluation = json.load(f)

with open("data/multilingual_heldout_results.json", encoding="utf-8") as f:
    results = json.load(f)

# P18 explicitly states the annual benefit and its three instalments.
for item in evaluation:
    if "PM-KISAN" in item["query"] or "PM Kisan" in item["query"]:
        item["relevant_passages"] = [
            "PMKISAN_SA2_P1",
            "PMKISAN_SA2_P18"
        ]

# Recalculate metrics using the existing retrieval rankings.
audited_results = []

for result in results:
    relevant = set(
        next(
            item["relevant_passages"]
            for item in evaluation
            if item["query"] == result["query"]
        )
    )

    retrieved = [p["id"] for p in result["top_results"]]

    recall = len(set(retrieved) & relevant) / len(relevant)

    reciprocal_rank = 0.0
    for rank, passage_id in enumerate(retrieved, start=1):
        if passage_id in relevant:
            reciprocal_rank = 1 / rank
            break

    updated = dict(result)
    updated["relevant_passages"] = sorted(relevant)
    updated["recall_at_3"] = recall
    updated["reciprocal_rank"] = reciprocal_rank

    audited_results.append(updated)

language_metrics = {}

for language in sorted(set(r["language"] for r in audited_results)):
    subset = [r for r in audited_results if r["language"] == language]

    language_metrics[language] = {
        "total_queries": len(subset),
        "recall_at_3": float(np.mean([r["recall_at_3"] for r in subset])),
        "mrr_at_3": float(np.mean([r["reciprocal_rank"] for r in subset]))
    }

metrics = {
    "experiment": "Audited held-out multilingual hybrid retrieval",
    "total_queries": len(audited_results),
    "recall_at_3": float(np.mean([r["recall_at_3"] for r in audited_results])),
    "mrr_at_3": float(np.mean([r["reciprocal_rank"] for r in audited_results])),
    "language_metrics": language_metrics,
    "label_change": "Added PMKISAN_SA2_P18 as relevant to both PM-KISAN annual assistance queries."
}

with open("data/heldout_multilingual_queries_audited.json", "w", encoding="utf-8") as f:
    json.dump(evaluation, f, ensure_ascii=False, indent=4)

with open("data/multilingual_heldout_audited_results.json", "w", encoding="utf-8") as f:
    json.dump(audited_results, f, ensure_ascii=False, indent=4)

with open("data/multilingual_heldout_audited_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=4)

print("\nAUDITED MULTILINGUAL RESULTS")
print("-" * 55)
print(f"{'LANGUAGE':<15} {'QUERIES':>10} {'RECALL@3':>12} {'MRR@3':>12}")

for language, values in language_metrics.items():
    print(
        f"{language:<15} {values['total_queries']:>10} "
        f"{values['recall_at_3']:>12.4f} {values['mrr_at_3']:>12.4f}"
    )

print("-" * 55)
print(
    f"{'OVERALL':<15} {len(audited_results):>10} "
    f"{metrics['recall_at_3']:>12.4f} {metrics['mrr_at_3']:>12.4f}"
)

print("\nSaved audited evaluation files.")
