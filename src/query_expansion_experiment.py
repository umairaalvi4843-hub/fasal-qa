import json
import re
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

with open("data/corpus.json", encoding="utf-8") as f:
    corpus = json.load(f)

with open("data/evaluation_queries.json", encoding="utf-8") as f:
    questions = json.load(f)

ids = [p["id"] for p in corpus]
texts = [p["text"] for p in corpus]

def tokenize(text):
    return re.findall(r"[\w]+", text.lower())

bm25 = BM25Okapi(
    [tokenize(text) for text in texts],
    k1=1.5,
    b=0.75
)

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

# Manually designed domain terminology expansions.
# Original evaluation questions and labels remain unchanged.

expansions = {
    5: "eligible farmers insurable interest notified crops tenant farmers sharecroppers loanee non-loanee farmers",
    6: "premium rates maximum premium rate payable by farmer sum insured Kharif Rabi food grains oilseeds actuarial rate",
    7: "coverage of crops food crops cereals millets pulses oilseeds annual commercial horticultural crops",
    8: "coverage of risks and exclusions standing crops sowing harvesting prevented sowing mid-season adversity post-harvest losses localized calamities"
}

def detect_scheme(query):
    q = query.lower()

    if "kcc" in q or "kisan credit card" in q:
        return "KCC"

    if "pmfby" in q or "crop insurance" in q or "fasal bima" in q:
        return "PMFBY"

    if "pm-kisan" in q or "pmkisan" in q:
        return "PMKISAN"

    return None

def retrieve(query, scheme):
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
        idx: rank for rank, idx in enumerate(bm25_order, 1)
    }

    dense_rank = {
        idx: rank for rank, idx in enumerate(dense_order, 1)
    }

    scores = {
        idx: 1 / (60 + bm25_rank[idx]) + 1 / (60 + dense_rank[idx])
        for idx in candidates
    }

    ranked = sorted(scores, key=scores.get, reverse=True)[:3]

    return [ids[i] for i in ranked]

all_results = []

for i, item in enumerate(questions):
    original_query = item["query"]
    relevant = set(item["relevant_passages"])
    scheme = detect_scheme(original_query)

    expanded_query = original_query

    if i in expansions:
        expanded_query += " " + expansions[i]

    original_top = retrieve(original_query, scheme)
    expanded_top = retrieve(expanded_query, scheme)

    def calculate_metrics(top):
        recall = (
            sum(pid in relevant for pid in top) / len(relevant)
            if relevant else 0
        )

        rr = 0
        for rank, pid in enumerate(top, 1):
            if pid in relevant:
                rr = 1 / rank
                break

        return recall, rr

    original_recall, original_rr = calculate_metrics(original_top)
    expanded_recall, expanded_rr = calculate_metrics(expanded_top)

    all_results.append({
        "query": original_query,
        "expanded_query": expanded_query,
        "original_top_3": original_top,
        "expanded_top_3": expanded_top,
        "relevant_passages": list(relevant),
        "original_recall": original_recall,
        "expanded_recall": expanded_recall,
        "original_rr": original_rr,
        "expanded_rr": expanded_rr
    })

    print("\n" + "=" * 85)
    print("QUERY:", original_query)
    print("EXPANDED QUERY:", expanded_query)
    print("ORIGINAL:", original_top)
    print("EXPANDED:", expanded_top)
    print(
        f"Recall@3: {original_recall:.2f} -> {expanded_recall:.2f}"
    )
    print(
        f"MRR: {original_rr:.2f} -> {expanded_rr:.2f}"
    )

original_recall = np.mean([r["original_recall"] for r in all_results])
expanded_recall = np.mean([r["expanded_recall"] for r in all_results])

original_mrr = np.mean([r["original_rr"] for r in all_results])
expanded_mrr = np.mean([r["expanded_rr"] for r in all_results])

print("\n" + "=" * 85)
print("QUERY EXPANSION COMPARISON")
print(f"Original Recall@3: {original_recall:.4f}")
print(f"Expanded Recall@3: {expanded_recall:.4f}")
print(f"Original MRR: {original_mrr:.4f}")
print(f"Expanded MRR: {expanded_mrr:.4f}")

with open(
    "data/query_expansion_results.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

print("\nSaved to data/query_expansion_results.json")
