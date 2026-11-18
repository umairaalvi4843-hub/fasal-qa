import json

SOURCE_PATH = "data/evaluation_queries_section_aware_v2.json"
OUTPUT_PATH = "data/evaluation_queries_section_aware_v2_audited.json"
CORPUS_PATH = "data/corpus_section_aware_v2.json"

with open(SOURCE_PATH, encoding="utf-8") as f:
    evaluation = json.load(f)

with open(CORPUS_PATH, encoding="utf-8") as f:
    corpus = json.load(f)

additions = {
    "Who is eligible for PMFBY crop insurance?": [
        "PMFBY_SA2_P389"
    ],
    "What types of crop damage are covered by PMFBY?": [
        "PMFBY_SA2_P267",
        "PMFBY_SA2_P309",
        "PMFBY_SA2_P437"
    ],
    "How is crop insurance compensation calculated?": [
        "PMFBY_SA2_P234"
    ],
    "How can a farmer apply for PMFBY?": [
        "PMFBY_SA2_P168",
        "PMFBY_SA2_P164"
    ],
    "How can farmers file complaints about PMFBY?": [
        "PMFBY_SA2_P58",
        "PMFBY_SA2_P60",
        "PMFBY_SA2_P351",
        "PMFBY_SA2_P360"
    ]
}

corpus_ids = {p["id"] for p in corpus}

for item in evaluation:
    query = item["query"]

    if query in additions:
        item["relevant_passages"] = sorted(
            set(item["relevant_passages"]) | set(additions[query])
        )

    missing = set(item["relevant_passages"]) - corpus_ids

    if missing:
        raise ValueError(
            f"Missing passage IDs for query '{query}': {sorted(missing)}"
        )

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(evaluation, f, ensure_ascii=False, indent=4)

print("AUDITED EVALUATION FILE CREATED")
print("Queries:", len(evaluation))
print("Original file preserved:", SOURCE_PATH)
print("New file:", OUTPUT_PATH)

print("\nUPDATED RELEVANCE LABELS")
for item in evaluation:
    print("\nQUERY:", item["query"])
    print("RELEVANT PASSAGES:", item["relevant_passages"])
    print("LABEL COUNT:", len(item["relevant_passages"]))
