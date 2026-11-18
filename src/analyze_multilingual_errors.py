import json

with open("data/multilingual_heldout_results.json", encoding="utf-8") as f:
    results = json.load(f)

failed = [
    item for item in results
    if item["recall_at_3"] < 1.0
]

for item in failed:
    print("\n" + "=" * 90)
    print("LANGUAGE:", item["language"])
    print("QUERY:", item["query"])
    print("GOLD PASSAGES:", item["relevant_passages"])
    print("RECALL@3:", item["recall_at_3"])
    print("MRR@3:", item["reciprocal_rank"])

    print("\nEXPECTED PASSAGE CONTENT:")

    for passage_id in item["relevant_passages"]:
        matching = next(
            (
                result for result in item["top_results"]
                if result["id"] == passage_id
            ),
            None
        )

        if matching:
            print(f"\n{passage_id} (retrieved):")
            print(matching["text"][:1200])
        else:
            print(f"\n{passage_id}: Not in top 3")

    print("\nACTUAL RETRIEVED PASSAGES:")

    for rank, passage in enumerate(item["top_results"], start=1):
        print(f"\nRank {rank}: {passage['id']}")
        print(passage["text"][:900])

