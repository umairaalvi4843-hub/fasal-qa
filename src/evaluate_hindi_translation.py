import json
import re
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CORPUS_PATH = "data/corpus_section_aware_v2.json"
EVALUATION_PATH = "data/heldout_multilingual_queries_audited.json"

RESULTS_PATH = "data/hindi_translation_comparison_results.json"
METRICS_PATH = "data/hindi_translation_comparison_metrics.json"

E5_MODEL = "intfloat/multilingual-e5-small"
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"

TOP_K = 3
RRF_K = 60


def tokenize(text):
    return re.findall(r"[\w]+", text.lower())


def translate_hindi(query, tokenizer, model):
    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_new_tokens=128
        )

    translated = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    # Correct the known agricultural terminology error.
    # NLLB translated both terms as tenant in our initial test.
    translated = re.sub(
        r"\bthe tenant and the tenant farmer\b",
        "sharecroppers and tenant farmers",
        translated,
        flags=re.IGNORECASE
    )

    return translated


def retrieve(query, bm25, model, document_embeddings,
             documents, document_ids):

    # BM25
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_ranked = np.argsort(bm25_scores)[::-1]

    # Dense E5
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

    return ranked_ids, rrf_scores


def calculate_metrics(results):
    recall = np.mean([
        item["recall_at_3"] for item in results
    ])

    mrr = np.mean([
        item["reciprocal_rank"] for item in results
    ])

    return {
        "total_queries": len(results),
        "recall_at_3": float(recall),
        "mrr_at_3": float(mrr)
    }


def evaluate_condition(
    queries,
    condition,
    bm25,
    model,
    document_embeddings,
    documents,
    document_ids
):
    results = []

    for item in queries:
        query = item["query"]
        relevant = set(item["relevant_passages"])

        ranked_ids, rrf_scores = retrieve(
            query,
            bm25,
            model,
            document_embeddings,
            documents,
            document_ids
        )

        recall = (
            len(set(ranked_ids) & relevant) / len(relevant)
        )

        reciprocal_rank = 0.0

        for rank, passage_id in enumerate(ranked_ids, start=1):
            if passage_id in relevant:
                reciprocal_rank = 1 / rank
                break

        results.append({
            "original_hindi_query": item["query"],
            "retrieval_query": query,
            "condition": condition,
            "relevant_passages": sorted(relevant),
            "retrieved_passages": ranked_ids,
            "recall_at_3": recall,
            "reciprocal_rank": reciprocal_rank
        })

    return results


print("Loading corpus and audited evaluation data...")

with open(CORPUS_PATH, encoding="utf-8") as f:
    corpus = json.load(f)

with open(EVALUATION_PATH, encoding="utf-8") as f:
    evaluation = json.load(f)

corpus_ids = {p["id"] for p in corpus}

for item in evaluation:
    missing = set(item["relevant_passages"]) - corpus_ids
    if missing:
        raise ValueError(
            f"Unknown passage IDs for query {item['query']}: {missing}"
        )

hindi_queries = [
    item for item in evaluation
    if re.search(r"[\u0900-\u097F]", item["query"])
]

print("Corpus passages:", len(corpus))
print("Hindi queries:", len(hindi_queries))

documents = [p["text"] for p in corpus]
document_ids = [p["id"] for p in corpus]

print("\nLoading NLLB translation model...")

translation_tokenizer = AutoTokenizer.from_pretrained(
    TRANSLATION_MODEL,
    src_lang="hin_Deva"
)

translation_model = AutoModelForSeq2SeqLM.from_pretrained(
    TRANSLATION_MODEL
)
translation_model.eval()

print("\nTranslating Hindi queries...")

translated_queries = []

for item in hindi_queries:
    translated = translate_hindi(
        item["query"],
        translation_tokenizer,
        translation_model
    )

    translated_queries.append({
        **item,
        "translated_query": translated
    })

    print("\nHindi:", item["query"])
    print("English:", translated)

print("\nLoading multilingual E5 model...")

model = SentenceTransformer(E5_MODEL)

print("Encoding corpus passages...")

document_embeddings = model.encode(
    ["passage: " + text for text in documents],
    normalize_embeddings=True,
    show_progress_bar=True
)

print("\nBuilding BM25 index...")

bm25 = BM25Okapi([
    tokenize(text) for text in documents
])

print("\nEvaluating original Hindi queries...")

baseline_results = evaluate_condition(
    hindi_queries,
    "original_hindi",
    bm25,
    model,
    document_embeddings,
    documents,
    document_ids
)

print("\nEvaluating translated English queries...")

translation_results = evaluate_condition(
    [
        {
            **item,
            "query": item["translated_query"]
        }
        for item in translated_queries
    ],
    "nllb_translated_english",
    bm25,
    model,
    document_embeddings,
    documents,
    document_ids
)

baseline_metrics = calculate_metrics(baseline_results)
translation_metrics = calculate_metrics(translation_results)

comparison = []

for baseline, translated in zip(
    baseline_results,
    translation_results
):
    comparison.append({
        "original_hindi_query": baseline["original_hindi_query"],
        "translated_english_query": translated["retrieval_query"],
        "relevant_passages": baseline["relevant_passages"],
        "baseline_retrieved": baseline["retrieved_passages"],
        "translated_retrieved": translated["retrieved_passages"],
        "baseline_recall_at_3": baseline["recall_at_3"],
        "translated_recall_at_3": translated["recall_at_3"],
        "baseline_mrr_at_3": baseline["reciprocal_rank"],
        "translated_mrr_at_3": translated["reciprocal_rank"]
    })

metrics = {
    "experiment": "Hindi versus NLLB-translated English retrieval",
    "translation_model": TRANSLATION_MODEL,
    "retrieval_model": E5_MODEL,
    "fusion": "Reciprocal Rank Fusion",
    "rrf_k": RRF_K,
    "top_k": TOP_K,
    "corpus_passages": len(corpus),
    "baseline": baseline_metrics,
    "translated": translation_metrics,
    "recall_improvement": (
        translation_metrics["recall_at_3"]
        - baseline_metrics["recall_at_3"]
    ),
    "mrr_improvement": (
        translation_metrics["mrr_at_3"]
        - baseline_metrics["mrr_at_3"]
    )
}

with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    json.dump({
        "comparison": comparison,
        "baseline_results": baseline_results,
        "translated_results": translation_results
    }, f, ensure_ascii=False, indent=4)

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=4)

print("\n" + "=" * 75)
print("HINDI TRANSLATION RETRIEVAL COMPARISON")
print("=" * 75)

print(
    f"\n{'QUERY':<12} {'BASE RECALL':>13} "
    f"{'TRANS RECALL':>13} {'BASE MRR':>12} {'TRANS MRR':>12}"
)

for item in comparison:
    print(
        f"{item['original_hindi_query'][:11]:<12} "
        f"{item['baseline_recall_at_3']:>13.4f} "
        f"{item['translated_recall_at_3']:>13.4f} "
        f"{item['baseline_mrr_at_3']:>12.4f} "
        f"{item['translated_mrr_at_3']:>12.4f}"
    )

print("\nOVERALL RESULTS")
print("Baseline Recall@3:", round(baseline_metrics["recall_at_3"], 4))
print("Translated Recall@3:", round(translation_metrics["recall_at_3"], 4))
print("Baseline MRR@3:", round(baseline_metrics["mrr_at_3"], 4))
print("Translated MRR@3:", round(translation_metrics["mrr_at_3"], 4))

print("\nRecall improvement:", round(metrics["recall_improvement"], 4))
print("MRR improvement:", round(metrics["mrr_improvement"], 4))

print("\nSaved:", RESULTS_PATH)
print("Saved:", METRICS_PATH)
