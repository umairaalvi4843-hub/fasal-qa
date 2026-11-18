import os
import re
import json
import time
import numpy as np

from pathlib import Path
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CORPUS_PATH = ROOT / "data" / "corpus_section_aware_v2.json"

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

TOP_K = 3
RRF_K = 60
MAX_RETRIES = 2


# ==================================================
# GEMINI CLIENT
# ==================================================

api_key = os.getenv("GEMINI_API_KEY")

client = None

if api_key:
    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=30000)
        )
        print("Gemini client configured.", flush=True)
    except Exception as error:
        print(f"Gemini initialization failed: {error}", flush=True)
else:
    print("Gemini key not found. Evidence-only mode enabled.", flush=True)


# ==================================================
# LOAD CORPUS
# ==================================================

print("\nLoading official scheme documents...", flush=True)

with open(CORPUS_PATH, "r", encoding="utf-8") as file:
    corpus = json.load(file)

documents = [item["text"] for item in corpus]
document_ids = [item["id"] for item in corpus]

print(f"Loaded {len(corpus)} passages.", flush=True)


# ==================================================
# INITIALIZE RETRIEVAL
# ==================================================

def tokenize(text):
    return re.findall(r"[\w]+", text.lower())


print("Initializing BM25...", flush=True)

bm25 = BM25Okapi([
    tokenize(document) for document in documents
])

print("Loading multilingual E5...", flush=True)

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Encoding document corpus once...", flush=True)

document_embeddings = embedding_model.encode(
    ["passage: " + document for document in documents],
    normalize_embeddings=True,
    show_progress_bar=True
)

print("\nFASAL QA READY.", flush=True)


# ==================================================
# HYBRID RETRIEVAL
# ==================================================

def retrieve_passages(query):

    query_tokens = tokenize(query)

    bm25_scores = bm25.get_scores(query_tokens)

    bm25_ranking = np.argsort(bm25_scores)[::-1]

    query_embedding = embedding_model.encode(
        ["query: " + query],
        normalize_embeddings=True
    )[0]

    dense_scores = np.dot(
        document_embeddings,
        query_embedding
    )

    dense_ranking = np.argsort(dense_scores)[::-1]

    rrf_scores = {}

    for rank, index in enumerate(bm25_ranking, start=1):
        rrf_scores[index] = rrf_scores.get(index, 0) + (
            1 / (RRF_K + rank)
        )

    for rank, index in enumerate(dense_ranking, start=1):
        rrf_scores[index] = rrf_scores.get(index, 0) + (
            1 / (RRF_K + rank)
        )

    ranking = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )[:TOP_K]

    return [
        {
            "id": document_ids[index],
            "text": documents[index],
            "rrf_score": round(rrf_scores[index], 6)
        }
        for index in ranking
    ]


# ==================================================
# EVIDENCE-ONLY FALLBACK
# ==================================================

def evidence_only_answer(question, passages):

    import re

    question_lower = question.lower()

    concept_groups = {
        "amount": ["amount", "money", "benefit", "financial", "rupees",
                   "receive", "payment", "income", "instalment", "installment"],
        "eligibility": ["eligible", "eligibility", "farmer", "tenant",
                        "sharecropper", "land", "qualify"],
        "application": ["apply", "application", "register", "registration",
                        "documents", "portal", "submit"],
        "deadline": ["deadline", "date", "last date", "when", "period",
                     "days", "months", "years"],
        "claim": ["claim", "compensation", "loss", "insurance", "yield",
                  "premium"],
        "loan": ["loan", "credit", "kcc", "interest", "collateral",
                 "repayment", "limit"],
        "grievance": ["complaint", "grievance", "resolve",
                      "resolution", "helpline", "ticket"]
    }

    relevant_concepts = [
        concept for concept, words in concept_groups.items()
        if any(word in question_lower for word in words)
    ]

    question_words = set(re.findall(r"[a-z0-9]+", question_lower))

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "do", "does",
        "did", "how", "what", "when", "where", "who", "which", "under",
        "for", "to", "of", "in", "on", "and", "or", "by", "with",
        "receive", "get", "every"
    }

    question_words -= stopwords
    candidates = []

    for passage in passages:
        text = passage["text"].strip()
        # Protect common abbreviations from being treated as sentence endings.
        protected_text = re.sub(
            r"\\bi\\.e\\.",
            "i.e<ABBREVIATION>",
            text,
            flags=re.IGNORECASE
        )
        sentences = re.split(r"(?<=[.!?])\\s+", protected_text)
        sentences = [
            sentence.replace("<ABBREVIATION>", ".")
            for sentence in sentences
        ]

        for sentence in sentences:
            sentence = sentence.strip()

            if len(sentence) < 25:
                continue

            sentence_lower = sentence.lower()
            sentence_words = set(re.findall(r"[a-z0-9]+", sentence_lower))

            score = len(question_words & sentence_words)

            for concept in relevant_concepts:
                if any(word in sentence_lower
                       for word in concept_groups[concept]):
                    score += 3

            has_amount = bool(
                re.search(
                    r"(?:₹|rs\.?\s*)\s*[\d,]+|[\d,]+\s*(?:per year|annually)",
                    sentence_lower
                )
            )

            if has_amount:
                score += 2

            candidates.append({
                "score": score,
                "sentence": sentence,
                "passage_id": passage["id"],
                "has_amount": has_amount
            })

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # For amount questions, prefer evidence that actually states an amount.
    if "amount" in relevant_concepts:
        amount_candidates = [
            item for item in candidates if item["has_amount"]
        ]
        if amount_candidates:
            candidates = amount_candidates

    selected = []
    seen = set()

    for item in candidates:
        sentence = item["sentence"]
        if sentence not in seen and item["score"] > 0:
            selected.append(item)
            seen.add(sentence)

        if len(selected) == 3:
            break

    if not selected:
        return (
            "I could not identify a sufficiently relevant statement "
            "in the retrieved passages. Please rephrase your question "
            "or consult the official scheme document."
        )

    parts = [
        "Gemini was unavailable. The following are the most relevant "
        "statements extracted directly from the official documents:"
    ]

    for item in selected:
        parts.append(
            f"\n\n[{item['passage_id']}]\n{item['sentence']}"
        )

    parts.append(
        "\n\nThese are direct document extracts, not a generated "
        "interpretation. Verify the official document before making "
        "a financial or eligibility decision."
    )

    return "".join(parts)


# ==================================================
# GEMINI ANSWER GENERATION
# ==================================================

def generate_with_gemini(question, passages):

    if client is None:
        raise RuntimeError("Gemini client is unavailable.")

    context = "\n\n".join(
        f"PASSAGE ID: {p['id']}\nCONTENT: {p['text']}"
        for p in passages
    )

    prompt = f"""
You are Fasal QA, a question-answering assistant for
Indian agricultural government schemes.

Answer using ONLY the official document passages below.

Rules:
1. Never invent facts or use outside knowledge.
2. If the evidence is insufficient, say so clearly.
3. Distinguish confirmed facts from missing information.
4. Use simple language suitable for farmers.
5. Respond in the same language as the question.
6. Respond naturally in Hinglish if the question is Hinglish.
7. Cite supporting passage IDs in square brackets.
8. Never cite a passage that does not support your claim.
9. Do not calculate compensation without all required inputs.

OFFICIAL DOCUMENT PASSAGES:

{context}

FARMER QUESTION:

{question}

Give a concise, accurate answer with citations.
"""

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            print(
                f"Contacting Gemini (attempt {attempt}/{MAX_RETRIES})...",
                flush=True
            )

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )

            if response.text:
                return response.text

            raise RuntimeError("Gemini returned an empty response.")

        except Exception as error:

            print(
                f"Gemini request failed: {type(error).__name__}: {error}",
                flush=True
            )

            if attempt < MAX_RETRIES:
                delay = 2 ** attempt
                print(f"Retrying in {delay} seconds...", flush=True)
                time.sleep(delay)

    raise RuntimeError("Gemini unavailable after retries.")


# ==================================================
# COMPLETE RAG PIPELINE
# ==================================================

def generate_answer(question):

    print("\nSearching official documents...", flush=True)

    passages = retrieve_passages(question)

    print("Evidence retrieved.", flush=True)

    try:
        answer = generate_with_gemini(question, passages)
        mode = "Gemini-generated answer"

    except Exception as error:
        print(
            f"\nUsing evidence-only fallback: {error}",
            flush=True
        )

        answer = evidence_only_answer(question, passages)
        mode = "Evidence-only fallback"

    return {
        "question": question,
        "answer": answer,
        "mode": mode,
        "retrieved_passages": [
            {
                "id": p["id"],
                "rrf_score": p["rrf_score"]
            }
            for p in passages
        ]
    }


# ==================================================
# INTERACTIVE APPLICATION
# ==================================================

if __name__ == "__main__":

    print("\n========================================")
    print("             FASAL QA")
    print("========================================")
    print("Ask about PM-KISAN, PMFBY or KCC.")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nYour question: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Thank you for using Fasal QA!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            result = generate_answer(question)

            print("\n" + "=" * 55)
            print("FASAL QA ANSWER")
            print("=" * 55)

            print(result["answer"])

            print("\nMODE:", result["mode"])

            print("\nRETRIEVED PASSAGES:")

            for passage in result["retrieved_passages"]:
                print(
                    f"- {passage['id']} "
                    f"(RRF: {passage['rrf_score']})"
                )

            print("=" * 55)

        except Exception as error:
            print(f"\nApplication error: {error}")
