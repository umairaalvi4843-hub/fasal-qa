import os
import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CORPUS_PATH = ROOT / "data" / "corpus_section_aware_v2.json"

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

TOP_K = 3
RRF_K = 60
MAX_RETRIES = 2
REQUEST_TIMEOUT = 20


TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
_translation_tokenizer = None
_translation_model = None


def translate_hindi_query(query):
    """Translate Devanagari Hindi into English for document retrieval."""
    global _translation_tokenizer, _translation_model

    if not re.search(r"[\u0900-\u097F]", query):
        return query

    if _translation_tokenizer is None:
        print("Loading Hindi-English translation model...", flush=True)
        _translation_tokenizer = AutoTokenizer.from_pretrained(
            TRANSLATION_MODEL,
            src_lang="hin_Deva"
        )
        _translation_model = AutoModelForSeq2SeqLM.from_pretrained(
            TRANSLATION_MODEL
        )
        _translation_model.eval()

    inputs = _translation_tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        output = _translation_model.generate(
            **inputs,
            forced_bos_token_id=(
                _translation_tokenizer.convert_tokens_to_ids("eng_Latn")
            ),
            max_new_tokens=128
        )

    translated = _translation_tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    translated = re.sub(
        r"\bthe tenant and the tenant farmer\b",
        "sharecroppers and tenant farmers",
        translated,
        flags=re.IGNORECASE
    )

    # Add official agricultural terminology where NLLB translations
    # do not reliably preserve the meaning of Hindi farmer categories.
    if "बटाईदार" in query:
        translated += " sharecropper sharecroppers"
    if "किरायेदार" in query:
        translated += " tenant farmer tenant farmers"

    print("Translated retrieval query:", translated, flush=True)
    return translated


# ==================================================
# GROQ CONFIGURATION
# ==================================================

api_key = os.getenv("GROQ_API_KEY")

if api_key:
    print(f"Groq configured with model: {GROQ_MODEL}", flush=True)
else:
    print("Groq key not found. Evidence-only mode enabled.", flush=True)


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

    search_queries = [query]

    translated_query = translate_hindi_query(query)
    if translated_query.strip().lower() != query.strip().lower():
        search_queries.append(translated_query)

    # Expand general PMFBY crop-loss and claim calculation questions
    # using terminology from the official yield-based claim formula.
    calculation_terms = (
        "crop loss",
        "crop losses",
        "claim calculation",
        "calculate claim",
        "insurance claim",
        "compensation",
        "payout",
        "indemnity",
        "how much claim",
        "loss calculation"
    )

    query_lower = query.lower()

    is_pmfby_calculation = (
        any(term in query_lower for term in calculation_terms)
        and any(
            term in query_lower
            for term in (
                "pmfby",
                "pradhan mantri fasal",
                "crop insurance",
                "crop loss",
                "crop losses"
            )
        )
    )

    if is_pmfby_calculation:
        search_queries.append(
            "PMFBY yield based claim formula "
            "Threshold Yield Actual Yield Sum Insured "
            "average yield best five years indemnity level"
        )

    rrf_scores = {}

    for search_query in search_queries:

        query_tokens = tokenize(search_query)

        bm25_scores = bm25.get_scores(query_tokens)

        bm25_ranking = np.argsort(bm25_scores)[::-1]

        query_embedding = embedding_model.encode(
            ["query: " + search_query],
            normalize_embeddings=True
        )[0]

        dense_scores = np.dot(
            document_embeddings,
            query_embedding
        )

        dense_ranking = np.argsort(dense_scores)[::-1]

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
# BASIC ANSWERABILITY CHECK
# ==================================================

def check_answerability(question, passages):

    question_lower = question.lower()

    # These topics require live or external information that
    # is not provided by the official scheme-document corpus.
    external_topics = (
        "today's mandi price",
        "today mandi price",
        "current mandi price",
        "today's market price",
        "current market price",
        "weather today",
        "today's weather",
        "live market price",
        "stock price"
    )

    if any(topic in question_lower for topic in external_topics):
        return False, (
            "This question requires current or external information "
            "that is not available in the official scheme documents."
        )

    retrieval_question = translate_hindi_query(question)

    question_tokens = {
        token for token in tokenize(retrieval_question)
        if len(token) > 2
    }

    # Ignore common words that do not indicate the topic.
    stop_words = {
        "what", "when", "where", "which", "who", "whom",
        "does", "do", "did", "can", "could", "would",
        "should", "how", "much", "many", "under", "about",
        "from", "with", "have", "has", "the", "and",
        "for", "are", "was", "were", "this", "that",
        "hai", "hain", "kya", "kaise", "kitna", "kitne",
        "mein", "ke", "ki", "ka", "ko", "par", "se"
    }

    question_tokens -= stop_words

    evidence_tokens = set()

    for passage in passages:
        evidence_tokens.update(tokenize(passage["text"]))

    overlap = question_tokens.intersection(evidence_tokens)

    # Conservative first-pass threshold. This is a heuristic,
    # not proof that the evidence is sufficient.
    if len(overlap) < 2:
        return False, (
            "The retrieved official passages do not contain enough "
            "matching information to answer this question reliably."
        )

    return True, "Relevant terms were found in retrieved evidence."


# ==================================================
# EVIDENCE-ONLY FALLBACK
# ==================================================

def evidence_only_answer(question, passages):

    question_lower = question.lower()

    concept_groups = {
        "amount": [
            "amount", "money", "benefit", "financial", "rupees",
            "receive", "payment", "income", "instalment", "installment"
        ],
        "eligibility": [
            "eligible", "eligibility", "farmer", "tenant",
            "sharecropper", "land", "qualify"
        ],
        "application": [
            "apply", "application", "register", "registration",
            "documents", "portal", "submit"
        ],
        "deadline": [
            "deadline", "date", "last date", "when", "period",
            "days", "months", "years"
        ],
        "claim": [
            "claim", "compensation", "loss", "insurance", "yield",
            "premium"
        ],
        "loan": [
            "loan", "credit", "kcc", "interest", "collateral",
            "repayment", "limit"
        ],
        "grievance": [
            "complaint", "grievance", "resolve",
            "resolution", "helpline", "ticket"
        ]
    }

    relevant_concepts = [
        concept for concept, words in concept_groups.items()
        if any(word in question_lower for word in words)
    ]

    question_words = set(
        re.findall(r"[a-z0-9]+", question_lower)
    )

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

        protected_text = re.sub(
            r"\bi\.e\.",
            "i.e<ABBREVIATION>",
            text,
            flags=re.IGNORECASE
        )

        sentences = re.split(
            r"(?<=[.!?])\s+",
            protected_text
        )

        sentences = [
            sentence.replace("<ABBREVIATION>", ".")
            for sentence in sentences
        ]

        for sentence in sentences:

            sentence = sentence.strip()

            if len(sentence) < 25:
                continue

            sentence_lower = sentence.lower()

            sentence_words = set(
                re.findall(r"[a-z0-9]+", sentence_lower)
            )

            score = len(question_words & sentence_words)

            for concept in relevant_concepts:
                if any(
                    word in sentence_lower
                    for word in concept_groups[concept]
                ):
                    score += 3

            has_amount = bool(
                re.search(
                    r"(?:₹|rs\.?\s*)\s*[\d,]+|"
                    r"[\d,]+\s*(?:per year|annually)",
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

    if "amount" in relevant_concepts:

        amount_candidates = [
            item for item in candidates
            if item["has_amount"]
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
        "Groq was unavailable. The following are the most relevant "
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
# GROQ ANSWER GENERATION
# ==================================================

def generate_with_groq(question, passages):

    current_key = os.getenv("GROQ_API_KEY")

    if not current_key:
        raise RuntimeError("GROQ_API_KEY is missing.")

    context = "\\n\\n".join(
        f"PASSAGE ID: {p['id']}\\nCONTENT: {p['text']}"
        for p in passages
    )

    system_prompt = """
You are Fasal QA, an assistant for Indian agricultural government schemes.

STRICT LANGUAGE RULE:
- If the farmer asks in English, answer entirely in English.
- If the farmer asks in Hindi, answer entirely in Hindi.
- If the farmer asks in Hinglish written in the Latin/Roman alphabet, answer in Hinglish using ONLY the Latin/Roman alphabet.
- For a Roman-script Hinglish question, do not write any Devanagari characters, even for Hindi words or headings.
- Keep scheme names, agricultural terms and explanations in Roman script when the question uses Roman-script Hinglish.
- If the farmer asks in Hindi written in Devanagari, answer in Hindi using Devanagari.
- If the farmer asks in English, answer in English.
- Match both the language and the writing script of the farmer's question.
- Determine the language and script from the farmer's question, NOT from the document passages.
- Never switch languages or scripts merely because a retrieved passage is written differently.
- Do not translate the final answer into a different language or script.

EVIDENCE RULES:
1. Use ONLY the official document passages supplied in the context.
2. Never invent facts or use outside knowledge.
3. If the supplied passages do not establish an answer, explicitly say that the available documents do not establish it.
4. IMPORTANT: The absence of a topic from the supplied passages is NOT proof that a scheme excludes, prohibits, or does not cover it.
5. Never turn missing information into a definite yes or no answer.
6. Only state that something is excluded, prohibited, or ineligible when a supplied passage explicitly supports that conclusion.
7. Distinguish confirmed facts from missing information.
8. If only part of a question is supported, answer that part and clearly identify what remains unestablished.
9. Use simple language suitable for farmers.
10. Cite every factual claim using the exact passage ID in square brackets, for example [PMKISAN_SA2_P18].
11. Use only passage IDs supplied in the context. Never invent or alter a passage ID.
12. Do not calculate compensation without all required inputs.
13. Do not confuse yield-based crop insurance claims with localized damage assessment.
14. Keep the answer concise and directly address the question.
"""

    user_prompt = f"""
OFFICIAL DOCUMENT PASSAGES:

{context}

FARMER QUESTION:

{question}

Answer the farmer in the same language as the question.
"""

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0,
        "max_completion_tokens": 1000
    }

    request = urllib.request.Request(
        GROQ_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {current_key}",
            "User-Agent": "FasalQA/1.0"
        },
        method="POST"
    )

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            print(
                f"Contacting Groq (attempt {attempt}/{MAX_RETRIES})...",
                flush=True
            )

            with urllib.request.urlopen(
                request,
                timeout=REQUEST_TIMEOUT
            ) as response:

                result = json.loads(
                    response.read().decode("utf-8")
                )

            answer = result["choices"][0]["message"]["content"]

            if answer and answer.strip():
                answer = answer.strip()

                # Normalize citations and reject IDs not present
                # in the retrieved passages.
                allowed_ids = {passage["id"] for passage in passages}

                citation_pattern = re.compile(
                    r"(?:【|\[)\s*"
                    r"((?:PMKISAN|PMFBY|KCC)_SA2_P\d+)"
                    r"(?:†L\d+(?:-L?\d+)?)?"
                    r"\s*(?:】|\])"
                )

                def normalize_citation(match):
                    passage_id = match.group(1)

                    if passage_id in allowed_ids:
                        return f"[{passage_id}]"

                    return ""

                answer = citation_pattern.sub(
                    normalize_citation,
                    answer
                )

                return answer.strip()

            raise RuntimeError("Groq returned an empty response.")

        except urllib.error.HTTPError as error:

            error_body = error.read().decode(
                "utf-8",
                errors="replace"
            )

            message = f"Groq HTTP {error.code}: {error_body}"

            print(message, flush=True)

            if error.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(message) from error

            if attempt == MAX_RETRIES:
                raise RuntimeError(message) from error

            # Groq may specify the wait time in its 429 response.
            if error.code == 429:
                wait_match = re.search(
                    r"try again in ([0-9.]+)s",
                    error_body,
                    flags=re.IGNORECASE
                )
                delay = (
                    float(wait_match.group(1)) + 1
                    if wait_match
                    else 6
                )
            else:
                delay = 2 ** (attempt - 1)

        except Exception as error:

            print(
                f"Groq request failed: {type(error).__name__}: {error}",
                flush=True
            )

            if attempt == MAX_RETRIES:
                raise

        delay = 2 ** (attempt - 1)

        print(f"Retrying in {delay} seconds...", flush=True)
        time.sleep(delay)

    raise RuntimeError("Groq unavailable after retries.")


# ==================================================
# COMPLETE RAG PIPELINE
# ==================================================

def generate_answer(question):

    print("\nSearching official documents...", flush=True)

    passages = retrieve_passages(question)

    print("Evidence retrieved.", flush=True)

    answerable, reason = check_answerability(question, passages)

    if not answerable:
        return {
            "question": question,
            "answer": reason,
            "mode": "Insufficient evidence",
            "retrieved_passages": [
                {
                    "id": p["id"],
                    "rrf_score": p["rrf_score"]
                }
                for p in passages
            ]
        }

    try:
        answer = generate_with_groq(question, passages)
        mode = "Groq-generated answer"

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
