import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag_answer import generate_answer

TEST_PATH = ROOT / "data" / "answer_quality_tests.json"
RESULT_PATH = ROOT / "data" / "answer_quality_results.json"


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(",", "")
    text = text.replace("₹", " ")
    text = re.sub(r"[^\w]+", " ", text.lower())
    return " ".join(text.split())


def contains_phrase(answer, phrase):
    normalized_phrase = normalize(phrase)

    # For symbols such as ₹, use literal matching.
    if not normalized_phrase:
        return phrase in answer

    return normalized_phrase in normalize(answer)



def language_check(answer, expected_language):
    if not expected_language:
        return True

    devanagari_count = len(
        re.findall(r"[\u0900-\u097F]", answer)
    )

    if expected_language == "hindi":
        return devanagari_count >= 5

    if expected_language == "hinglish":
        if devanagari_count > 2:
            return False

        markers = [
            "ke tahat", "har saal", "ke liye",
            "kar sakte", "kisan", "hai", "hain",
            "milta", "milti", "apply"
        ]
        lowered = answer.lower()
        return any(
            re.search(r"\b" + re.escape(marker) + r"\b", lowered)
            for marker in markers
        )

    return True


tests = json.loads(TEST_PATH.read_text(encoding="utf-8"))
results = []

print("\nFASAL QA ANSWER QUALITY EVALUATION")
print("=" * 60)

for test in tests:
    print(f"\nRunning {test['id']}: {test['query']}")

    try:
        output = generate_answer(test["query"])
        answer = output["answer"]
        normalized_answer = normalize(answer)

        retrieved_ids = {
            item["id"] for item in output["retrieved_passages"]
        }

        citations = re.findall(
            r"\[((?:PMKISAN|PMFBY|KCC)_SA2_P\d+)\]",
            answer
        )

        checks = {
            "mode": output["mode"] == test["expected_mode"],
            "language": language_check(
                answer, test.get("expected_language")
            ),
            "required_facts": (
                any(
                    contains_phrase(answer, phrase)
                    for phrase in test["required_facts"]
                )
                if test.get("required_fact_mode") == "any"
                else all(
                    contains_phrase(answer, phrase)
                    for phrase in test["required_facts"]
                )
            ),
            "forbidden_claims": not any(
                contains_phrase(answer, phrase)
                for phrase in test["forbidden_claims"]
            ),
            "citation_format": (
                "【" not in answer and "】" not in answer
            ),
            "citation_grounding": all(
                citation in retrieved_ids
                for citation in citations
            ),
            "expected_citation": (
                all(
                    citation in citations
                    for citation in test["expected_citations"]
                )
                if test["expected_citations"]
                else True
            )
        }

        passed = all(checks.values())

        result = {
            "id": test["id"],
            "query": test["query"],
            "mode": output["mode"],
            "answer": answer,
            "retrieved_passages": list(retrieved_ids),
            "citations": citations,
            "checks": checks,
            "passed": passed
        }

        results.append(result)

        print("Mode:", output["mode"])
        for name, status in checks.items():
            print(f"{name}: {'PASS' if status else 'FAIL'}")
        print("Result:", "PASS" if passed else "FAIL")

    except Exception as error:
        results.append({
            "id": test["id"],
            "query": test["query"],
            "error": f"{type(error).__name__}: {error}",
            "passed": False
        })
        print("ERROR:", type(error).__name__, error)

passed_count = sum(item["passed"] for item in results)
total_count = len(results)

summary = {
    "total_tests": total_count,
    "passed": passed_count,
    "failed": total_count - passed_count,
    "pass_rate": round(passed_count / total_count, 4),
    "results": results
}

RESULT_PATH.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8"
)

print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
print(f"Passed: {passed_count}/{total_count}")
print(f"Failed: {total_count - passed_count}/{total_count}")
print(f"Pass rate: {summary['pass_rate'] * 100:.1f}%")
print(f"Detailed results: {RESULT_PATH.relative_to(ROOT)}")
