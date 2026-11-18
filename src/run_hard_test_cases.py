import json
import sys
import time
from pathlib import Path

# Make the project folders importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(DATA_DIR))

from rag_answer import generate_answer
from hard_test_cases_15 import HARD_TEST_CASES


LANGUAGES = ["english", "hindi", "hinglish"]
RESULTS_FILE = DATA_DIR / "hard_test_case_results.json"


def main():
    results = []
    total = len(HARD_TEST_CASES) * len(LANGUAGES)

    print("=" * 65)
    print("FASAL QA - HARD TEST CASE EVALUATION")
    print(f"Test scenarios: {len(HARD_TEST_CASES)}")
    print(f"Language versions per scenario: {len(LANGUAGES)}")
    print(f"Total RAG runs: {total}")
    print("=" * 65)

    for case_number, case in enumerate(HARD_TEST_CASES, start=1):

        if len(case) != 3:
            print(f"Skipping test case {case_number}: expected 3 language versions.")
            continue

        for language_index, question in enumerate(case):
            language = LANGUAGES[language_index]
            run_number = len(results) + 1

            print(f"\n[{run_number}/{total}] Case {case_number} | {language}")
            print(f"Question: {question}")

            try:
                response = generate_answer(question)

                result = {
                    "case_id": f"HC{case_number:02d}",
                    "language": language,
                    "question": question,
                    "status": "success",
                    "answer": response.get("answer", ""),
                    "mode": response.get("mode", ""),
                    "retrieved_passages": response.get("retrieved_passages", [])
                }

                print("\nAnswer:")
                print(result["answer"])

                print("\nRetrieved passages:")
                for passage in result["retrieved_passages"]:
                    print(passage)

            except Exception as error:
                result = {
                    "case_id": f"HC{case_number:02d}",
                    "language": language,
                    "question": question,
                    "status": "error",
                    "error": str(error)
                }

                print(f"ERROR: {error}")

            results.append(result)

            # Save progress after every run so results are not lost.
            RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(RESULTS_FILE, "w", encoding="utf-8") as file:
                json.dump(results, file, ensure_ascii=False, indent=2)

            # Avoid overwhelming the Groq API.
            if run_number < total:
                print("\nWaiting 15 seconds before the next run...")
                time.sleep(15)

    successful = sum(1 for item in results if item["status"] == "success")
    failed = len(results) - successful

    print("\n" + "=" * 65)
    print("HARD TEST CASE RUNS COMPLETED")
    print(f"Successful runs: {successful}")
    print(f"Failed runs: {failed}")
    print(f"Results saved to: {RESULTS_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()
