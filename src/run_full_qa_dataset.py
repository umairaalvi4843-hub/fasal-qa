import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(DATA_DIR))

from rag_answer import generate_answer
from qa_dataset_500 import groups

LANGUAGES = ["english", "hindi", "hinglish"]

RESULTS_FILE = DATA_DIR / "full_qa_dataset_results.json"
CHECKPOINT_INTERVAL = 1
DELAY_SECONDS = 15


def main():
    results = []

    # Count all valid language-version questions.
    total = sum(
        len(case) for cases in groups.values() for case in cases
        if isinstance(case, (tuple, list)) and len(case) == 3
    )

    print("=" * 65)
    print("FASAL QA - FULL DATASET EVALUATION")
    print(f"Question topics: {sum(len(cases) for cases in groups.values())}")
    print(f"Total language-version runs: {total}")
    print("=" * 65)

    run_number = 0

    for scheme, cases in groups.items():
        for case_number, case in enumerate(cases, start=1):

            if not isinstance(case, (tuple, list)) or len(case) != 3:
                print(f"Skipping invalid case: {scheme}, case {case_number}")
                continue

            for language, question in zip(LANGUAGES, case):
                run_number += 1

                print(f"\n[{run_number}/{total}] {scheme} | {language}")
                print(f"Question: {question}")

                try:
                    response = generate_answer(question)

                    result = {
                        "scheme": scheme,
                        "case_number": case_number,
                        "language": language,
                        "question": question,
                        "status": "success",
                        "answer": response.get("answer", ""),
                        "mode": response.get("mode", ""),
                        "retrieved_passages": response.get(
                            "retrieved_passages", []
                        )
                    }

                    print("\nAnswer:")
                    print(result["answer"])

                    print("\nRetrieved passages:")
                    for passage in result["retrieved_passages"]:
                        print(passage)

                except Exception as error:
                    result = {
                        "scheme": scheme,
                        "case_number": case_number,
                        "language": language,
                        "question": question,
                        "status": "error",
                        "error": str(error)
                    }

                    print(f"ERROR: {error}")

                results.append(result)

                # Save after every question so progress is preserved.
                with open(RESULTS_FILE, "w", encoding="utf-8") as file:
                    json.dump(results, file, ensure_ascii=False, indent=2)

                if run_number < total:
                    print(f"\nWaiting {DELAY_SECONDS} seconds...")
                    time.sleep(DELAY_SECONDS)

    successful = sum(r["status"] == "success" for r in results)
    failed = len(results) - successful

    print("\n" + "=" * 65)
    print("FULL DATASET RUN COMPLETED")
    print(f"Total runs: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results: {RESULTS_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()
