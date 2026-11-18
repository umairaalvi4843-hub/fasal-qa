import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_answer import generate_answer

RESULTS_FILE = DATA_DIR / "hard_test_case_results.json"
FALLBACK_MODE = "Evidence-only fallback"
WAIT_SECONDS = 15


def main():
    if not RESULTS_FILE.exists():
        print(f"Results file not found: {RESULTS_FILE}")
        return

    with open(RESULTS_FILE, "r", encoding="utf-8") as file:
        results = json.load(file)

    fallback_indices = [
        i for i, item in enumerate(results)
        if item.get("mode") == FALLBACK_MODE
    ]

    print("=" * 65)
    print("FASAL QA - TARGETED HARD TEST RETRY")
    print(f"Existing results: {len(results)}")
    print(f"Fallback answers to retry: {len(fallback_indices)}")
    print("Previously generated answers will be preserved.")
    print("=" * 65)

    if not fallback_indices:
        print("No fallback answers found. Nothing to retry.")
        return

    for attempt, index in enumerate(fallback_indices, start=1):
        old_result = results[index]

        print(f"\n[{attempt}/{len(fallback_indices)}] "
              f"{old_result.get('case_id')} | "
              f"{old_result.get('language')}")
        print("Question:", old_result.get("question"))

        try:
            response = generate_answer(old_result["question"])

            new_result = {
                **old_result,
                "status": "success",
                "answer": response.get("answer", ""),
                "mode": response.get("mode", ""),
                "retrieved_passages": response.get(
                    "retrieved_passages", []
                )
            }

            results[index] = new_result

            print("New mode:", new_result["mode"])
            print("Answer:", new_result["answer"])

        except Exception as error:
            print("Retry failed:", error)
            results[index]["retry_error"] = str(error)

        with open(RESULTS_FILE, "w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=2)

        if attempt < len(fallback_indices):
            print(f"Waiting {WAIT_SECONDS} seconds...")
            time.sleep(WAIT_SECONDS)

    remaining = sum(
        1 for item in results
        if item.get("mode") == FALLBACK_MODE
    )

    print("\n" + "=" * 65)
    print("TARGETED RETRY COMPLETED")
    print(f"Total saved results: {len(results)}")
    print(f"Fallback answers remaining: {remaining}")
    print(f"Updated file: {RESULTS_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()
