import json

CORPUS_PATH = "data/corpus_section_aware_v2.json"

TARGET_IDS = [
    "KCC_SA2_P27",
    "KCC_SA2_P22",
    "KCC_SA2_P26",
    "PMFBY_SA2_P389",
    "PMFBY_SA2_P278",
    "PMFBY_SA2_P308",
    "PMFBY_SA2_P168",
    "PMFBY_SA2_P416",
    "PMFBY_SA2_P267",
    "PMFBY_SA2_P437",
    "PMFBY_SA2_P309",
    "PMFBY_SA2_P82",
    "PMFBY_SA2_P232",
    "PMFBY_SA2_P164",
    "PMFBY_SA2_P58",
    "PMFBY_SA2_P60",
    "PMFBY_SA2_P351",
    "PMFBY_SA2_P360"
]

with open(CORPUS_PATH, encoding="utf-8") as f:
    corpus = json.load(f)

passages = {p["id"]: p["text"] for p in corpus}

for passage_id in TARGET_IDS:
    print("\n" + "=" * 90)
    print("PASSAGE:", passage_id)
    print("=" * 90)
    print(passages.get(passage_id, "PASSAGE NOT FOUND"))
