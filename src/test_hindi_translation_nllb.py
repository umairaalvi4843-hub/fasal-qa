import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"
INPUT_FILE = "data/heldout_multilingual_queries_audited.json"

print("Loading NLLB tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="hin_Deva")

print("Loading NLLB model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    dataset = json.load(file)

hindi_queries = [
    item["query"]
    for item in dataset
    if re.search(r"[\u0900-\u097F]", item["query"])
]

print(f"\nFound {len(hindi_queries)} Hindi queries.\n")

for query in hindi_queries:
    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        translated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_new_tokens=128
        )

    english_query = tokenizer.decode(
        translated[0],
        skip_special_tokens=True
    )

    print("Hindi:   ", query)
    print("English: ", english_query)
    print("-" * 70)

print("\nNLLB translation test completed.")
