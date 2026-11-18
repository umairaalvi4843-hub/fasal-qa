import json
import re
import torch
from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-hi-en"
INPUT_FILE = "data/heldout_multilingual_queries_audited.json"

GLOSSARY = {
    "प्रधानमंत्री फसल बीमा योजना": "Pradhan Mantri Fasal Bima Yojana",
    "पीएमएफबीवाई": "PMFBY",
    "पीएमएफबीवाई": "PMFBY",
    "पीएम-किसान": "PM-KISAN",
    "पीएम किसान": "PM-KISAN",
    "किसान क्रेडिट कार्ड": "Kisan Credit Card",
    "बटाईदार": "sharecropper",
    "किरायेदार किसान": "tenant farmer",
    "फसल बीमा योजना": "crop insurance scheme",
    "शिकायत": "complaint",
    "समाधान": "resolution",
}

# Match longer terms first.
terms = sorted(GLOSSARY.keys(), key=len, reverse=True)
term_pattern = re.compile("|".join(re.escape(term) for term in terms))

print("Loading translation model...")
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)
model.eval()


def translate_segment(segment):
    if not segment.strip():
        return segment

    inputs = tokenizer(
        [segment],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128)

    return tokenizer.decode(output[0], skip_special_tokens=True)


def translate_query(query):
    pieces = []
    last_position = 0

    for match in term_pattern.finditer(query):
        # Translate the Hindi text before the protected term.
        preceding_text = query[last_position:match.start()]
        if preceding_text.strip():
            pieces.append(translate_segment(preceding_text))

        # Insert the known English equivalent directly.
        pieces.append(GLOSSARY[match.group()])
        last_position = match.end()

    # Translate the remaining text.
    remaining_text = query[last_position:]
    if remaining_text.strip():
        pieces.append(translate_segment(remaining_text))

    return " ".join(piece.strip() for piece in pieces if piece.strip())


with open(INPUT_FILE, "r", encoding="utf-8") as file:
    dataset = json.load(file)

hindi_queries = [
    item["query"]
    for item in dataset
    if re.search(r"[\u0900-\u097F]", item["query"])
]

print(f"\nFound {len(hindi_queries)} Hindi queries.\n")

for query in hindi_queries:
    english_query = translate_query(query)

    print("Hindi:   ", query)
    print("English: ", english_query)
    print("-" * 70)

print("\nSegment-based translation test completed.")
