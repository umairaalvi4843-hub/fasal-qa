import os
import json
from pypdf import PdfReader

PDF_PREFIX = {
    "pmkisan_guidelines.pdf": "PMKISAN",
    "pmfby_guidelines.pdf": "PMFBY",
    "kcc_guidelines.pdf": "KCC",
}

def chunk_text(text, chunk_size=256, overlap=50):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks

def main():
    corpus = []

    for filename, prefix in PDF_PREFIX.items():
        path = os.path.join("docs", filename)

        if not os.path.exists(path):
            print("SKIP (not found): " + path)
            continue

        print("Reading: " + path)

        reader = PdfReader(path)
        full_text = ""

        for page_num, page in enumerate(reader.pages, start=1):
            extracted = page.extract_text() or ""
            full_text += extracted + "\n"

        print("  Pages: " + str(len(reader.pages)))
        print("  Extracted characters: " + str(len(full_text)))

        if not full_text.strip():
            print("  WARNING: No text extracted. PDF may require OCR.")

        chunks = chunk_text(full_text)

        print("  Chunks: " + str(len(chunks)))

        for i, chunk in enumerate(chunks, start=1):
            corpus.append({
                "id": prefix + "_P" + str(i),
                "text": chunk
            })

    os.makedirs("data", exist_ok=True)

    with open("data/corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print("\nExtraction complete!")
    print("Total passages: " + str(len(corpus)))
    print("Saved to data/corpus.json")

if __name__ == "__main__":
    main()
