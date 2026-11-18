import os
import re
import json
from pypdf import PdfReader

PDF_PREFIX = {
    "pmkisan_guidelines.pdf": "PMKISAN",
    "pmfby_guidelines.pdf": "PMFBY",
    "kcc_guidelines.pdf": "KCC",
}

TARGET_WORDS = 220
MAX_WORDS = 280


def clean_line(line):
    line = re.sub(r"\s+", " ", line).strip()

    # Remove recurring website/page headers
    line = re.sub(
        r"www\.pmfby\.gov\.in\s+Operational Guidelines 2023",
        "",
        line,
        flags=re.IGNORECASE
    )

    return line.strip()


def is_numbered_heading(line):
    return bool(
        re.match(r"^\d+(\.\d+)*\.?\s+[A-Z]", line.strip())
    )


def extract_paragraphs(pdf_path):
    reader = PdfReader(pdf_path)
    paragraphs = []

    for page in reader.pages:
        text = page.extract_text() or ""
        lines = [clean_line(line) for line in text.splitlines()]

        current = []

        for line in lines:
            if not line:
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
                continue

            # Preserve numbered headings as separate paragraphs
            if is_numbered_heading(line):
                if current:
                    paragraphs.append(" ".join(current))
                    current = []

                paragraphs.append(line)
            else:
                current.append(line)

        if current:
            paragraphs.append(" ".join(current))

    return reader, paragraphs


def split_long_paragraph(paragraph):
    words = paragraph.split()
    pieces = []

    for start in range(0, len(words), TARGET_WORDS):
        pieces.append(" ".join(words[start:start + TARGET_WORDS]))

    return pieces


def make_chunks(paragraphs):
    chunks = []
    current = []
    current_length = 0

    for paragraph in paragraphs:
        words = paragraph.split()

        if not words:
            continue

        # Split unusually long paragraphs separately
        if len(words) > MAX_WORDS:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_length = 0

            chunks.extend(split_long_paragraph(paragraph))
            continue

        # Keep a paragraph intact where possible
        if current and current_length + len(words) > TARGET_WORDS:
            chunks.append(" ".join(current))
            current = []
            current_length = 0

        current.append(paragraph)
        current_length += len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks


def main():
    corpus = []

    for filename, prefix in PDF_PREFIX.items():
        path = os.path.join("docs", filename)

        if not os.path.exists(path):
            print("SKIP (not found):", path)
            continue

        print("\nReading:", path)

        reader, paragraphs = extract_paragraphs(path)
        chunks = make_chunks(paragraphs)

        print("Pages:", len(reader.pages))
        print("Paragraphs:", len(paragraphs))
        print("Chunks:", len(chunks))

        for i, chunk in enumerate(chunks, start=1):
            corpus.append({
                "id": f"{prefix}_SA2_P{i}",
                "text": chunk
            })

    os.makedirs("data", exist_ok=True)

    output_path = "data/corpus_section_aware_v2.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print("\nExtraction complete!")
    print("Total passages:", len(corpus))
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
