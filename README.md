# Fasal QA

**Fasal QA** is a multilingual, evidence-grounded question-answering project focused on Indian agricultural government schemes. It aims to help farmers retrieve reliable information from official scheme guidelines using natural-language questions.

## Schemes Covered

- **PM-KISAN:** Pradhan Mantri Kisan Samman Nidhi
- **PMFBY:** Pradhan Mantri Fasal Bima Yojana
- **KCC:** Kisan Credit Card

## Dataset

The initial corpus contains **496 text passages** extracted from three government guideline documents.

| Scheme    | Passages |
| --------- | -------: |
| PM-KISAN  |       21 |
| PMFBY     |      448 |
| KCC       |       27 |
| **Total** |  **496** |

The corpus is stored in `data/corpus.json`.

## Current Implementation

- PDF text extraction and overlapping passage generation.
- BM25 lexical retrieval using `rank-bm25`.
- A manually labelled evaluation set of 12 questions.
- Retrieval evaluation using Recall@3 and Mean Reciprocal Rank (MRR).

### Initial BM25 Results

| Metric               |  Score |
| -------------------- | -----: |
| Average Recall@3     | 0.2292 |
| Mean Reciprocal Rank | 0.3056 |
| Evaluation questions |     12 |

These are preliminary results from a small evaluation set.

## Project Structure

```text
fasal-qa/
├── data/
├── docs/
├── src/
│   ├── extract_pdfs.py
│   ├── bm25_baseline.py
│   └── evaluate_bm25.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Clone the repository:

```bash
git clone <repository-url>
cd fasal-qa
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Extract text from the PDFs:

```bash
python src/extract_pdfs.py
```

Run BM25 retrieval:

```bash
python src/bm25_baseline.py
```

Evaluate retrieval:

```bash
python src/evaluate_bm25.py
```

## Roadmap

- [x] Collect government scheme guideline documents.
- [x] Extract PDF text and build the initial corpus.
- [x] Implement BM25 lexical retrieval.
- [x] Establish an initial retrieval evaluation.
- [ ] Implement multilingual dense retrieval using E5 embeddings.
- [ ] Compare lexical and dense retrieval.
- [ ] Develop hybrid retrieval.
- [ ] Expand Hindi, Bengali and Hinglish evaluation.
- [ ] Build a retrieval-augmented question-answering pipeline.
- [ ] Add source references and hallucination control.
- [ ] Perform broader evaluation and error analysis.

## Note

Fasal QA is an ongoing project. The current implementation is a retrieval baseline, not a complete question-answering system. Always verify scheme information against the latest official guidelines before relying on it for real-world decisions.
