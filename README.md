# Bank Complaint Analytics: NLP Routing & Semantic Search

Two simple NLP tools a bank's complaints team would use, built on real US bank complaints.

**Data:** CFPB Consumer Complaint Database, complaints with a written narrative (Hugging Face mirror
`BEE-spoke-data/consumer-finance-complaints`, has-text shard 0: 563K complaints, 2015 to Feb 2024).
**Get the data:** download the has-text shard 0 parquet file from the Hugging Face dataset above and save it as `data/complaints_with_text_part0.parquet` (about 290 MB, so it isn't in this repo).
**Run (project root):** `pip install -r requirements.txt`, then `python -m complaint_nlp.pipeline`. Takes about 6 minutes on CPU.
**Tests:** `python -m pytest --cov=complaint_nlp`. 11 unit tests that need no data or model download. A GitHub Actions workflow is included to run them automatically once the repo is on GitHub.

## Code structure

```
complaint_nlp/
  config.py      all settings (seed, date cut-off, sample sizes, product renames)
  data.py        clean_text, prepare (filter + merge renamed products), balanced_sample
  routing.py     TF-IDF vectorizer, logistic-regression classifier, evaluate, majority baseline
  search.py      FAISS index build/search, cosine top-k, precision@k
  embeddings.py  Hugging Face sentence-transformer wrapper (lazy import, so tests don't need PyTorch)
  pipeline.py    end-to-end run that writes outputs/results.json
tests/           pytest unit tests for data, routing and search, using small synthetic inputs
.github/workflows/tests.yml   CI: install the light dependencies, run pytest with coverage
```

The original version was one long script. It was refactored into small, single-purpose modules so that each piece
can be unit tested in isolation, and the full pipeline was rerun to confirm the refactor reproduces exactly the same
results. The theme-clustering experiment was removed in the refactor because it wasn't meaningful (see the end of
this file).

## Preparing the data

- Kept complaints from 2018 onwards, after the CFPB's 2017 product-category change, with at least 20 words of narrative. Removed exact duplicates.
- Stripped `XXXX`, which is how the CFPB masks personal details, so the model can't learn from the masking.
- **Merged renamed products** so each queue has one name. For example, "Credit card" and "Credit card or prepaid card" are the same queue under two names.
- Took a **balanced sample of 2,500 per product: 22,500 complaints across 9 product queues.** Credit reporting is about 65% of raw volume and would otherwise swamp everything else.
- Split 80/20 into training and test sets, keeping the product mix the same in both.

## 1. Routing: which product team should get this complaint?

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Always guess the biggest class | 11% | 0.02 |
| **TF-IDF (words and word pairs) + logistic regression** | **81%** | **0.81** |
| MiniLM sentence embeddings + logistic regression | 77% | 0.77 |

**Why the simple model wins:** product routing depends on keywords such as "mortgage", "student loan" and "debt collector", and TF-IDF captures these directly. MiniLM squeezes each complaint into 384 numbers and loses some of that detail. Its first 128 tokens are also cut off.

## 2. Semantic search: find similar past complaints

- Each complaint is turned into a 384-dimension vector with **`all-MiniLM-L6-v2`** (a Hugging Face sentence-transformer) and stored in a **FAISS** inner-product index. Because the vectors are normalised, the inner product equals cosine similarity.
- **Test:** 2,000 unseen complaints were used as search queries. A result counts as relevant when it has the same CFPB *issue* as the query; there are 86 issues, which is much finer than the 9 products.
- **Precision@5: 35% with embeddings vs. 30% with TF-IDF keyword search.** Search is where embeddings beat keywords, because they match meaning even when people use different words.

## What it would take at a bank

- Use the bank's own complaint queues as labels, retrain regularly, and send low-confidence predictions to a human.
- Add a generation step on top of the search to turn it into RAG (retrieval-augmented generation), for example to summarise similar resolved complaints for an agent. **Not built here.**

## An experiment that didn't work

An earlier version also tried K-means "theme" clustering of checking and savings complaints. The clusters mostly grouped complaints **by bank name** (Chase, Wells Fargo, Chime), not by type of problem, so they weren't meaningful.
