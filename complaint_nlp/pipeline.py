"""End-to-end run: prepare data, compare routing models, evaluate semantic search.

Run from the project root: python -m complaint_nlp.pipeline
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from . import config, data, routing, search
from .embeddings import embed

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "complaints_with_text_part0.parquet"
OUT = ROOT / "outputs"


def main():
    OUT.mkdir(exist_ok=True)
    raw = data.load(RAW)
    res = {"raw_rows": len(raw), "date_min": raw["Date received"].min(), "date_max": raw["Date received"].max()}
    d = data.prepare(raw)
    res["clean_rows"] = len(d)
    s = data.balanced_sample(d)
    res["sample_rows"] = len(s)
    res["products"] = s["Product"].value_counts().to_dict()

    E = embed(s["text"])
    tr, te = train_test_split(np.arange(len(s)), test_size=config.TEST_SHARE, stratify=s["Product"],
                              random_state=config.SEED)
    y = s["Product"].to_numpy()

    # Routing
    vec = routing.make_vectorizer()
    Xtr, Xte = vec.fit_transform(s["text"].iloc[tr]), vec.transform(s["text"].iloc[te])
    p_tfidf = routing.make_classifier().fit(Xtr, y[tr]).predict(Xte)
    p_emb = routing.make_classifier().fit(E[tr], y[tr]).predict(E[te])
    res["routing"] = {
        "Majority class": routing.evaluate(y[te], routing.majority_baseline(y[tr], len(te))),
        "TF-IDF + logistic regression": routing.evaluate(y[te], p_tfidf),
        "MiniLM embeddings + logistic regression": routing.evaluate(y[te], p_emb),
    }
    pd.DataFrame(classification_report(y[te], p_tfidf, output_dict=True)).T.round(3) \
        .to_csv(OUT / "routing_report_tfidf.csv")

    # Semantic search vs. keyword search
    q = np.random.default_rng(config.SEED).choice(te, config.SEARCH_QUERIES, replace=False)
    issue = s["Issue"].to_numpy()
    nn_emb = search.search(search.build_index(E[tr]), E[q], config.TOP_K)
    nn_kw = search.top_k_by_similarity((vec.transform(s["text"].iloc[q]) @ Xtr.T).toarray(), config.TOP_K)
    res["search"] = {
        "queries": config.SEARCH_QUERIES, "index_size": int(len(tr)), "issues": int(s["Issue"].nunique()),
        "precision_at_5_same_issue_embeddings": round(search.precision_at_k(nn_emb, issue[tr], issue[q]), 3),
        "precision_at_5_same_issue_tfidf": round(search.precision_at_k(nn_kw, issue[tr], issue[q]), 3),
    }

    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
