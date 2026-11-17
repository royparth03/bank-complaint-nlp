"""Routing: predict which product queue a complaint belongs to."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def make_vectorizer(min_df: int = 3) -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_features=200_000, sublinear_tf=True,
                           stop_words="english")


def make_classifier() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")


def evaluate(y_true, y_pred) -> dict:
    return {"accuracy": round(accuracy_score(y_true, y_pred), 3),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 3)}


def majority_baseline(y_train, n: int) -> np.ndarray:
    values, counts = np.unique(y_train, return_counts=True)
    return np.full(n, values[np.argmax(counts)])
