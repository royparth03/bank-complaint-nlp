import numpy as np

from complaint_nlp import routing

TEXTS = ["my mortgage payment escrow went up", "mortgage servicer lost my escrow payment",
         "home mortgage escrow analysis wrong", "student loan servicer misapplied payment",
         "student loan forgiveness application denied", "my student loan interest is wrong"] * 3
LABELS = ["Mortgage"] * 3 + ["Student loan"] * 3
LABELS = LABELS * 3


def test_router_learns_obvious_keywords():
    vec = routing.make_vectorizer(min_df=1)
    X = vec.fit_transform(TEXTS)
    model = routing.make_classifier().fit(X, LABELS)
    pred = model.predict(vec.transform(["escrow on my mortgage", "student loan interest"]))
    assert pred.tolist() == ["Mortgage", "Student loan"]


def test_evaluate_reports_accuracy_and_macro_f1():
    out = routing.evaluate(["a", "a", "b", "b"], ["a", "b", "b", "b"])
    assert out == {"accuracy": 0.75, "macro_f1": 0.733}


def test_majority_baseline_predicts_most_common_label():
    assert routing.majority_baseline(np.array(["x", "y", "y"]), 2).tolist() == ["y", "y"]
