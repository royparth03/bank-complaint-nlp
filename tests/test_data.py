import pandas as pd

from complaint_nlp import data

LONG = " ".join(["word"] * 25)


def frame(rows):
    return pd.DataFrame(rows, columns=["Date received", "Product", "Issue", "Consumer complaint narrative"])


def test_clean_text_strips_masking_and_extra_spaces():
    out = data.clean_text(pd.Series(["Call XXXX on XX/XX/2020   now", None]))
    assert out.tolist() == ["Call on 2020 now", ""]


def test_prepare_drops_old_short_and_duplicate_complaints():
    df = frame([
        ["2019-05-01", "Mortgage", "i", LONG],
        ["2019-05-02", "Mortgage", "i", LONG],              # duplicate text
        ["2016-01-01", "Mortgage", "i", LONG + " old"],     # before the cut-off
        ["2019-05-03", "Mortgage", "i", "too short"],       # fewer than 20 words
    ])
    out = data.prepare(df, min_volume=1)
    assert len(out) == 1
    assert "text" in out.columns


def test_prepare_merges_renamed_products():
    df = frame([
        ["2019-01-01", "Credit card", "i", LONG + " a"],
        ["2019-01-01", "Credit card or prepaid card", "i", LONG + " b"],
        ["2019-01-01", "Credit reporting or other personal consumer reports", "i", LONG + " c"],
    ])
    out = data.prepare(df, min_volume=1)
    assert set(out["Product"]) == {"Credit card or prepaid card", "Credit reporting"}


def test_prepare_drops_rare_products():
    df = frame([["2019-01-01", "Mortgage", "i", LONG + f" {n}"] for n in range(3)]
               + [["2019-01-01", "Student loan", "i", LONG + " x"]])
    assert set(data.prepare(df, min_volume=2)["Product"]) == {"Mortgage"}


def test_balanced_sample_caps_each_product_and_is_reproducible():
    df = pd.DataFrame({"Product": ["A"] * 10 + ["B"] * 3, "text": [str(n) for n in range(13)]})
    s1 = data.balanced_sample(df, per_product=5, seed=1)
    s2 = data.balanced_sample(df, per_product=5, seed=1)
    assert s1["Product"].value_counts().to_dict() == {"A": 5, "B": 3}
    pd.testing.assert_frame_equal(s1, s2)
