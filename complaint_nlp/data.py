"""Loading, cleaning and sampling complaint narratives."""
import pandas as pd

from . import config


def clean_text(text: pd.Series) -> pd.Series:
    """Strip CFPB's XXXX masking of personal data and collapse whitespace."""
    return (text.fillna("")
            .str.replace(r"X{2,}[/\-]?", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True))


def prepare(df: pd.DataFrame, date_from: str = config.DATE_FROM, min_words: int = config.MIN_WORDS,
            min_volume: int = config.MIN_PRODUCT_VOLUME) -> pd.DataFrame:
    """Clean narratives, keep recent and long-enough complaints, merge renamed products."""
    d = df.rename(columns={"Consumer complaint narrative": "text"}).copy()
    d["text"] = clean_text(d["text"])
    d = d[(d["Date received"] >= date_from) & (d["text"].str.split().str.len() >= min_words)]
    d = d.drop_duplicates("text")
    d["Product"] = d["Product"].replace(config.PRODUCT_MAP)
    counts = d["Product"].value_counts()
    return d[d["Product"].isin(counts[counts >= min_volume].index)]


def balanced_sample(df: pd.DataFrame, per_product: int = config.PER_PRODUCT,
                    seed: int = config.SEED) -> pd.DataFrame:
    """At most `per_product` complaints from each product, reproducibly."""
    return (df.groupby("Product", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), per_product), random_state=seed))
              .reset_index(drop=True))


def load(path) -> pd.DataFrame:
    return pd.read_parquet(path, columns=config.COLUMNS)
