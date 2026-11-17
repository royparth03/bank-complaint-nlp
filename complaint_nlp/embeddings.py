"""Sentence embeddings with a Hugging Face model (imported lazily so tests don't need PyTorch)."""
import numpy as np

from . import config


def embed(texts, model_name: str = config.EMBED_MODEL, max_tokens: int = config.EMBED_MAX_TOKENS) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = max_tokens
    return model.encode(list(texts), batch_size=64, show_progress_bar=False,
                        normalize_embeddings=True, convert_to_numpy=True).astype("float32")
