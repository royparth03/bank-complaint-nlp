"""Semantic search over normalised embeddings with a FAISS index."""
import faiss
import numpy as np


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Exact inner-product index; on L2-normalised vectors this is cosine similarity."""
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype="float32"))
    return index


def search(index: faiss.IndexFlatIP, queries: np.ndarray, k: int) -> np.ndarray:
    """Row i holds the corpus positions of the k most similar items to query i."""
    _, neighbours = index.search(np.ascontiguousarray(queries, dtype="float32"), k)
    return neighbours


def top_k_by_similarity(similarity: np.ndarray, k: int) -> np.ndarray:
    """Same output shape as `search`, from a dense query x corpus similarity matrix."""
    return np.argsort(-similarity, axis=1)[:, :k]


def precision_at_k(neighbours: np.ndarray, corpus_labels: np.ndarray, query_labels: np.ndarray) -> float:
    """Share of retrieved items whose label matches their query's label."""
    return float((corpus_labels[neighbours] == query_labels[:, None]).mean())
