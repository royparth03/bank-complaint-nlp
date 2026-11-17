import numpy as np

from complaint_nlp import search


def unit(v):
    v = np.asarray(v, dtype="float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_search_returns_most_similar_items_first():
    corpus = unit([[1, 0], [0, 1], [1, 1]])
    nn = search.search(search.build_index(corpus), unit([[1, 0.1]]), k=2)
    assert nn.tolist() == [[0, 2]]


def test_index_matches_cosine_similarity_ranking():
    rng = np.random.default_rng(0)
    corpus, queries = unit(rng.normal(size=(50, 8))), unit(rng.normal(size=(5, 8)))
    faiss_nn = search.search(search.build_index(corpus), queries, k=3)
    numpy_nn = search.top_k_by_similarity(queries @ corpus.T, k=3)
    assert np.array_equal(faiss_nn, numpy_nn)


def test_precision_at_k():
    neighbours = np.array([[0, 1], [2, 3]])
    corpus_labels = np.array(["a", "b", "b", "b"])
    query_labels = np.array(["a", "b"])
    assert search.precision_at_k(neighbours, corpus_labels, query_labels) == 0.75
