"""
Unit tests for src/vectorstore.py — VectorStore.
Run with: pytest tests/test_vectorstore.py -v
"""

import numpy as np
import pytest

from src.vectorstore import VectorStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_unit_vectors(n: int, d: int = 4, seed: int = 0) -> np.ndarray:
    """Return n random L2-normalized vectors of dimension d."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def make_docs(n: int) -> list:
    return [f"document_{i}" for i in range(n)]


def make_meta(n: int) -> list:
    return [{"chunk_index": i, "source": "test.txt"} for i in range(n)]


# ---------------------------------------------------------------------------
# 1. add() and __len__
# ---------------------------------------------------------------------------


def test_add_and_len():
    store = VectorStore()
    assert len(store) == 0
    embs = make_unit_vectors(5)
    store.add(embs, make_docs(5), make_meta(5))
    assert len(store) == 5


# ---------------------------------------------------------------------------
# 2. search() returns exactly top_k results
# ---------------------------------------------------------------------------


def test_search_returns_top_k():
    store = VectorStore()
    store.add(make_unit_vectors(10), make_docs(10), make_meta(10))
    results = store.search(make_unit_vectors(1)[0], top_k=3)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# 3. Querying with the exact stored vector gives score ~1.0
# ---------------------------------------------------------------------------


def test_identical_vector_scores_one():
    store = VectorStore()
    embs = make_unit_vectors(5)
    store.add(embs, make_docs(5), make_meta(5))
    # Query with the second stored vector exactly
    results = store.search(embs[2], top_k=1)
    assert len(results) == 1
    assert abs(results[0]["score"] - 1.0) < 1e-5, f"Expected score ~1.0, got {results[0]['score']}"


# ---------------------------------------------------------------------------
# 4. Result dicts have all required keys
# ---------------------------------------------------------------------------


def test_result_keys():
    store = VectorStore()
    store.add(make_unit_vectors(3), make_docs(3), make_meta(3))
    results = store.search(make_unit_vectors(1)[0], top_k=2)
    required = {"text", "metadata", "score", "index"}
    for r in results:
        assert required.issubset(r.keys()), f"Missing keys: {required - r.keys()}"


# ---------------------------------------------------------------------------
# 5. Results are sorted by score descending
# ---------------------------------------------------------------------------


def test_scores_descending():
    store = VectorStore()
    store.add(make_unit_vectors(20), make_docs(20), make_meta(20))
    results = store.search(make_unit_vectors(1, seed=99)[0], top_k=10)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Scores not in descending order"


# ---------------------------------------------------------------------------
# 6. Multiple calls to add() accumulate correctly
# ---------------------------------------------------------------------------


def test_multiple_adds():
    store = VectorStore()
    store.add(make_unit_vectors(3), make_docs(3), make_meta(3))
    store.add(make_unit_vectors(4, seed=1), make_docs(4), make_meta(4))
    assert len(store) == 7


# ---------------------------------------------------------------------------
# 7. save() / load() roundtrip produces identical results
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    store = VectorStore()
    embs = make_unit_vectors(8)
    docs = make_docs(8)
    meta = make_meta(8)
    store.add(embs, docs, meta)

    save_path = str(tmp_path / "store")
    store.save(save_path)

    store2 = VectorStore()
    store2.load(save_path)

    assert len(store2) == 8

    query = embs[3]
    r1 = store.search(query, top_k=3)
    r2 = store2.search(query, top_k=3)

    for a, b in zip(r1, r2, strict=True):
        assert a["text"] == b["text"]
        assert abs(a["score"] - b["score"]) < 1e-5
        assert a["index"] == b["index"]


# ---------------------------------------------------------------------------
# 8. top_k larger than corpus does not crash
# ---------------------------------------------------------------------------


def test_top_k_larger_than_corpus():
    store = VectorStore()
    store.add(make_unit_vectors(3), make_docs(3), make_meta(3))
    results = store.search(make_unit_vectors(1)[0], top_k=100)
    assert len(results) == 3  # capped at corpus size


# ---------------------------------------------------------------------------
# 9. Metadata dict is preserved exactly
# ---------------------------------------------------------------------------


def test_metadata_preserved():
    store = VectorStore()
    meta = [{"source": "doc.pdf", "page": 7, "chunk_index": 2}]
    store.add(make_unit_vectors(1), ["hello"], meta)
    results = store.search(make_unit_vectors(1)[0], top_k=1)
    assert results[0]["metadata"] == meta[0]


# ---------------------------------------------------------------------------
# 10. Searching an empty store returns empty list (no crash)
# ---------------------------------------------------------------------------


def test_search_empty_store():
    store = VectorStore()
    results = store.search(make_unit_vectors(1)[0], top_k=5)
    assert results == []


# ---------------------------------------------------------------------------
# 11. Mismatched lengths raise ValueError
# ---------------------------------------------------------------------------


def test_mismatched_lengths_raise():
    store = VectorStore()
    with pytest.raises(ValueError):
        store.add(make_unit_vectors(3), make_docs(2), make_meta(3))


# ---------------------------------------------------------------------------
# 12. save() creates both .npz and .json files
# ---------------------------------------------------------------------------


def test_save_creates_both_files(tmp_path):
    store = VectorStore()
    store.add(make_unit_vectors(2), make_docs(2), make_meta(2))
    save_path = str(tmp_path / "mystore")
    store.save(save_path)
    assert (tmp_path / "mystore.npz").exists()
    assert (tmp_path / "mystore.json").exists()


# ---------------------------------------------------------------------------
# 13. Text content is preserved through save/load
# ---------------------------------------------------------------------------


def test_text_preserved_after_load(tmp_path):
    store = VectorStore()
    docs = ["The quick brown fox", "jumped over the lazy dog", "hello world"]
    store.add(make_unit_vectors(3), docs, make_meta(3))

    save_path = str(tmp_path / "store")
    store.save(save_path)

    store2 = VectorStore()
    store2.load(save_path)

    query = make_unit_vectors(3)[0]  # arbitrary query
    results = store2.search(query, top_k=3)
    retrieved_texts = {r["text"] for r in results}
    for doc in docs:
        assert doc in retrieved_texts
