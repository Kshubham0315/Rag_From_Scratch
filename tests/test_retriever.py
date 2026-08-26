"""
Unit tests for src/retriever.py — HybridRetriever.

All heavy dependencies (VectorStore, BM25, EmbeddingModel) are mocked
so these tests run instantly without loading ML models.

Run with: pytest tests/test_retriever.py -v
"""

from unittest.mock import MagicMock

import numpy as np

from src.retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_dense_results(n: int) -> list:
    """Simulate vectorstore.search() output."""
    return [
        {
            "text": f"dense_doc_{i}",
            "metadata": {"chunk_index": i, "source": "test.txt"},
            "score": 1.0 - i * 0.1,
            "index": i,
        }
        for i in range(n)
    ]


def make_sparse_results(n: int, offset: int = 0) -> list:
    """Simulate bm25.get_top_n() output."""
    return [{"text": f"sparse_doc_{i + offset}", "score": 1.0 - i * 0.1, "index": i + offset} for i in range(n)]


def make_retriever(dense_results=None, sparse_results=None, corpus_size=20):
    """Build a HybridRetriever with mocked components."""
    if dense_results is None:
        dense_results = make_dense_results(10)
    if sparse_results is None:
        sparse_results = make_sparse_results(10)

    vs = MagicMock()
    vs.search.return_value = dense_results
    vs._documents = [f"doc_{i}" for i in range(corpus_size)]
    vs._metadata = [{"chunk_index": i, "source": "test.txt"} for i in range(corpus_size)]
    vs.__len__ = MagicMock(return_value=corpus_size)

    bm = MagicMock()
    bm.get_top_n.return_value = sparse_results

    embedder = MagicMock()
    embedder.embed_query.return_value = np.ones(384, dtype=np.float32)

    return HybridRetriever(vectorstore=vs, bm25=bm, embedder=embedder)


# ---------------------------------------------------------------------------
# 1. retrieve() returns exactly top_k results
# ---------------------------------------------------------------------------


def test_retrieve_returns_top_k():
    retriever = make_retriever()
    results = retriever.retrieve("test query", top_k=3)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# 2. RRF merges results from both dense and sparse sources
# ---------------------------------------------------------------------------


def test_rrf_merges_both_sources():
    # Dense covers indices 0-4, sparse covers indices 5-9 (no overlap)
    dense = make_dense_results(5)
    sparse = make_sparse_results(5, offset=5)
    retriever = make_retriever(dense_results=dense, sparse_results=sparse, corpus_size=10)

    results = retriever.retrieve("query", top_k=10)
    result_indices = {r["index"] for r in results}

    # Both dense (0-4) and sparse (5-9) should be represented
    dense_present = any(i in result_indices for i in range(5))
    sparse_present = any(i in result_indices for i in range(5, 10))
    assert dense_present, "No dense results in output"
    assert sparse_present, "No sparse results in output"


# ---------------------------------------------------------------------------
# 3. RRF score formula is computed correctly
# ---------------------------------------------------------------------------


def test_rrf_score_formula():
    retriever = make_retriever()
    k = 60

    # Doc 0 is rank 1 in both lists → RRF = 1/(60+1) + 1/(60+1)
    expected_0 = 1.0 / (k + 1) + 1.0 / (k + 1)

    # Manually compute via internal method
    list_a = [{"index": 0}, {"index": 1}, {"index": 2}]
    list_b = [{"index": 0}, {"index": 3}]
    fused = retriever._reciprocal_rank_fusion([list_a, list_b], rrf_k=k)

    # Doc 0 should have the highest score
    top = fused[0]
    assert top["index"] == 0
    assert abs(top["rrf_score"] - expected_0) < 1e-10, f"Expected {expected_0:.8f}, got {top['rrf_score']:.8f}"


# ---------------------------------------------------------------------------
# 4. Documents appearing in both lists are deduplicated
# ---------------------------------------------------------------------------


def test_deduplication():
    # Doc 0 appears in both dense and sparse
    dense = [
        {"text": "doc_0", "metadata": {}, "score": 0.9, "index": 0},
        {"text": "doc_1", "metadata": {}, "score": 0.8, "index": 1},
    ]
    sparse = [{"text": "doc_0", "score": 0.7, "index": 0}, {"text": "doc_2", "score": 0.6, "index": 2}]
    retriever = make_retriever(dense_results=dense, sparse_results=sparse)

    results = retriever.retrieve("query", top_k=10)
    indices = [r["index"] for r in results]

    # Index 0 must appear exactly once
    assert indices.count(0) == 1, f"Index 0 appears {indices.count(0)} times"


# ---------------------------------------------------------------------------
# 5. Results have all required keys
# ---------------------------------------------------------------------------


def test_result_has_required_keys():
    retriever = make_retriever()
    results = retriever.retrieve("query", top_k=5)
    required = {"text", "metadata", "score", "index"}
    for r in results:
        assert required.issubset(r.keys()), f"Missing keys: {required - r.keys()}"


# ---------------------------------------------------------------------------
# 6. Higher rrf_k flattens score differences
# ---------------------------------------------------------------------------


def test_rrf_k_effect():
    list_a = [{"index": i} for i in range(5)]
    list_b = [{"index": i} for i in range(5)]

    r = make_retriever()

    fused_low_k = r._reciprocal_rank_fusion([list_a, list_b], rrf_k=1)
    fused_high_k = r._reciprocal_rank_fusion([list_a, list_b], rrf_k=1000)

    # Spread between first and last score should be smaller with higher k
    spread_low = fused_low_k[0]["rrf_score"] - fused_low_k[-1]["rrf_score"]
    spread_high = fused_high_k[0]["rrf_score"] - fused_high_k[-1]["rrf_score"]

    assert spread_high < spread_low, f"Expected high-k spread ({spread_high:.6f}) < low-k spread ({spread_low:.6f})"


# ---------------------------------------------------------------------------
# 7. Works when sparse and dense results have no overlap (dense-only enrichment)
# ---------------------------------------------------------------------------


def test_dense_only_fallback():
    # All sparse hits not in dense; vectorstore._documents covers them
    dense = make_dense_results(3)  # indices 0,1,2
    sparse = make_sparse_results(3, offset=10)  # indices 10,11,12
    retriever = make_retriever(
        dense_results=dense,
        sparse_results=sparse,
        corpus_size=20,
    )
    results = retriever.retrieve("query", top_k=6)
    assert len(results) == 6
