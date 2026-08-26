"""
Unit tests for src/reranker.py — CrossEncoderReranker.

These tests load the actual cross-encoder/ms-marco-MiniLM-L-6-v2 model.
First run will download ~85MB. Subsequent runs use HF cache.

Run with: pytest tests/test_reranker.py -v
"""

import pytest

from src.reranker import CrossEncoderReranker

# ---------------------------------------------------------------------------
# Fixture: shared reranker instance (loaded once per test session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reranker():
    return CrossEncoderReranker()


def make_docs(texts: list, base_score: float = 0.5) -> list:
    return [
        {
            "text": t,
            "metadata": {"source": "test.txt", "chunk_index": i},
            "score": base_score,
            "index": i,
        }
        for i, t in enumerate(texts)
    ]


# ---------------------------------------------------------------------------
# 1. Returns all documents when top_k is None
# ---------------------------------------------------------------------------


def test_rerank_returns_all_when_no_top_k(reranker):
    docs = make_docs(["first doc", "second doc", "third doc"])
    results = reranker.rerank("test query", docs, top_k=None)
    assert len(results) == 3


# ---------------------------------------------------------------------------
# 2. Returns exactly top_k when specified
# ---------------------------------------------------------------------------


def test_rerank_top_k(reranker):
    docs = make_docs(["doc one", "doc two", "doc three", "doc four"])
    results = reranker.rerank("query", docs, top_k=2)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# 3. "rerank_score" key is added to each result
# ---------------------------------------------------------------------------


def test_rerank_score_key_added(reranker):
    docs = make_docs(["hello world", "goodbye world"])
    results = reranker.rerank("hello", docs)
    for r in results:
        assert "rerank_score" in r, "rerank_score key missing from result"


# ---------------------------------------------------------------------------
# 4. Results are sorted by rerank_score descending
# ---------------------------------------------------------------------------


def test_scores_descending(reranker):
    docs = make_docs(["apple fruit", "machine learning", "deep neural network", "cat and dog"])
    results = reranker.rerank("artificial intelligence", docs)
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Results not sorted descending"


# ---------------------------------------------------------------------------
# 5. A semantically relevant document is promoted above an unrelated one
# ---------------------------------------------------------------------------


def test_relevant_doc_promoted(reranker):
    query = "What is the capital of France?"
    docs = make_docs(
        [
            "The capital of France is Paris, a major European city.",  # highly relevant
            "Photosynthesis is the process by which plants make food.",  # irrelevant
        ]
    )
    results = reranker.rerank(query, docs)
    # The relevant document should end up first
    assert "Paris" in results[0]["text"] or "France" in results[0]["text"], (
        f"Relevant doc not at top. Top result: {results[0]['text']}"
    )


# ---------------------------------------------------------------------------
# 6. All original keys are preserved in output
# ---------------------------------------------------------------------------


def test_original_keys_preserved(reranker):
    docs = make_docs(["some text"])
    results = reranker.rerank("query", docs)
    required = {"text", "metadata", "score", "index", "rerank_score"}
    assert required.issubset(results[0].keys())


# ---------------------------------------------------------------------------
# 7. Single-document input does not crash
# ---------------------------------------------------------------------------


def test_single_document(reranker):
    docs = make_docs(["only one document here"])
    results = reranker.rerank("query", docs)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# 8. Empty document list returns empty list
# ---------------------------------------------------------------------------


def test_empty_documents(reranker):
    results = reranker.rerank("query", [])
    assert results == []
