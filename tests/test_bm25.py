"""
Unit tests for src/bm25.py — BM25.

All tests use small hand-crafted corpora so expected scores can be
reasoned about directly without a calculator.

Run with: pytest tests/test_bm25.py -v
"""

import numpy as np
import pytest

from src.bm25 import BM25

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SMALL_CORPUS = [
    "the cat sat on the mat",
    "the dog barked at the cat",
    "machine learning is a subset of artificial intelligence",
]


def fitted_bm25(**kwargs) -> BM25:
    bm = BM25(**kwargs)
    bm.fit(SMALL_CORPUS)
    return bm


# ---------------------------------------------------------------------------
# 1. fit() does not crash
# ---------------------------------------------------------------------------


def test_fit_does_not_crash():
    bm = BM25()
    bm.fit(SMALL_CORPUS)
    assert bm._fitted


# ---------------------------------------------------------------------------
# 2. get_scores() returns array of length N
# ---------------------------------------------------------------------------


def test_get_scores_shape():
    bm = fitted_bm25()
    scores = bm.get_scores("cat")
    assert len(scores) == len(SMALL_CORPUS)


# ---------------------------------------------------------------------------
# 3. Document with most query terms scores highest
# ---------------------------------------------------------------------------


def test_exact_term_match_scores_highest():
    corpus = [
        "apple banana cherry",
        "apple banana cherry date elderberry",  # contains all + more
        "unrelated document about football",
    ]
    bm = BM25()
    bm.fit(corpus)
    scores = bm.get_scores("apple banana cherry")
    # Both docs 0 and 1 match all query terms; doc 2 does not
    assert scores[2] == 0.0
    assert scores[0] > 0.0
    assert scores[1] > 0.0


# ---------------------------------------------------------------------------
# 4. Document with zero query term overlap scores 0.0
# ---------------------------------------------------------------------------


def test_zero_score_no_overlap():
    bm = fitted_bm25()
    scores = bm.get_scores("machine learning")
    # Docs 0 and 1 mention neither "machine" nor "learning"
    assert scores[0] == 0.0
    assert scores[1] == 0.0
    assert scores[2] > 0.0


# ---------------------------------------------------------------------------
# 5. Rare term has higher IDF than common term
# ---------------------------------------------------------------------------


def test_idf_rare_term_higher():
    # "cat" appears in 2 of 3 docs → lower IDF
    # "machine" appears in 1 of 3 docs → higher IDF
    bm = fitted_bm25()
    assert bm._idf["machine"] > bm._idf["cat"]


# ---------------------------------------------------------------------------
# 6. Longer document is penalized relative to shorter (b normalization)
# ---------------------------------------------------------------------------


def test_longer_doc_penalized():
    # Two docs with identical TF for "cat" but different lengths
    corpus = [
        "cat",  # 1 token
        "cat " + "filler " * 50,  # 51 tokens, same TF for "cat"
    ]
    bm = BM25(b=1.0)  # full length normalization
    bm.fit(corpus)
    scores = bm.get_scores("cat")
    # The shorter document should score higher with b=1.0
    assert scores[0] > scores[1], f"Expected short doc ({scores[0]:.4f}) > long doc ({scores[1]:.4f})"


# ---------------------------------------------------------------------------
# 7. get_top_n results are sorted by score descending
# ---------------------------------------------------------------------------


def test_get_top_n_sorted_desc():
    bm = fitted_bm25()
    results = bm.get_top_n("the cat", n=3)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 8. get_top_n returns exactly n results
# ---------------------------------------------------------------------------


def test_get_top_n_length():
    bm = fitted_bm25()
    results = bm.get_top_n("cat", n=2)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# 9. OOV query term does not crash; contributes 0 to score
# ---------------------------------------------------------------------------


def test_unseen_query_term_no_crash():
    bm = fitted_bm25()
    scores = bm.get_scores("zzz_totally_unknown_token_xyz")
    assert all(s == 0.0 for s in scores)


# ---------------------------------------------------------------------------
# 10. Tokenizer lowercases text
# ---------------------------------------------------------------------------


def test_tokenize_lowercases():
    assert BM25._tokenize("Hello World") == BM25._tokenize("hello world")


# ---------------------------------------------------------------------------
# 11. Tokenizer strips punctuation
# ---------------------------------------------------------------------------


def test_tokenize_strips_punctuation():
    tokens = BM25._tokenize("word. another, word!")
    assert "word" in tokens
    assert "another" in tokens
    # No punctuation should remain
    for token in tokens:
        assert all(c not in ".,!?" for c in token)


# ---------------------------------------------------------------------------
# 12. Different k1/b values produce different scores
# ---------------------------------------------------------------------------


def test_k1_b_parameters_affect_scores():
    bm1 = BM25(k1=0.5, b=0.0)
    bm2 = BM25(k1=2.0, b=1.0)
    bm1.fit(SMALL_CORPUS)
    bm2.fit(SMALL_CORPUS)
    s1 = bm1.get_scores("the cat")
    s2 = bm2.get_scores("the cat")
    # Scores should differ with different hyperparameters
    assert not np.allclose(s1, s2), "Different k1/b should produce different scores"


# ---------------------------------------------------------------------------
# 13. get_top_n result dicts have required keys
# ---------------------------------------------------------------------------


def test_get_top_n_result_keys():
    bm = fitted_bm25()
    results = bm.get_top_n("cat", n=2)
    required = {"text", "score", "index"}
    for r in results:
        assert required.issubset(r.keys())


# ---------------------------------------------------------------------------
# 14. IDF formula is always positive (Robertson-Walker +1 invariant)
# ---------------------------------------------------------------------------


def test_idf_always_positive():
    # Fit on a corpus where one term appears in every document
    corpus = ["cat dog", "cat bird", "cat fish"]  # "cat" in all 3
    bm = BM25()
    bm.fit(corpus)
    assert bm._idf["cat"] > 0.0, "IDF must be positive even for universal terms"


# ---------------------------------------------------------------------------
# 15. get_scores before fit() raises RuntimeError
# ---------------------------------------------------------------------------


def test_get_scores_before_fit_raises():
    bm = BM25()
    with pytest.raises(RuntimeError, match="fit"):
        bm.get_scores("query")
