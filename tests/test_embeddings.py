"""
Unit tests for src/embeddings.py — EmbeddingModel.

NOTE: These tests load the actual all-MiniLM-L6-v2 model from HuggingFace.
The first run will download ~90MB. Subsequent runs use the local cache.

Run with: pytest tests/test_embeddings.py -v
"""

import numpy as np
import pytest

from src.embeddings import EmbeddingModel

# ---------------------------------------------------------------------------
# Fixture: shared model instance (loaded once per test session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    return EmbeddingModel()


# ---------------------------------------------------------------------------
# 1. Output shape is (N, 384)
# ---------------------------------------------------------------------------


def test_output_shape(model):
    texts = ["hello world", "foo bar baz", "another sentence"]
    result = model.embed_texts(texts)
    assert result.shape == (3, 384), f"Expected (3, 384), got {result.shape}"


# ---------------------------------------------------------------------------
# 2. Row norms are all ~1.0 (L2 normalized)
# ---------------------------------------------------------------------------


def test_l2_normalized(model):
    texts = ["normalize me", "and me too", "third sentence here"]
    result = model.embed_texts(texts)
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-5, err_msg="Embeddings are not L2-normalized")


# ---------------------------------------------------------------------------
# 3. Single text returns shape (1, 384)
# ---------------------------------------------------------------------------


def test_single_text(model):
    result = model.embed_texts(["hello"])
    assert result.shape == (1, 384)


# ---------------------------------------------------------------------------
# 4. embed_query returns 1D shape (384,)
# ---------------------------------------------------------------------------


def test_embed_query_shape(model):
    result = model.embed_query("what is machine learning?")
    assert result.shape == (384,), f"Expected (384,), got {result.shape}"


# ---------------------------------------------------------------------------
# 5. Batch results are consistent with single-item embedding
# ---------------------------------------------------------------------------


def test_batch_consistency(model):
    texts = ["apple", "banana", "cherry"]
    batch_result = model.embed_texts(texts)
    for i, text in enumerate(texts):
        single = model.embed_query(text)
        np.testing.assert_allclose(batch_result[i], single, atol=1e-5, err_msg=f"Batch vs single mismatch at index {i}")


# ---------------------------------------------------------------------------
# 6. Semantically similar texts produce higher cosine similarity
#    than semantically unrelated texts
# ---------------------------------------------------------------------------


def test_similar_texts_closer(model):
    # "cat" and "kitten" should be more similar than "cat" and "blockchain"
    vecs = model.embed_texts(["cat", "kitten", "blockchain technology"])
    sim_cat_kitten = float(np.dot(vecs[0], vecs[1]))
    sim_cat_blockchain = float(np.dot(vecs[0], vecs[2]))
    assert sim_cat_kitten > sim_cat_blockchain, (
        f"Expected cat~kitten ({sim_cat_kitten:.4f}) > cat~blockchain ({sim_cat_blockchain:.4f})"
    )


# ---------------------------------------------------------------------------
# 7. Empty list returns shape (0, 384) without crash
# ---------------------------------------------------------------------------


def test_empty_list(model):
    result = model.embed_texts([])
    assert result.shape == (0, 384)


# ---------------------------------------------------------------------------
# 8. Large batch with small batch_size produces same result as large batch_size
# ---------------------------------------------------------------------------


def test_large_batch_batching(model):
    texts = [f"sentence number {i}" for i in range(20)]
    result_small = model.embed_texts(texts, batch_size=4)
    result_large = model.embed_texts(texts, batch_size=20)
    np.testing.assert_allclose(
        result_small, result_large, atol=1e-5, err_msg="Different batch sizes produced different embeddings"
    )
