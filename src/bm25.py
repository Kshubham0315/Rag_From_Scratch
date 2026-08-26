"""
BM25 sparse retrieval implemented from scratch.

Implements the Okapi BM25 ranking function using standard TF-IDF mathematics.
No external IR library is used — this file demonstrates the algorithm from
first principles for educational / portfolio purposes.

Formula:
    score(d, q) = Σ_{t in q}  IDF(t) * (tf(t,d) * (k1+1))
                               / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

    IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
             Robertson-Walker IDF — the outer +1 ensures IDF > 0 even for
             terms appearing in every document.
"""

from __future__ import annotations

import math
import string
from collections import Counter

import numpy as np

from src.utils import top_k_indices


class BM25:
    """
    Okapi BM25 ranking model.

    Args:
        k1: Term-frequency saturation parameter (standard default 1.5).
            Higher values give more weight to high-TF terms.
        b:  Document-length normalization parameter (standard default 0.75).
            b=1 fully normalizes by length; b=0 ignores length.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

        # Populated by fit()
        self._corpus: list[str] = []
        self._tf: list[dict[str, int]] = []  # term frequencies per document
        self._dl: list[int] = []  # document lengths (token counts)
        self._df: dict[str, int] = {}  # document frequency per term
        self._idf: dict[str, float] = {}  # precomputed IDF per term
        self._N: int = 0  # corpus size
        self._avgdl: float = 0.0
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, corpus: list[str]) -> None:
        """
        Tokenize and index the corpus.

        Args:
            corpus: List of document strings (one per chunk).
        """
        if not corpus:
            self._fitted = True
            return

        self._corpus = corpus
        self._N = len(corpus)
        tokenized = [self._tokenize(doc) for doc in corpus]
        self._dl = [len(tokens) for tokens in tokenized]
        self._avgdl = sum(self._dl) / self._N if self._N > 0 else 0.0

        # Term frequencies per document
        self._tf = [Counter(tokens) for tokens in tokenized]

        # Document frequencies: how many docs contain each term
        self._df = {}
        for token_count in self._tf:
            for term in token_count:
                self._df[term] = self._df.get(term, 0) + 1

        self._compute_idf()
        self._fitted = True

    def _compute_idf(self) -> None:
        """
        Precompute Robertson-Walker IDF for every term in the vocabulary.

        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

        The +1 outside the log guarantees IDF > 0 for all terms, including
        terms that appear in every document (where the inner fraction would
        be ~0 without it).
        """
        n = self._N
        for term, df in self._df.items():
            self._idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def get_scores(self, query: str) -> np.ndarray:
        """
        Compute BM25 scores for *all* documents against a query.

        Args:
            query: Raw query string (will be tokenized internally).

        Returns:
            np.ndarray of shape (N,) with a non-negative score per document.
        """
        if not self._fitted:
            raise RuntimeError("BM25.fit() must be called before get_scores()")

        if self._N == 0:
            return np.array([], dtype=np.float64)

        query_terms = self._tokenize(query)
        scores = np.zeros(self._N, dtype=np.float64)

        for term in query_terms:
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue  # unseen term contributes nothing

            for doc_idx, tf_dict in enumerate(self._tf):
                tf = tf_dict.get(term, 0)
                if tf == 0:
                    continue

                dl = self._dl[doc_idx]
                # BM25 TF component
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[doc_idx] += idf * (numerator / denominator)

        return scores

    def get_top_n(self, query: str, n: int) -> list[dict]:
        """
        Return the top-n scoring documents for a query.

        Returns:
            List of dicts: [{"text": str, "score": float, "index": int}]
            sorted by score descending.
        """
        scores = self.get_scores(query)
        if len(scores) == 0:
            return []

        top_idx = top_k_indices(scores, min(n, len(scores)))

        return [
            {
                "text": self._corpus[int(i)],
                "score": float(scores[i]),
                "index": int(i),
            }
            for i in top_idx
        ]

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Lowercase, strip punctuation, and whitespace-split.

        Intentionally simple — no stopword removal — to keep the
        implementation transparent and easy to reason about.
        """
        return text.lower().translate(str.maketrans("", "", string.punctuation)).split()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = f"fitted, N={self._N}" if self._fitted else "not fitted"
        return f"BM25(k1={self.k1}, b={self.b}, {status})"
