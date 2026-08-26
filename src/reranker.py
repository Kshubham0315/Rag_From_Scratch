"""
Cross-encoder re-ranking using a lightweight MS-MARCO cross-encoder model.

Why cross-encoder?
  - Bi-encoders (like embeddings.py) encode query and document independently,
    enabling fast pre-computation but sacrificing some accuracy.
  - Cross-encoders receive [CLS] query [SEP] document [SEP] as a single input,
    allowing full attention between every query and document token.
  - This is 5–10× more accurate at relevance scoring but requires one forward
    pass per candidate — O(candidates) not O(1) — so it's used only on the
    small set of candidates retrieved by the hybrid retriever.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Trained on MS-MARCO passage ranking (binary relevance)
  - ~22M parameters — fast enough for real-time re-ranking of 5–20 candidates
  - Raw logit output: higher → more relevant (no softmax needed for ranking)
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification

from src.utils import load_hf_model, move_to_device


class CrossEncoderReranker:
    """
    Re-ranks a list of retrieved documents using a cross-encoder model.

    Args:
        model_name: HuggingFace model ID for the cross-encoder.
        device:     "cuda", "mps", "cpu", or None for auto-detection.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.tokenizer, self.model, self.device = load_hf_model(model_name, AutoModelForSequenceClassification, device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int | None = None,
        batch_size: int = 16,
    ) -> list[dict]:
        """
        Re-score and sort a list of retrieved document dicts.

        Adds a "rerank_score" key (raw cross-encoder logit) to each dict.
        All original keys (text, metadata, score, index) are preserved.

        Args:
            query:      Raw query string.
            documents:  List of result dicts from the retriever.
            top_k:      If given, return only the top_k highest-scored docs.
            batch_size: Max (query, doc) pairs per forward pass.

        Returns:
            List of dicts sorted by rerank_score descending.
        """
        if not documents:
            return []

        texts = [doc["text"] for doc in documents]
        scores = self._score_pairs(query, texts, batch_size=batch_size)

        reranked = []
        for doc, score in zip(documents, scores, strict=True):
            reranked.append({**doc, "rerank_score": float(score)})

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked

    # ------------------------------------------------------------------
    # Pair scoring
    # ------------------------------------------------------------------

    def _score_pairs(
        self,
        query: str,
        texts: list[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Score (query, text) pairs with the cross-encoder.

        Input format: the tokenizer automatically produces:
            [CLS] query [SEP] document [SEP]

        Returns:
            np.ndarray of shape (N,), float32 raw logits.
            Higher logit = more relevant (no softmax — ranking only).
        """
        all_scores: list[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            queries = [query] * len(batch_texts)

            encoded = self.tokenizer(
                queries,
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = move_to_device(encoded, self.device)

            with torch.no_grad():
                logits = self.model(**encoded).logits

            # logits shape: (B, 1) or (B,) depending on model config
            batch_scores = logits.squeeze(-1).cpu().float().numpy()
            all_scores.append(np.atleast_1d(batch_scores))

        if not all_scores:
            return np.array([], dtype=np.float32)
        return np.concatenate(all_scores).astype(np.float32)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"CrossEncoderReranker(model={self.model_name!r}, device={self.device!r})"
