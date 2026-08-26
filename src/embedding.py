"""
Sentence embedding using sentence-transformers/all-MiniLM-L6-v2 loaded directly
via HuggingFace `transformers` — no sentence-transformers library required.

Key mechanics demonstrated:
  - Manual mean pooling with attention-mask weighting
  - L2 normalization for cosine-similarity-via-dot-product compatibility
  - Batched inference with torch.no_grad()
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel

from src.utils import load_hf_model, move_to_device


class EmbeddingModel:
    """
    Bi-encoder sentence embedder backed by all-MiniLM-L6-v2.

    Outputs L2-normalized float32 vectors of dimension 384.

    Args:
        model_name: Any HuggingFace model ID that supports mean-pooling
                    (e.g. sentence-transformers/all-MiniLM-L6-v2).
        device:     "cuda", "mps", "cpu", or None for auto-detection.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.tokenizer, self.model, self.device = load_hf_model(model_name, AutoModel, device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of strings in batches.

        Returns:
            np.ndarray of shape (N, 384), float32, L2-normalized.
        """
        if not texts:
            return np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)

        all_embeddings: list[np.ndarray] = []

        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        if show_progress:
            try:
                from tqdm import tqdm

                batches = tqdm(batches, desc="Embedding")
            except ImportError:
                pass

        for batch in batches:
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = move_to_device(encoded, self.device)

            with torch.no_grad():
                output = self.model(**encoded)

            pooled = self._mean_pool(
                output.last_hidden_state,
                encoded["attention_mask"],
            )
            all_embeddings.append(pooled.cpu().float().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        return self._l2_normalize(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns:
            np.ndarray of shape (384,), float32, L2-normalized.
        """
        return self.embed_texts([query])[0]

    # ------------------------------------------------------------------
    # Mean pooling
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_pool(
        token_embeddings: torch.Tensor,  # (B, T, D)
        attention_mask: torch.Tensor,  # (B, T)
    ) -> torch.Tensor:  # (B, D)
        """
        Average token embeddings weighted by the attention mask.

        Padding tokens (mask == 0) are excluded from the average.
        This is the standard approach for sentence-transformers models —
        NOT the [CLS] token, which carries single-token context only.
        """
        # Expand mask to (B, T, D) to broadcast against token embeddings
        mask_expanded = (
            attention_mask.unsqueeze(-1)  # (B, T, 1)
            .expand_as(token_embeddings)  # (B, T, D)
            .float()
        )
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)  # (B, D)
        # Clamp denominator to avoid division by zero on all-padding rows
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # (B, D)
        return sum_embeddings / sum_mask

    # ------------------------------------------------------------------
    # L2 normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
        """
        Divide each row by its L2 norm.

        After normalization, cosine_similarity(a, b) == np.dot(a, b),
        which reduces all similarity search to a single matrix multiply.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)  # prevent division by zero
        return (embeddings / norms).astype(np.float32)

    def __repr__(self) -> str:
        return f"EmbeddingModel(model={self.model_name!r}, device={self.device!r})"
