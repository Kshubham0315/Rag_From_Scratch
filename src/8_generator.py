"""
LLM answer generator — raw HTTP call to any OpenAI-compatible API.

No SDK wrapper. Uses only the standard `requests` library to call the
/chat/completions endpoint directly, demonstrating the raw API mechanics.

Prompt design:
  - System message establishes the RAG contract (answer ONLY from context,
    cite sources as [Source N])
  - User message contains numbered context blocks followed by the question

Source attribution:
  - Each chunk is labeled [Source N] (filename, chunk K) in the prompt
  - The generator returns a sources list extracted from chunk metadata so
    the UI can display exactly which chunks were cited
"""

from __future__ import annotations

import requests

from src.utils import source_basename


class LLMGenerator:
    """
    Generates grounded answers by calling an OpenAI-compatible chat API.

    Args:
        api_key:     Bearer token for the API.
        base_url:    Base URL of the API (e.g. "https://api.openai.com/v1").
                     Any OpenAI-compatible endpoint works (Together, Groq, etc.).
        model:       Model identifier (e.g. "gpt-4o-mini").
        temperature: Sampling temperature (lower = more deterministic).
        max_tokens:  Maximum tokens to generate.
        timeout:     HTTP request timeout in seconds.
    """

    SYSTEM_PROMPT = (
        "You are a precise question-answering assistant. "
        "Answer questions using ONLY the provided context. "
        "If the context does not contain enough information to answer, "
        "say \"I don't have enough information in the provided documents "
        'to answer this question." '
        "Do not speculate or use knowledge outside the provided context. "
        "Cite sources using [Source N] notation where N matches the context "
        "block number."
    )

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_prompt(self, query: str, chunks: list[dict]) -> list[dict]:
        """
        Build a chat messages list from a query and retrieved chunks.

        Returns:
            List of {"role": ..., "content": ...} dicts.
        """
        context_blocks: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata", {})
            filename = source_basename(meta)
            chunk_idx = meta.get("chunk_index", "?")
            header = f"[Source {i}] ({filename}, chunk {chunk_idx}):"
            context_blocks.append(f"{header}\n{chunk['text']}")

        context_str = "\n\n".join(context_blocks) if context_blocks else "(no context provided)"

        user_content = f"Context:\n\n{context_str}\n\nQuestion: {query}"

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def generate(self, query: str, chunks: list[dict]) -> dict:
        """
        Generate an answer for *query* grounded in *chunks*.

        Returns:
            {
                "answer":            str,
                "sources":           list[dict],  # [{filename, chunk_index, score}]
                "prompt_tokens":     int,
                "completion_tokens": int,
                "model":             str,
            }
        """
        messages = self.build_prompt(query, chunks)
        response_data = self._call_api(messages)

        choice = response_data["choices"][0]
        answer = choice["message"]["content"]

        usage = response_data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        sources = self._extract_sources(chunks)

        return {
            "answer": answer,
            "sources": sources,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "model": response_data.get("model", self.model),
        }

    # ------------------------------------------------------------------
    # Raw HTTP call
    # ------------------------------------------------------------------

    def _call_api(self, messages: list[dict]) -> dict:
        """
        POST to {base_url}/chat/completions.

        Raises:
            requests.HTTPError:  on 4xx / 5xx responses.
            requests.Timeout:    if the request exceeds self.timeout seconds.
            requests.ConnectionError: on network failures.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Source attribution
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sources(chunks: list[dict]) -> list[dict]:
        """
        Build a source attribution list from chunk metadata.

        Returns one dict per chunk with keys:
            filename    – basename of the source document
            chunk_index – sequential chunk number within the document
            score       – retrieval / rerank score (whichever is available)
        """
        sources = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata", {})
            filename = source_basename(meta)

            # Prefer rerank_score > retrieval score > 0
            score = chunk.get("rerank_score", chunk.get("score", 0.0))

            sources.append(
                {
                    "source_num": i,
                    "filename": filename,
                    "chunk_index": meta.get("chunk_index", "?"),
                    "score": round(float(score), 6),
                }
            )
        return sources

    def __repr__(self) -> str:
        return f"LLMGenerator(model={self.model!r}, base_url={self.base_url!r})"
