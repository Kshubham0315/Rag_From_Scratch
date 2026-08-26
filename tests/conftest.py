"""
Shared pytest fixtures for the RAG test suite.

Provides reusable document and chunk factories that match the
standardized dict shapes used throughout the pipeline.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def sample_document() -> dict:
    """A standardized document dict as returned by src/ingestion."""
    return {
        "text": "The quick brown fox jumps over the lazy dog.",
        "filename": "sample.txt",
        "num_pages": None,
        "metadata": {
            "source": "/tmp/sample.txt",
            "file_type": "txt",
            "pages": None,
        },
    }


@pytest.fixture()
def sample_chunks() -> list[dict]:
    """A list of chunk dicts as produced by the chunker + retriever pipeline."""
    return [
        {
            "text": "First chunk of text about RAG pipelines.",
            "metadata": {
                "source": "/docs/rag.pdf",
                "chunk_index": 0,
                "file_type": "pdf",
            },
            "score": 0.95,
        },
        {
            "text": "Second chunk about vector search.",
            "metadata": {
                "source": "/docs/rag.pdf",
                "chunk_index": 1,
                "file_type": "pdf",
            },
            "score": 0.82,
        },
        {
            "text": "Third chunk from a different document.",
            "metadata": {
                "source": "/notes/notes.txt",
                "chunk_index": 0,
                "file_type": "txt",
            },
            "score": 0.71,
        },
    ]
