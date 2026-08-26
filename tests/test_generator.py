"""
Unit tests for src/generator.py — LLMGenerator.

The actual API call is mocked with unittest.mock so tests run offline
without a real API key.

Run with: pytest tests/test_generator.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from src.generator import LLMGenerator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_generator(**kwargs) -> LLMGenerator:
    defaults = {
        "api_key": "test-key",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    }
    defaults.update(kwargs)
    return LLMGenerator(**defaults)


def make_chunks(n: int = 3) -> list:
    return [
        {
            "text": f"Context text for chunk {i}.",
            "metadata": {
                "source": f"/path/to/document_{i}.pdf",
                "chunk_index": i,
            },
            "score": 0.9 - i * 0.1,
            "rerank_score": 0.8 - i * 0.1,
            "index": i,
        }
        for i in range(n)
    ]


def make_mock_response(answer: str = "The answer is 42.") -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "id": "chatcmpl-abc",
        "choices": [{"message": {"role": "assistant", "content": answer}}],
        "usage": {"prompt_tokens": 150, "completion_tokens": 30},
        "model": "gpt-4o-mini",
    }
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ---------------------------------------------------------------------------
# 1. build_prompt returns a system + user message pair
# ---------------------------------------------------------------------------


def test_build_prompt_structure():
    gen = make_generator()
    messages = gen.build_prompt("What is X?", make_chunks(2))
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


# ---------------------------------------------------------------------------
# 2. Sources are numbered [Source 1], [Source 2], ... in the user message
# ---------------------------------------------------------------------------


def test_build_prompt_sources_numbered():
    gen = make_generator()
    chunks = make_chunks(3)
    messages = gen.build_prompt("query", chunks)
    user_content = messages[1]["content"]
    assert "[Source 1]" in user_content
    assert "[Source 2]" in user_content
    assert "[Source 3]" in user_content


# ---------------------------------------------------------------------------
# 3. Empty chunk list produces a graceful prompt
# ---------------------------------------------------------------------------


def test_build_prompt_no_chunks():
    gen = make_generator()
    messages = gen.build_prompt("What is X?", [])
    user_content = messages[1]["content"]
    assert "no context" in user_content.lower() or "Context:" in user_content


# ---------------------------------------------------------------------------
# 4. _extract_sources returns dicts with required keys
# ---------------------------------------------------------------------------


def test_extract_sources_keys():
    chunks = make_chunks(2)
    sources = LLMGenerator._extract_sources(chunks)
    assert len(sources) == 2
    required = {"source_num", "filename", "chunk_index", "score"}
    for s in sources:
        assert required.issubset(s.keys()), f"Missing keys: {required - s.keys()}"


# ---------------------------------------------------------------------------
# 5. _extract_sources uses basename of source path
# ---------------------------------------------------------------------------


def test_extract_sources_basename():
    chunks = [
        {"text": "x", "metadata": {"source": "/long/path/to/my_file.pdf", "chunk_index": 0}, "score": 0.5, "index": 0}
    ]
    sources = LLMGenerator._extract_sources(chunks)
    assert sources[0]["filename"] == "my_file.pdf"


# ---------------------------------------------------------------------------
# 6. generate() calls requests.post exactly once
# ---------------------------------------------------------------------------


@patch("src.generator.requests.post")
def test_generate_calls_api_once(mock_post):
    mock_post.return_value = make_mock_response()
    gen = make_generator()
    gen.generate("query", make_chunks(2))
    assert mock_post.call_count == 1


# ---------------------------------------------------------------------------
# 7. generate() returns an "answer" key
# ---------------------------------------------------------------------------


@patch("src.generator.requests.post")
def test_generate_returns_answer_key(mock_post):
    mock_post.return_value = make_mock_response("The answer is 42.")
    gen = make_generator()
    result = gen.generate("query", make_chunks(2))
    assert "answer" in result
    assert result["answer"] == "The answer is 42."


# ---------------------------------------------------------------------------
# 8. generate() returns integer token counts
# ---------------------------------------------------------------------------


@patch("src.generator.requests.post")
def test_generate_returns_token_counts(mock_post):
    mock_post.return_value = make_mock_response()
    gen = make_generator()
    result = gen.generate("query", make_chunks(2))
    assert isinstance(result["prompt_tokens"], int)
    assert isinstance(result["completion_tokens"], int)
    assert result["prompt_tokens"] == 150
    assert result["completion_tokens"] == 30


# ---------------------------------------------------------------------------
# 9. HTTP 401 error propagates as requests.HTTPError
# ---------------------------------------------------------------------------


@patch("src.generator.requests.post")
def test_http_error_propagates(mock_post):
    import requests as req_lib

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = req_lib.HTTPError("401 Unauthorized")
    mock_post.return_value = mock_resp

    gen = make_generator()
    with pytest.raises(req_lib.HTTPError):
        gen.generate("query", make_chunks(1))


# ---------------------------------------------------------------------------
# 10. Timeout propagates as requests.Timeout
# ---------------------------------------------------------------------------


@patch("src.generator.requests.post")
def test_api_timeout_propagates(mock_post):
    import requests as req_lib

    mock_post.side_effect = req_lib.Timeout("Request timed out")

    gen = make_generator()
    with pytest.raises(req_lib.Timeout):
        gen.generate("query", make_chunks(1))


# ---------------------------------------------------------------------------
# 11. generate() result includes "sources" key
# ---------------------------------------------------------------------------


@patch("src.generator.requests.post")
def test_generate_includes_sources(mock_post):
    mock_post.return_value = make_mock_response()
    gen = make_generator()
    result = gen.generate("query", make_chunks(3))
    assert "sources" in result
    assert len(result["sources"]) == 3


# ---------------------------------------------------------------------------
# 12. The system message contains the key RAG instruction
# ---------------------------------------------------------------------------


def test_system_message_rag_contract():
    gen = make_generator()
    messages = gen.build_prompt("query", [])
    system_content = messages[0]["content"]
    assert "ONLY" in system_content or "only" in system_content
    assert "context" in system_content.lower()
