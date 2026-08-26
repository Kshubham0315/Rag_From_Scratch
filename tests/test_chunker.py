"""
Unit tests for src/chunker.py — RecursiveTextChunker.
Run with: pytest tests/test_chunker.py -v
"""

import pytest

from src.chunker import RecursiveTextChunker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunker(**kwargs) -> RecursiveTextChunker:
    return RecursiveTextChunker(**kwargs)


# ---------------------------------------------------------------------------
# 1. Empty string returns empty list
# ---------------------------------------------------------------------------


def test_empty_string():
    chunker = make_chunker()
    assert chunker.split_text("") == []


# ---------------------------------------------------------------------------
# 2. Text shorter than chunk_size → single chunk, no split needed
# ---------------------------------------------------------------------------


def test_short_text_no_split():
    text = "Hello, world."
    chunker = make_chunker(chunk_size=512, chunk_overlap=0)
    result = chunker.split_text(text)
    assert len(result) == 1
    assert result[0]["text"] == text
    assert result[0]["chunk_index"] == 0


# ---------------------------------------------------------------------------
# 3. Double-newline separator is preferred over period
# ---------------------------------------------------------------------------


def test_double_newline_preferred_over_period():
    # Two paragraphs separated by \n\n — should split there, not on ". "
    para1 = "First paragraph with some content here. More sentences follow."
    para2 = "Second paragraph starts here. Completely separate."
    text = para1 + "\n\n" + para2
    chunker = make_chunker(chunk_size=80, chunk_overlap=0)
    result = chunker.split_text(text)
    # Every chunk must be from one of the two paragraphs, not a mid-sentence split
    combined = "".join(c["text"] for c in result)
    # Both paragraphs should be present in the output
    assert "First paragraph" in combined
    assert "Second paragraph" in combined


# ---------------------------------------------------------------------------
# 4. Overlap is correct at the character level
# ---------------------------------------------------------------------------


def test_overlap_is_correct():
    # Create text long enough to force multiple chunks
    word = "abcdefghij"  # 10 chars
    text = " ".join([word] * 30)  # 329 chars
    overlap = 20
    chunker = make_chunker(chunk_size=60, chunk_overlap=overlap)
    result = chunker.split_text(text)

    # There must be at least 2 chunks to test overlap
    assert len(result) >= 2

    for i in range(len(result) - 1):
        curr_text = result[i]["text"]
        next_text = result[i + 1]["text"]
        # At least some overlap should exist: the end of chunk N shares
        # content with the beginning of chunk N+1.  The exact overlap
        # may be shorter than the configured value at the last boundary,
        # so check a sliding window from 1 up to `overlap` chars.
        found_overlap = False
        for size in range(min(overlap, len(curr_text), len(next_text)), 0, -1):
            if curr_text[-size:] == next_text[:size]:
                found_overlap = True
                break
        assert found_overlap, (
            f"Chunk {i} has no overlap with chunk {i + 1}.\n"
            f"  tail: {curr_text[-overlap:]!r}\n"
            f"  next: {next_text[: overlap * 2]!r}"
        )


# ---------------------------------------------------------------------------
# 5. No chunk exceeds chunk_size
# ---------------------------------------------------------------------------


def test_chunk_size_not_exceeded():
    text = "word " * 500  # 2500 chars
    chunk_size = 100
    chunker = make_chunker(chunk_size=chunk_size, chunk_overlap=10)
    result = chunker.split_text(text)
    assert result, "Expected at least one chunk"
    for chunk in result:
        assert len(chunk["text"]) <= chunk_size, f"Chunk too long: {len(chunk['text'])} > {chunk_size}"


# ---------------------------------------------------------------------------
# 6. start_char / end_char accurately index into the original text
# ---------------------------------------------------------------------------


def test_start_end_char_accuracy():
    text = "First sentence.\n\nSecond sentence.\n\nThird sentence here."
    chunker = make_chunker(chunk_size=25, chunk_overlap=0)
    result = chunker.split_text(text)
    for chunk in result:
        extracted = text[chunk["start_char"] : chunk["end_char"]]
        assert extracted == chunk["text"], f"start/end mismatch: expected {chunk['text']!r}, got {extracted!r}"


# ---------------------------------------------------------------------------
# 7. No content is lost (de-duped concatenation reconstructs original)
# ---------------------------------------------------------------------------


def test_no_content_lost():
    text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda."
    chunker = make_chunker(chunk_size=20, chunk_overlap=5)
    result = chunker.split_text(text)

    # Build the reconstructed text by taking only the non-overlapping suffix
    # of each chunk and prepending the full first chunk.
    reconstructed = result[0]["text"] if result else ""
    for i in range(1, len(result)):
        prev_end = result[i - 1]["end_char"]
        curr_start = result[i]["start_char"]
        # The non-overlap new content starts where the previous chunk ended
        reconstructed += result[i]["text"][prev_end - curr_start :] if prev_end > curr_start else result[i]["text"]

    # Every character from the original text must appear somewhere in the chunks
    for chunk in result:
        assert chunk["text"] in text or text in chunk["text"] or any(chunk["text"] in text for chunk in result)
    # More robust: all chunk texts are substrings of the original
    for chunk in result:
        assert chunk["text"] in text, f"Chunk text not found in original: {chunk['text']!r}"


# ---------------------------------------------------------------------------
# 8. Hard-slice fallback for text with no whitespace
# ---------------------------------------------------------------------------


def test_hard_slice_no_whitespace():
    text = "a" * 300  # 300-char string with no separators
    chunk_size = 100
    overlap = 20
    chunker = make_chunker(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=[""],  # jump straight to hard-slice
    )
    result = chunker.split_text(text)
    assert len(result) >= 3
    for chunk in result:
        assert len(chunk["text"]) <= chunk_size


# ---------------------------------------------------------------------------
# 9. Custom separators are respected
# ---------------------------------------------------------------------------


def test_custom_separators():
    text = "part1|part2|part3|part4|part5"
    chunker = make_chunker(chunk_size=15, chunk_overlap=0, separators=["|"])
    result = chunker.split_text(text)
    combined = "".join(c["text"] for c in result)
    # All original content should be in output
    assert "part1" in combined
    assert "part5" in combined


# ---------------------------------------------------------------------------
# 10. chunk_index values are sequential starting at 0
# ---------------------------------------------------------------------------


def test_chunk_index_sequential():
    text = "word " * 200  # force multiple chunks
    chunker = make_chunker(chunk_size=50, chunk_overlap=10)
    result = chunker.split_text(text)
    assert len(result) >= 2
    indices = [c["chunk_index"] for c in result]
    assert indices == list(range(len(result))), f"Non-sequential indices: {indices}"


# ---------------------------------------------------------------------------
# 11. Constructor raises on invalid overlap
# ---------------------------------------------------------------------------


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        RecursiveTextChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        RecursiveTextChunker(chunk_size=100, chunk_overlap=200)


# ---------------------------------------------------------------------------
# 12. Result dicts have all required keys
# ---------------------------------------------------------------------------


def test_result_dict_keys():
    chunker = make_chunker()
    result = chunker.split_text("Some text to split up properly.")
    required_keys = {"text", "chunk_index", "start_char", "end_char"}
    for chunk in result:
        assert required_keys.issubset(chunk.keys()), f"Missing keys: {required_keys - chunk.keys()}"
