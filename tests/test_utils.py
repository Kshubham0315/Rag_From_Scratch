"""
Unit tests for src/utils.py — detect_device, top_k_indices, and shared helpers.
Run with: pytest tests/test_utils.py -v
"""

import tempfile

import numpy as np

from src.utils import detect_device, move_to_device, resolve_path, source_basename, top_k_indices

# ---------------------------------------------------------------------------
# detect_device
# ---------------------------------------------------------------------------


def test_detect_device_returns_string():
    result = detect_device()
    assert isinstance(result, str)
    assert result in {"cuda", "mps", "cpu"}


# ---------------------------------------------------------------------------
# top_k_indices — basic
# ---------------------------------------------------------------------------


def test_top_k_returns_correct_count():
    scores = np.array([0.1, 0.9, 0.5, 0.3])
    result = top_k_indices(scores, 2)
    assert len(result) == 2


def test_top_k_sorted_descending():
    scores = np.array([0.1, 0.9, 0.5, 0.3])
    result = top_k_indices(scores, 3)
    selected_scores = scores[result]
    assert list(selected_scores) == sorted(selected_scores, reverse=True)


def test_top_k_correct_indices():
    scores = np.array([0.1, 0.9, 0.5, 0.3])
    result = top_k_indices(scores, 2)
    assert result[0] == 1  # 0.9 is highest
    assert result[1] == 2  # 0.5 is second


# ---------------------------------------------------------------------------
# top_k_indices — edge cases
# ---------------------------------------------------------------------------


def test_top_k_zero():
    scores = np.array([0.1, 0.9, 0.5])
    result = top_k_indices(scores, 0)
    assert len(result) == 0
    assert result.dtype == np.intp


def test_top_k_equals_n():
    scores = np.array([0.3, 0.1, 0.9])
    result = top_k_indices(scores, 3)
    assert len(result) == 3
    assert result[0] == 2  # 0.9


def test_top_k_exceeds_n():
    scores = np.array([0.3, 0.1])
    result = top_k_indices(scores, 100)
    assert len(result) == 2


def test_top_k_single_element():
    scores = np.array([42.0])
    result = top_k_indices(scores, 1)
    assert len(result) == 1
    assert result[0] == 0


def test_top_k_empty_array():
    scores = np.array([])
    result = top_k_indices(scores, 5)
    assert len(result) == 0


def test_top_k_identical_scores():
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    result = top_k_indices(scores, 2)
    assert len(result) == 2
    # All scores equal, so any 2 indices are valid
    assert all(0 <= idx < 4 for idx in result)


def test_top_k_negative_scores():
    scores = np.array([-0.5, -0.1, -0.9, -0.3])
    result = top_k_indices(scores, 2)
    assert result[0] == 1  # -0.1 is highest
    assert result[1] == 3  # -0.3 is second


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------


def test_resolve_path_existing_file():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"hello")
        tmp = f.name
    result = resolve_path(tmp)
    assert result.exists()
    assert result.is_absolute()
    result.unlink()  # cleanup


def test_resolve_path_missing_file():
    import pytest

    with pytest.raises(FileNotFoundError, match="File not found"):
        resolve_path("/nonexistent/path/to/file.txt")


# ---------------------------------------------------------------------------
# source_basename
# ---------------------------------------------------------------------------


def test_source_basename_with_path():
    meta = {"source": "/home/user/docs/report.pdf"}
    assert source_basename(meta) == "report.pdf"


def test_source_basename_unknown():
    meta = {"source": "unknown"}
    assert source_basename(meta) == "unknown"


def test_source_basename_missing_key():
    assert source_basename({}) == "unknown"


def test_source_basename_windows_path():
    meta = {"source": "C:\\Users\\docs\\file.txt"}
    result = source_basename(meta)
    assert result == "file.txt"


# ---------------------------------------------------------------------------
# move_to_device
# ---------------------------------------------------------------------------


def test_move_to_device_cpu():
    import torch

    tensors = {"ids": torch.tensor([1, 2, 3]), "mask": torch.tensor([1, 1, 0])}
    result = move_to_device(tensors, "cpu")
    assert all(v.device.type == "cpu" for v in result.values())
    assert set(result.keys()) == {"ids", "mask"}


def test_move_to_device_empty_dict():
    result = move_to_device({}, "cpu")
    assert result == {}
