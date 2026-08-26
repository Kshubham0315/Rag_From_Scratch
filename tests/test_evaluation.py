"""
Unit tests for src/evaluation.py.

All tests use hand-crafted inputs with analytically known correct outputs.

Run with: pytest tests/test_evaluation.py -v
"""

from unittest.mock import MagicMock

from src.evaluation import (
    _compute_mrr,
    compute_faithfulness,
    compute_retrieval_metrics,
    run_evaluation_suite,
)

# ---------------------------------------------------------------------------
# compute_retrieval_metrics
# ---------------------------------------------------------------------------


def test_perfect_retrieval():
    retrieved = [0, 1, 2]
    relevant = [0, 1, 2]
    result = compute_retrieval_metrics(retrieved, relevant)
    assert result["precision_at_k"] == 1.0
    assert result["recall_at_k"] == 1.0
    assert result["mrr"] == 1.0


def test_zero_retrieval():
    retrieved = [5, 6, 7]
    relevant = [0, 1, 2]
    result = compute_retrieval_metrics(retrieved, relevant)
    assert result["precision_at_k"] == 0.0
    assert result["recall_at_k"] == 0.0
    assert result["mrr"] == 0.0


def test_partial_precision():
    # 2 of 5 retrieved are relevant → precision = 0.4
    retrieved = [0, 1, 99, 98, 97]
    relevant = [0, 1, 2, 3]
    result = compute_retrieval_metrics(retrieved, relevant, k=5)
    assert abs(result["precision_at_k"] - 0.4) < 1e-9


def test_partial_recall():
    # 2 of 4 relevant docs retrieved → recall = 0.5
    retrieved = [0, 1]
    relevant = [0, 1, 2, 3]
    result = compute_retrieval_metrics(retrieved, relevant)
    assert abs(result["recall_at_k"] - 0.5) < 1e-9


def test_mrr_first_hit_at_rank_3():
    # First relevant doc is at position 2 (0-based) → rank 3 → MRR = 1/3
    retrieved = [10, 11, 0]
    relevant = [0]
    result = compute_retrieval_metrics(retrieved, relevant)
    assert abs(result["mrr"] - 1.0 / 3) < 1e-9


def test_mrr_no_hit():
    retrieved = [5, 6, 7]
    relevant = [0, 1]
    result = compute_retrieval_metrics(retrieved, relevant)
    assert result["mrr"] == 0.0


def test_f1_harmonic_mean():
    # P = 0.5, R = 1.0 → F1 = 2*0.5*1.0/(0.5+1.0) = 2/3
    retrieved = [0, 99]  # 1 hit, 1 miss → P = 0.5
    relevant = [0]  # 1 relevant → R = 1.0
    result = compute_retrieval_metrics(retrieved, relevant)
    expected_f1 = 2 * 0.5 * 1.0 / (0.5 + 1.0)
    assert abs(result["f1_at_k"] - expected_f1) < 1e-9


def test_k_cutoff_applied():
    # k=2: only first 2 of [0,1,99,98,97] are evaluated
    retrieved = [0, 1, 99, 98, 97]
    relevant = [0, 1, 2]
    result = compute_retrieval_metrics(retrieved, relevant, k=2)
    # Precision: 2/2 = 1.0
    assert abs(result["precision_at_k"] - 1.0) < 1e-9
    # Recall: 2/3
    assert abs(result["recall_at_k"] - 2.0 / 3) < 1e-9


def test_empty_retrieved():
    result = compute_retrieval_metrics([], [0, 1, 2])
    assert result["precision_at_k"] == 0.0
    assert result["recall_at_k"] == 0.0
    assert result["mrr"] == 0.0


def test_empty_relevant():
    result = compute_retrieval_metrics([0, 1], [])
    # No relevant docs: recall undefined, set to 0
    assert result["recall_at_k"] == 0.0


# ---------------------------------------------------------------------------
# _compute_mrr
# ---------------------------------------------------------------------------


def test_mrr_first_position():
    assert _compute_mrr([0, 1, 2], {0}) == 1.0


def test_mrr_second_position():
    assert abs(_compute_mrr([9, 0, 1], {0}) - 0.5) < 1e-9


def test_mrr_helper_no_hit():
    assert _compute_mrr([5, 6], {0, 1}) == 0.0


# ---------------------------------------------------------------------------
# compute_faithfulness
# ---------------------------------------------------------------------------


def test_faithfulness_calls_judge_fn():
    judge = MagicMock(return_value=0.9)
    score = compute_faithfulness("answer text", ["context chunk"], judge)
    judge.assert_called_once_with("answer text", ["context chunk"])
    assert abs(score - 0.9) < 1e-9


def test_faithfulness_empty_answer():
    judge = MagicMock(return_value=1.0)
    score = compute_faithfulness("", ["context"], judge)
    assert score == 0.0
    judge.assert_not_called()


def test_faithfulness_returns_float():
    judge = MagicMock(return_value=0.75)
    score = compute_faithfulness("some answer", ["ctx"], judge)
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# run_evaluation_suite
# ---------------------------------------------------------------------------


def test_run_suite_aggregates():
    test_cases = [
        {"query": "q1", "relevant_ids": [0, 1], "reference_answer": "ans1"},
        {"query": "q2", "relevant_ids": [2], "reference_answer": "ans2"},
    ]

    def pipeline_fn(query):
        if query == "q1":
            return {"retrieved_ids": [0, 1], "answer": "answer 1"}
        else:
            return {"retrieved_ids": [2], "answer": "answer 2"}

    result = run_evaluation_suite(test_cases, pipeline_fn)
    # Both cases have perfect retrieval → mean_precision = 1.0
    assert abs(result["mean_precision"] - 1.0) < 1e-9
    assert abs(result["mean_recall"] - 1.0) < 1e-9
    assert abs(result["mean_mrr"] - 1.0) < 1e-9


def test_run_suite_per_case_length():
    test_cases = [{"query": f"q{i}", "relevant_ids": [i]} for i in range(5)]

    def pipeline_fn(q):
        return {"retrieved_ids": [], "answer": ""}

    result = run_evaluation_suite(test_cases, pipeline_fn)
    assert len(result["per_case"]) == 5


def test_run_suite_empty_test_cases():
    result = run_evaluation_suite([], lambda q: {"retrieved_ids": [], "answer": ""})
    assert result["mean_precision"] == 0.0
    assert result["per_case"] == []


def test_run_suite_partial_results():
    test_cases = [
        {"query": "q1", "relevant_ids": [0, 1]},  # retrieves 1/2 → recall=0.5
        {"query": "q2", "relevant_ids": [5]},  # retrieves 0/1 → recall=0.0
    ]

    def pipeline_fn(query):
        if query == "q1":
            return {"retrieved_ids": [0], "answer": ""}
        return {"retrieved_ids": [99], "answer": ""}

    result = run_evaluation_suite(test_cases, pipeline_fn)
    assert abs(result["mean_recall"] - 0.25) < 1e-9  # (0.5 + 0.0) / 2
