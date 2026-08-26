"""
Retrieval and generation evaluation metrics.

Retrieval metrics (computed without an LLM):
    precision@k  — fraction of retrieved docs that are relevant
    recall@k     — fraction of relevant docs that were retrieved
    f1@k         — harmonic mean of precision and recall
    mrr          — Mean Reciprocal Rank (1 / rank of first relevant doc)

Generation metrics (require an LLM judge function):
    faithfulness — does the answer stay within the provided context?
                   Implemented as a thin wrapper around a caller-supplied
                   judge function so the judge can be swapped out
                   (LLM-as-judge, rule-based, NLI model, etc.)

Design note:
    These functions accept plain Python lists of IDs (int or str), not
    any framework-specific types. The evaluation harness wires them to
    the pipeline via a `pipeline_fn` callable that maps query → result dict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def compute_retrieval_metrics(
    retrieved_ids: list[Union[int, str]],
    relevant_ids: list[Union[int, str]],
    k: int | None = None,
) -> dict[str, float]:
    """
    Compute precision, recall, F1, and MRR for a single query.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs (rank 1 first).
        relevant_ids:  Set of ground-truth relevant document IDs.
        k:             Cutoff. If None, uses len(retrieved_ids).

    Returns:
        {
            "precision_at_k": float,
            "recall_at_k":    float,
            "f1_at_k":        float,
            "mrr":            float,
        }
    """
    if k is None:
        k = len(retrieved_ids)

    retrieved_at_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)

    hits = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)

    precision = hits / k if k > 0 else 0.0
    recall = hits / len(relevant_set) if len(relevant_set) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    mrr = _compute_mrr(retrieved_at_k, relevant_set)

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "f1_at_k": f1,
        "mrr": mrr,
    }


def _compute_mrr(
    retrieved_ids: list[Union[int, str]],
    relevant_set: set,
) -> float:
    """
    Compute Reciprocal Rank for a single query.

    MRR = 1 / rank_of_first_relevant_document
    Returns 0.0 if no relevant document is found in retrieved_ids.
    """
    for rank_0based, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (rank_0based + 1)  # 1-based rank
    return 0.0


# ---------------------------------------------------------------------------
# Generation faithfulness
# ---------------------------------------------------------------------------


def compute_faithfulness(
    answer: str,
    context: list[str],
    judge_fn: Callable[[str, list[str]], float],
) -> float:
    """
    Score how faithfully the answer is grounded in the provided context.

    This is a typed wrapper — the actual scoring logic lives in `judge_fn`
    so callers can plug in any judge (LLM-as-judge, NLI model, etc.).

    Args:
        answer:   The generated answer string.
        context:  List of context chunk strings used to generate the answer.
        judge_fn: Callable(answer, context) → float in [0, 1].
                  1.0 = fully faithful, 0.0 = hallucinated / not grounded.

    Returns:
        float score from judge_fn.
    """
    if not answer:
        return 0.0
    return float(judge_fn(answer, context))


# ---------------------------------------------------------------------------
# Evaluation suite
# ---------------------------------------------------------------------------


def run_evaluation_suite(
    test_cases: list[dict[str, Any]],
    pipeline_fn: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    """
    Run the full evaluation suite over a set of labelled test cases.

    Args:
        test_cases: List of dicts, each containing:
                    {
                        "query":          str,
                        "relevant_ids":   list[int | str],
                        "reference_answer": str  (optional, for future use)
                    }
        pipeline_fn: Callable(query) → {"retrieved_ids": list, "answer": str}
                     The pipeline under evaluation.

    Returns:
        {
            "mean_precision": float,
            "mean_recall":    float,
            "mean_f1":        float,
            "mean_mrr":       float,
            "per_case":       list[dict],  # metrics for each test case
        }
    """
    per_case: list[dict[str, Any]] = []

    for case in test_cases:
        query = case["query"]
        relevant_ids = case["relevant_ids"]

        pipeline_result = pipeline_fn(query)
        retrieved_ids = pipeline_result.get("retrieved_ids", [])

        metrics = compute_retrieval_metrics(retrieved_ids, relevant_ids)
        per_case.append(
            {
                "query": query,
                **metrics,
            }
        )

    n = len(per_case)
    if n == 0:
        return {
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "mean_f1": 0.0,
            "mean_mrr": 0.0,
            "per_case": [],
        }

    return {
        "mean_precision": sum(c["precision_at_k"] for c in per_case) / n,
        "mean_recall": sum(c["recall_at_k"] for c in per_case) / n,
        "mean_f1": sum(c["f1_at_k"] for c in per_case) / n,
        "mean_mrr": sum(c["mrr"] for c in per_case) / n,
        "per_case": per_case,
    }
