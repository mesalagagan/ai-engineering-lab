"""Tests for pure helpers in the retrieval evaluation experiment."""

import pytest
from experiments.evaluate_retrieval import (
    RELEVANT_TICKETS,
    RetrievalMetrics,
    calculate_metrics,
    mean_metric,
)


def test_relevant_ticket_labels_cover_all_demo_queries() -> None:
    assert set(RELEVANT_TICKETS) == {
        "payment failed",
        "transaction declined",
        "I cannot log into my account",
        "I need to reset my password",
        "I want my money back",
        "checkout payment error",
    }


def test_calculate_metrics_returns_precision_and_recall() -> None:
    metrics = calculate_metrics(["T1", "T3", "T2"], {"T1", "T2"}, k=3)

    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == 1.0


def test_mean_metric_calculates_the_requested_attribute() -> None:
    metrics = [
        RetrievalMetrics(precision=1.0, recall=0.5),
        RetrievalMetrics(precision=0.0, recall=1.0),
    ]

    assert mean_metric(metrics, "precision") == pytest.approx(0.5)
    assert mean_metric(metrics, "recall") == pytest.approx(0.75)


def test_mean_metric_rejects_empty_metrics() -> None:
    with pytest.raises(ValueError, match="metrics must not be empty"):
        mean_metric([], "precision")
