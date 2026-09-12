"""Tests for educational retrieval evaluation metrics."""

import pytest

from ai_engineering_lab.evaluation import precision_at_k, recall_at_k
from ai_engineering_lab.retrieval import TicketRetriever


@pytest.mark.parametrize(
    ("retrieved_ticket_ids", "relevant_ticket_ids", "expected"),
    [
        (["A", "B"], {"A", "B"}, 1.0),
        (["A", "B", "C", "D"], {"A", "C"}, 0.5),
        (["A", "B"], {"C"}, 0.0),
    ],
)
def test_precision_at_k_cases(
    retrieved_ticket_ids: list[str],
    relevant_ticket_ids: set[str],
    expected: float,
) -> None:
    assert precision_at_k(retrieved_ticket_ids, relevant_ticket_ids, 2) == expected


@pytest.mark.parametrize(
    ("retrieved_ticket_ids", "relevant_ticket_ids", "expected"),
    [
        (["A", "B"], {"A", "B"}, 1.0),
        (["A", "C", "B"], {"A", "B", "D"}, 2 / 3),
        (["A", "B"], {"C", "D"}, 0.0),
    ],
)
def test_recall_at_k_cases(
    retrieved_ticket_ids: list[str],
    relevant_ticket_ids: set[str],
    expected: float,
) -> None:
    assert recall_at_k(retrieved_ticket_ids, relevant_ticket_ids, 3) == expected


def test_k_limits_both_calculations() -> None:
    retrieved_ticket_ids = ["irrelevant", "relevant", "also-relevant"]
    relevant_ticket_ids = {"relevant", "also-relevant"}

    assert precision_at_k(retrieved_ticket_ids, relevant_ticket_ids, 1) == 0.0
    assert recall_at_k(retrieved_ticket_ids, relevant_ticket_ids, 1) == 0.0


def test_metrics_reject_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be greater than zero"):
        precision_at_k(["A"], {"A"}, 0)
    with pytest.raises(ValueError, match="k must be greater than zero"):
        recall_at_k(["A"], {"A"}, -1)


def test_recall_rejects_empty_relevant_ticket_set() -> None:
    with pytest.raises(ValueError, match="relevant_ticket_ids must not be empty"):
        recall_at_k(["A"], set(), 1)


def test_evaluates_ticket_retriever_results() -> None:
    historical_tickets = [
        {"ticket_id": "AUTH-1", "text": "password reset failed"},
        {"ticket_id": "AUTH-2", "text": "cannot reset account password"},
        {"ticket_id": "BILL-1", "text": "invoice payment failed"},
        {"ticket_id": "FEAT-1", "text": "feature request for dashboard"},
    ]
    retriever = TicketRetriever(historical_tickets)

    results = retriever.search("password reset problem", top_k=3)
    retrieved_ticket_ids = [result.ticket_id for result in results]
    relevant_ticket_ids = {"AUTH-1", "AUTH-2"}

    assert precision_at_k(retrieved_ticket_ids, relevant_ticket_ids, 3) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved_ticket_ids, relevant_ticket_ids, 3) == 1.0
