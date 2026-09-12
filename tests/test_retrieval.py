"""Tests for reusable historical ticket retrieval."""

from unittest.mock import patch

import pytest

from ai_engineering_lab.retrieval import RetrievedTicket, TicketRetriever

HISTORICAL_TICKETS = [
    {"ticket_id": "AUTH-1", "text": "password reset failed"},
    {"ticket_id": "AUTH-2", "text": "cannot reset account password"},
    {"ticket_id": "BILL-1", "text": "invoice payment failed"},
    {"ticket_id": "FEAT-1", "text": "feature request for dashboard"},
]


def test_historical_tickets_are_indexed_during_initialization() -> None:
    retriever = TicketRetriever(HISTORICAL_TICKETS)

    assert retriever.vectorizer.vocabulary
    assert len(retriever._ticket_vectors) == len(HISTORICAL_TICKETS)


def test_search_returns_preserved_retrieved_ticket_objects() -> None:
    retriever = TicketRetriever(HISTORICAL_TICKETS)

    results = retriever.search("password reset problem", top_k=1)

    assert isinstance(results[0], RetrievedTicket)
    assert results[0].ticket_id == "AUTH-1"
    assert results[0].text == "password reset failed"


def test_search_results_are_ordered_by_descending_score() -> None:
    retriever = TicketRetriever(HISTORICAL_TICKETS)

    results = retriever.search("password reset problem")

    assert [result.score for result in results] == sorted(
        (result.score for result in results), reverse=True
    )


def test_top_k_limits_result_count() -> None:
    retriever = TicketRetriever(HISTORICAL_TICKETS)

    assert len(retriever.search("account problem", top_k=2)) == 2


def test_top_k_zero_returns_no_results() -> None:
    retriever = TicketRetriever(HISTORICAL_TICKETS)

    assert retriever.search("account problem", top_k=0) == []


def test_negative_top_k_raises_value_error() -> None:
    retriever = TicketRetriever(HISTORICAL_TICKETS)

    with pytest.raises(ValueError, match="top_k must be non-negative"):
        retriever.search("account problem", top_k=-1)


def test_search_reuses_the_fitted_vectorizer() -> None:
    retriever = TicketRetriever(HISTORICAL_TICKETS)

    with patch.object(retriever.vectorizer, "fit", wraps=retriever.vectorizer.fit) as fit:
        retriever.search("password reset")
        retriever.search("invoice payment")

    fit.assert_not_called()


def test_unknown_query_terms_do_not_crash() -> None:
    retriever = TicketRetriever(HISTORICAL_TICKETS)

    results = retriever.search("completely unknown phrase", top_k=2)

    assert len(results) == 2
    assert all(result.score == 0.0 for result in results)
