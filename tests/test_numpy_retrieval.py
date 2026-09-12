"""Tests for NumPy-backed ticket retrieval."""

import numpy as np
import pytest

from ai_engineering_lab.numpy_retrieval import NumPyTicketRetriever
from ai_engineering_lab.retrieval import TicketRetriever

HISTORICAL_TICKETS = [
    {"ticket_id": "AUTH-1", "text": "password reset failed"},
    {"ticket_id": "AUTH-2", "text": "cannot reset account password"},
    {"ticket_id": "BILL-1", "text": "invoice payment failed"},
    {"ticket_id": "FEAT-1", "text": "feature request for dashboard"},
]


def test_ticket_matrix_has_one_row_per_ticket_and_feature_columns() -> None:
    retriever = NumPyTicketRetriever(HISTORICAL_TICKETS)

    assert isinstance(retriever.ticket_matrix, np.ndarray)
    assert retriever.ticket_matrix.ndim == 2
    assert retriever.ticket_matrix.shape == (
        len(HISTORICAL_TICKETS),
        len(retriever.vectorizer.vocabulary),
    )


def test_search_returns_most_similar_ticket_with_metadata_and_score() -> None:
    retriever = NumPyTicketRetriever(HISTORICAL_TICKETS)

    results = retriever.search("password reset problem", top_k=1)

    assert len(results) == 1
    assert results[0].ticket_id == "AUTH-1"
    assert results[0].text == "password reset failed"
    assert results[0].score == pytest.approx(0.8164965809)


def test_search_results_are_ordered_by_descending_score() -> None:
    retriever = NumPyTicketRetriever(HISTORICAL_TICKETS)

    results = retriever.search("password reset problem")

    assert [result.score for result in results] == sorted(
        (result.score for result in results), reverse=True
    )


def test_top_k_limits_results() -> None:
    retriever = NumPyTicketRetriever(HISTORICAL_TICKETS)

    assert len(retriever.search("password reset", top_k=2)) == 2


def test_top_k_zero_returns_empty_list() -> None:
    retriever = NumPyTicketRetriever(HISTORICAL_TICKETS)

    assert retriever.search("password reset", top_k=0) == []


def test_negative_top_k_raises_value_error() -> None:
    retriever = NumPyTicketRetriever(HISTORICAL_TICKETS)

    with pytest.raises(ValueError, match="top_k must be non-negative"):
        retriever.search("password reset", top_k=-1)


def test_zero_vector_query_returns_zero_scores_without_crashing() -> None:
    retriever = NumPyTicketRetriever(HISTORICAL_TICKETS)

    results = retriever.search("completely unknown phrase")

    assert len(results) == 3
    assert all(result.score == pytest.approx(0.0) for result in results)


def test_empty_historical_tickets_are_rejected() -> None:
    with pytest.raises(ValueError, match="historical_tickets must not be empty"):
        NumPyTicketRetriever([])


def test_numpy_retrieval_matches_existing_retriever_for_ordinary_inputs() -> None:
    python_retriever = TicketRetriever(HISTORICAL_TICKETS)
    numpy_retriever = NumPyTicketRetriever(HISTORICAL_TICKETS)

    python_results = python_retriever.search("password reset problem", top_k=3)
    numpy_results = numpy_retriever.search("password reset problem", top_k=3)

    assert [result.ticket_id for result in numpy_results] == [
        result.ticket_id for result in python_results
    ]
    assert [result.text for result in numpy_results] == [result.text for result in python_results]
    assert [result.score for result in numpy_results] == pytest.approx(
        [result.score for result in python_results]
    )
