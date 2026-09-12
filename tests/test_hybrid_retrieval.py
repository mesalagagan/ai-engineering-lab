"""Tests for hybrid lexical and semantic ticket retrieval."""

from collections.abc import Iterable
from typing import cast
from unittest.mock import patch

import pytest

from ai_engineering_lab.embeddings import TextEmbedder
from ai_engineering_lab.hybrid_retrieval import HybridTicketRetriever
from ai_engineering_lab.retrieval import RetrievedTicket

HISTORICAL_TICKETS = [
    {"ticket_id": "T1", "text": "password reset"},
    {"ticket_id": "T2", "text": "invoice payment"},
    {"ticket_id": "T3", "text": "dashboard request"},
]


class FakeEmbedder:
    """Deterministic 3D embedder that avoids loading model weights."""

    embedding_dimension = 3

    def __init__(self) -> None:
        self._vectors = {
            "password reset": [1.0, 0.0, 0.0],
            "invoice payment": [0.0, 1.0, 0.0],
            "dashboard request": [0.0, 0.0, 1.0],
            "password query": [1.0, 0.0, 0.0],
            "unknown query": [0.0, 0.0, 0.0],
        }

    def embed_text(self, text: str) -> list[float]:
        return self._vectors.get(text, [0.0, 0.0, 0.0])

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        return [self._vectors.get(text, [0.0, 0.0, 0.0]) for text in texts]


def make_retriever() -> HybridTicketRetriever:
    """Build a hybrid retriever with deterministic fake embeddings."""
    return HybridTicketRetriever(
        HISTORICAL_TICKETS,
        embedder=cast(TextEmbedder, FakeEmbedder()),
    )


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError, match="historical_tickets must not be empty"):
        HybridTicketRetriever([], embedder=cast(TextEmbedder, FakeEmbedder()))


def test_invalid_rrf_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="rrf_k must be greater than zero"):
        HybridTicketRetriever(
            HISTORICAL_TICKETS,
            embedder=cast(TextEmbedder, FakeEmbedder()),
            rrf_k=0,
        )


def test_invalid_query_is_rejected() -> None:
    with pytest.raises(TypeError, match="query must be a string"):
        make_retriever().search(123)  # type: ignore[arg-type]


def test_negative_top_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="top_k must be non-negative"):
        make_retriever().search("password query", top_k=-1)


def test_top_k_zero_returns_empty_list() -> None:
    assert make_retriever().search("password query", top_k=0) == []


def test_ticket_in_both_rankings_receives_both_rrf_contributions() -> None:
    retriever = make_retriever()

    results = retriever.search("password query", top_k=3)

    assert results[0].ticket_id == "T1"
    assert results[0].score == pytest.approx(2 / (retriever.rrf_k + 1))
    assert results[0].score > results[1].score


def test_results_are_deterministic_for_tied_fused_scores() -> None:
    retriever = make_retriever()
    lexical_results = [
        RetrievedTicket("T1", HISTORICAL_TICKETS[0]["text"], 0.0),
        RetrievedTicket("T2", HISTORICAL_TICKETS[1]["text"], 0.0),
        RetrievedTicket("T3", HISTORICAL_TICKETS[2]["text"], 0.0),
    ]
    semantic_results = [
        RetrievedTicket("T3", HISTORICAL_TICKETS[2]["text"], 0.0),
        RetrievedTicket("T2", HISTORICAL_TICKETS[1]["text"], 0.0),
        RetrievedTicket("T1", HISTORICAL_TICKETS[0]["text"], 0.0),
    ]

    with (
        patch.object(retriever.lexical_retriever, "search", return_value=lexical_results),
        patch.object(retriever.semantic_retriever, "search", return_value=semantic_results),
    ):
        results = retriever.search("password query", top_k=3)

    assert [result.ticket_id for result in results] == ["T1", "T3", "T2"]


def test_returned_scores_are_rrf_scores() -> None:
    retriever = make_retriever()

    results = retriever.search("password query", top_k=3)

    assert results[0].score == pytest.approx(2 / (retriever.rrf_k + 1))
    assert results[1].score == pytest.approx(2 / (retriever.rrf_k + 2))
    assert all(isinstance(result.score, float) for result in results)


def test_retriever_properties_are_available() -> None:
    retriever = make_retriever()

    assert retriever.lexical_retriever is not None
    assert retriever.semantic_retriever is not None
    assert retriever.rrf_k == 60
