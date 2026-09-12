"""Tests for dense semantic ticket retrieval."""

from collections.abc import Iterable
from inspect import getsource
from typing import cast

import pytest

from ai_engineering_lab.embeddings import TextEmbedder
from ai_engineering_lab.retrieval import RetrievedTicket
from ai_engineering_lab.semantic_retrieval import SemanticTicketRetriever

HISTORICAL_TICKETS = [
    {"ticket_id": "T1", "text": "password reset"},
    {"ticket_id": "T2", "text": "invoice payment"},
    {"ticket_id": "T3", "text": "dashboard request"},
]


class FakeEmbedder:
    """Small deterministic embedder that records batch and query calls."""

    embedding_dimension = 3

    def __init__(self) -> None:
        self.batch_inputs: list[list[str]] = []
        self.single_inputs: list[str] = []
        self._vectors = {
            "password reset": [1.0, 0.0, 0.0],
            "invoice payment": [0.0, 1.0, 0.0],
            "dashboard request": [0.0, 0.0, 1.0],
            "authentication issue": [1.0, 0.0, 0.0],
            "zero query": [0.0, 0.0, 0.0],
        }

    def embed_text(self, text: str) -> list[float]:
        self.single_inputs.append(text)
        return self._vectors.get(text, [0.0, 0.0, 0.0])

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        self.batch_inputs.append(text_list)
        return [self._vectors.get(text, [0.0, 0.0, 0.0]) for text in text_list]


def make_retriever(fake_embedder: FakeEmbedder | None = None) -> SemanticTicketRetriever:
    """Build a retriever with a deterministic fake embedder."""
    fake_embedder = fake_embedder or FakeEmbedder()
    return SemanticTicketRetriever(HISTORICAL_TICKETS, cast(TextEmbedder, fake_embedder))


def test_empty_historical_tickets_raise_value_error() -> None:
    with pytest.raises(ValueError, match="historical_tickets must not be empty"):
        SemanticTicketRetriever([], cast(TextEmbedder, FakeEmbedder()))


def test_supplied_embedder_is_reused() -> None:
    fake_embedder = FakeEmbedder()

    retriever = make_retriever(fake_embedder)

    assert retriever.embedder is fake_embedder


def test_historical_embeddings_are_generated_in_one_batch() -> None:
    fake_embedder = FakeEmbedder()

    make_retriever(fake_embedder)

    assert fake_embedder.batch_inputs == [[ticket["text"] for ticket in HISTORICAL_TICKETS]]
    assert fake_embedder.single_inputs == []


def test_embedding_matrix_shape_and_dimension() -> None:
    retriever = make_retriever()

    assert retriever.embedding_matrix.shape == (3, 3)
    assert retriever.embedding_dimension == 3


def test_query_validation_rejects_non_string_input() -> None:
    retriever = make_retriever()

    with pytest.raises(TypeError, match="query must be a string"):
        retriever.search(123)  # type: ignore[arg-type]


def test_most_similar_ticket_is_first_and_metadata_is_preserved() -> None:
    retriever = make_retriever()

    results = retriever.search("authentication issue", top_k=1)

    assert isinstance(results[0], RetrievedTicket)
    assert results[0].ticket_id == "T1"
    assert results[0].text == "password reset"
    assert results[0].score == pytest.approx(1.0)
    assert isinstance(results[0].score, float)


def test_top_k_limits_results() -> None:
    assert len(make_retriever().search("authentication issue", top_k=2)) == 2


def test_top_k_zero_returns_empty_list() -> None:
    assert make_retriever().search("authentication issue", top_k=0) == []


def test_negative_top_k_raises_value_error() -> None:
    with pytest.raises(ValueError, match="top_k must be non-negative"):
        make_retriever().search("authentication issue", top_k=-1)


def test_zero_vector_query_returns_zero_scores() -> None:
    results = make_retriever().search("zero query")

    assert all(result.score == pytest.approx(0.0) for result in results)


def test_query_dimension_mismatch_raises_value_error() -> None:
    fake_embedder = FakeEmbedder()
    retriever = make_retriever(fake_embedder)
    fake_embedder._vectors["bad query"] = [1.0, 2.0]

    with pytest.raises(ValueError, match="match the embedding dimension"):
        retriever.search("bad query")


def test_stable_ranking_preserves_order_for_tied_scores() -> None:
    fake_embedder = FakeEmbedder()
    fake_embedder._vectors["tie query"] = [1.0, 1.0, 0.0]
    retriever = make_retriever(fake_embedder)

    results = retriever.search("tie query", top_k=3)

    assert [result.ticket_id for result in results] == ["T1", "T2", "T3"]


def test_module_does_not_use_tfidf_vectorizer() -> None:
    source = getsource(SemanticTicketRetriever)

    assert "TfidfVectorizer" not in source
