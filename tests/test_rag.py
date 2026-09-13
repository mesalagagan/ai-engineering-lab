"""Tests for the testable retrieval-augmented generation foundation."""

from dataclasses import FrozenInstanceError

import pytest

from ai_engineering_lab.rag import (
    ExtractiveAnswerGenerator,
    RAGContext,
    RAGPipeline,
    build_context,
)
from ai_engineering_lab.retrieval import RetrievedTicket

TICKETS = (
    RetrievedTicket("T1", "payment failed", 0.9),
    RetrievedTicket("T6", "checkout payment error", 0.8),
)


class FakeRetriever:
    """Retriever double that records the pipeline search call."""

    def __init__(self, results: tuple[RetrievedTicket, ...] = TICKETS) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 3) -> list[RetrievedTicket]:
        self.calls.append((query, top_k))
        return list(self.results[:top_k])


class FakeAnswerGenerator:
    """Answer-generator double that records the supplied query and context."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, query: str, context: str) -> str:
        self.calls.append((query, context))
        return "generated answer"


def test_build_context_validates_query_type() -> None:
    with pytest.raises(TypeError, match="query must be a string"):
        build_context(123, TICKETS)  # type: ignore[arg-type]


def test_build_context_preserves_order_and_excludes_scores() -> None:
    context = build_context("payment issue", TICKETS)

    assert isinstance(context, RAGContext)
    assert context.query == "payment issue"
    assert context.retrieved_tickets == TICKETS
    assert context.formatted_context == (
        "Ticket T1:\npayment failed\n\nTicket T6:\ncheckout payment error"
    )
    assert "0.9" not in context.formatted_context
    assert "0.8" not in context.formatted_context


def test_rag_context_is_immutable() -> None:
    context = build_context("payment issue", TICKETS)

    with pytest.raises(FrozenInstanceError):
        context.query = "changed"  # type: ignore[misc]


def test_pipeline_calls_retriever_and_passes_context_to_generator() -> None:
    retriever = FakeRetriever()
    generator = FakeAnswerGenerator()
    pipeline = RAGPipeline(retriever, generator)

    answer = pipeline.answer("payment issue", top_k=1)

    assert answer == "generated answer"
    assert retriever.calls == [("payment issue", 1)]
    assert generator.calls == [("payment issue", "Ticket T1:\npayment failed")]
    assert pipeline.retriever is retriever
    assert pipeline.answer_generator is generator


def test_pipeline_rejects_invalid_query_and_top_k() -> None:
    pipeline = RAGPipeline(FakeRetriever(), FakeAnswerGenerator())

    with pytest.raises(TypeError, match="query must be a string"):
        pipeline.answer(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="top_k must be non-negative"):
        pipeline.answer("payment issue", top_k=-1)


def test_top_k_zero_calls_retriever_and_generates_with_empty_context() -> None:
    retriever = FakeRetriever()
    generator = FakeAnswerGenerator()
    pipeline = RAGPipeline(retriever, generator)

    assert pipeline.answer("payment issue", top_k=0) == "generated answer"
    assert retriever.calls == [("payment issue", 0)]
    assert generator.calls == [("payment issue", "")]


def test_empty_retrieval_uses_extractive_fallback() -> None:
    pipeline = RAGPipeline(FakeRetriever(results=()), ExtractiveAnswerGenerator())

    assert pipeline.answer("unknown issue") == "I could not find supporting ticket information."


def test_retrieved_results_are_not_mutated() -> None:
    original_results = list(TICKETS)
    retriever = FakeRetriever()
    pipeline = RAGPipeline(retriever, FakeAnswerGenerator())

    pipeline.answer("payment issue")

    assert retriever.results == tuple(original_results)
    assert original_results == list(TICKETS)


def test_extractive_generator_returns_context_without_external_api() -> None:
    generator = ExtractiveAnswerGenerator()

    assert generator("payment issue", "Ticket T1:\npayment failed") == (
        "Based on the retrieved tickets:\nTicket T1:\npayment failed"
    )
