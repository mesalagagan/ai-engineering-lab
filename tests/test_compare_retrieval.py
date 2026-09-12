"""Tests for the lexical-versus-semantic retrieval experiment."""

from collections.abc import Iterable
from typing import cast

from experiments.compare_retrieval import (
    build_demo_queries,
    build_demo_tickets,
    build_retrievers,
    compare_query,
)

from ai_engineering_lab.embeddings import TextEmbedder
from ai_engineering_lab.semantic_retrieval import SemanticTicketRetriever


class FakeEmbedder:
    """Deterministic semantic embedder for experiment helper tests."""

    embedding_dimension = 3

    def _embed(self, text: str) -> list[float]:
        normalized_text = text.casefold()
        if "payment" in normalized_text or "pay" in normalized_text:
            return [1.0, 0.0, 0.0]
        if "password" in normalized_text:
            return [0.0, 1.0, 0.0]
        if "sign" in normalized_text:
            return [0.0, 0.5, 0.5]
        if "refund" in normalized_text or "money" in normalized_text:
            return [0.0, 0.0, 1.0]
        return [0.0, 0.0, 0.0]

    def embed_text(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]


def test_demo_dataset_and_queries_are_small_and_reusable() -> None:
    assert len(build_demo_tickets()) == 6
    assert len(build_demo_queries()) == 6
    assert all(set(ticket) == {"ticket_id", "text"} for ticket in build_demo_tickets())


def test_both_retrievers_use_the_same_corpus_and_query() -> None:
    embedder = FakeEmbedder()
    lexical_retriever, semantic_retriever = build_retrievers(
        build_demo_tickets(),
        cast(TextEmbedder, embedder),
    )

    lexical_results, semantic_results = compare_query(
        lexical_retriever,
        semantic_retriever,
        "payment failed",
        top_k=3,
    )

    assert len(lexical_results) == 3
    assert len(semantic_results) == 3
    assert lexical_results[0].ticket_id == "T1"
    assert semantic_results[0].ticket_id == "T1"


def test_compare_query_returns_ranked_result_objects() -> None:
    _, semantic_retriever = build_retrievers(
        build_demo_tickets(),
        cast(TextEmbedder, FakeEmbedder()),
    )

    _, semantic_results = compare_query(
        build_retrievers(build_demo_tickets(), cast(TextEmbedder, FakeEmbedder()))[0],
        semantic_retriever,
        "I need to reset my password",
    )

    assert semantic_results[0].ticket_id == "T4"
    assert all(isinstance(result.score, float) for result in semantic_results)


def test_experiment_does_not_require_a_real_model_for_helpers() -> None:
    _, semantic_retriever = build_retrievers(
        build_demo_tickets(),
        cast(TextEmbedder, FakeEmbedder()),
    )

    assert isinstance(semantic_retriever, SemanticTicketRetriever)
