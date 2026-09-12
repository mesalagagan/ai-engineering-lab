"""Quantitatively compare TF-IDF and semantic retrieval on demo tickets.

The relevance labels in this experiment are manually created illustrative ground
truth, not production annotations. Both systems use the same corpus, queries,
and top-k cutoff so their rankings can be evaluated under the same conditions.
Raw TF-IDF and embedding scores are intentionally not compared because they use
different score spaces; precision and recall compare retrieval usefulness.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_engineering_lab.embeddings import TextEmbedder
from ai_engineering_lab.evaluation import precision_at_k, recall_at_k
from ai_engineering_lab.retrieval import TicketRetriever
from ai_engineering_lab.semantic_retrieval import SemanticTicketRetriever
from experiments.compare_retrieval import (
    build_demo_queries,
    build_demo_tickets,
    build_retrievers,
    compare_query,
)

TOP_K = 3

# These labels are manually created demo ground truth, not production labels.
RELEVANT_TICKETS: dict[str, set[str]] = {
    "payment failed": {"T1", "T2", "T6"},
    "transaction declined": {"T1", "T2"},
    "I cannot log into my account": {"T3", "T4"},
    "I need to reset my password": {"T4", "T3"},
    "I want my money back": {"T5"},
    "checkout payment error": {"T1", "T6"},
}


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Precision and recall for one ranked result list."""

    precision: float
    recall: float


def calculate_metrics(
    retrieved_ticket_ids: list[str],
    relevant_ticket_ids: set[str],
    k: int = TOP_K,
) -> RetrievalMetrics:
    """Calculate precision@k and recall@k for one ranked result list."""
    return RetrievalMetrics(
        precision=precision_at_k(retrieved_ticket_ids, relevant_ticket_ids, k),
        recall=recall_at_k(retrieved_ticket_ids, relevant_ticket_ids, k),
    )


def evaluate_query(
    lexical_retriever: TicketRetriever,
    semantic_retriever: SemanticTicketRetriever,
    query: str,
    relevant_ticket_ids: set[str],
    k: int = TOP_K,
) -> tuple[list[str], RetrievalMetrics, list[str], RetrievalMetrics]:
    """Evaluate both retrievers for one query using the same cutoff."""
    lexical_results, semantic_results = compare_query(
        lexical_retriever,
        semantic_retriever,
        query,
        top_k=k,
    )
    lexical_ids = [result.ticket_id for result in lexical_results]
    semantic_ids = [result.ticket_id for result in semantic_results]
    return (
        lexical_ids,
        calculate_metrics(lexical_ids, relevant_ticket_ids, k),
        semantic_ids,
        calculate_metrics(semantic_ids, relevant_ticket_ids, k),
    )


def mean_metric(metrics: list[RetrievalMetrics], attribute: str) -> float:
    """Return the mean value of one metric attribute across evaluated queries."""
    if not metrics:
        raise ValueError("metrics must not be empty")
    if attribute == "precision":
        return mean(metric.precision for metric in metrics)
    if attribute == "recall":
        return mean(metric.recall for metric in metrics)
    raise ValueError("attribute must be precision or recall")


def _print_query_evaluation(
    query: str,
    relevant_ticket_ids: set[str],
    lexical_ids: list[str],
    lexical_metrics: RetrievalMetrics,
    semantic_ids: list[str],
    semantic_metrics: RetrievalMetrics,
) -> None:
    """Print one query's labels, IDs, and evaluation metrics."""
    print(f"\nQuery: {query}")
    print(f"Relevant ticket IDs: {sorted(relevant_ticket_ids)}")
    print(f"TF-IDF IDs: {lexical_ids}")
    print(
        f"  precision@{TOP_K}={lexical_metrics.precision:.4f}"
        f"  recall@{TOP_K}={lexical_metrics.recall:.4f}"
    )
    print(f"Semantic IDs: {semantic_ids}")
    print(
        f"  precision@{TOP_K}={semantic_metrics.precision:.4f}"
        f"  recall@{TOP_K}={semantic_metrics.recall:.4f}"
    )


def main() -> None:
    """Run the illustrative quantitative retrieval comparison."""
    tickets = build_demo_tickets()
    queries = build_demo_queries()
    embedder = TextEmbedder()
    # The same corpus and identical queries make the metric comparison fair.
    lexical_retriever, semantic_retriever = build_retrievers(tickets, embedder)
    lexical_metrics: list[RetrievalMetrics] = []
    semantic_metrics: list[RetrievalMetrics] = []

    for query in queries:
        # Precision measures returned relevance; recall measures found relevance.
        evaluation = evaluate_query(
            lexical_retriever,
            semantic_retriever,
            query,
            RELEVANT_TICKETS[query],
        )
        query_lexical_ids, query_lexical_metrics, query_semantic_ids, query_semantic_metrics = (
            evaluation
        )
        lexical_metrics.append(query_lexical_metrics)
        semantic_metrics.append(query_semantic_metrics)
        _print_query_evaluation(
            query,
            RELEVANT_TICKETS[query],
            query_lexical_ids,
            query_lexical_metrics,
            query_semantic_ids,
            query_semantic_metrics,
        )

    # This tiny hand-labeled dataset is illustrative, not a statistical benchmark.
    print("\nMean metrics across demo queries:")
    print(f"TF-IDF mean precision@{TOP_K}: {mean_metric(lexical_metrics, 'precision'):.4f}")
    print(f"TF-IDF mean recall@{TOP_K}: {mean_metric(lexical_metrics, 'recall'):.4f}")
    print(f"Semantic mean precision@{TOP_K}: {mean_metric(semantic_metrics, 'precision'):.4f}")
    print(f"Semantic mean recall@{TOP_K}: {mean_metric(semantic_metrics, 'recall'):.4f}")


if __name__ == "__main__":
    main()
