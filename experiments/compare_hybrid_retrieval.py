"""Compare lexical, semantic, and RRF hybrid ticket retrieval."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_engineering_lab.embeddings import TextEmbedder
from ai_engineering_lab.hybrid_retrieval import HybridTicketRetriever
from experiments.compare_retrieval import (
    build_demo_queries,
    build_demo_tickets,
    build_retrievers,
)

TOP_K = 3


def _print_results(label: str, results: list[tuple[str, float]]) -> None:
    """Print ticket IDs and scores for one retrieval strategy."""
    print(label)
    for rank, (ticket_id, score) in enumerate(results, start=1):
        print(f"  {rank}. {ticket_id}  score={score:.6f}")


def main() -> None:
    """Run the hybrid retrieval comparison experiment."""
    tickets = build_demo_tickets()
    embedder = TextEmbedder()
    lexical_retriever, semantic_retriever = build_retrievers(tickets, embedder)
    hybrid_retriever = HybridTicketRetriever(tickets, embedder=embedder)

    print(
        "RRF scores are only for ranking within the hybrid retriever; "
        "do not compare them directly with TF-IDF or embedding scores."
    )
    for query in build_demo_queries():
        lexical_results = lexical_retriever.search(query, top_k=TOP_K)
        semantic_results = semantic_retriever.search(query, top_k=TOP_K)
        hybrid_results = hybrid_retriever.search(query, top_k=TOP_K)
        print(f"\nQuery: {query}")
        _print_results(
            "TF-IDF lexical IDs and scores:",
            [(result.ticket_id, result.score) for result in lexical_results],
        )
        _print_results(
            "Dense semantic IDs and scores:",
            [(result.ticket_id, result.score) for result in semantic_results],
        )
        _print_results(
            "Hybrid IDs and RRF scores:",
            [(result.ticket_id, result.score) for result in hybrid_results],
        )


if __name__ == "__main__":
    main()
