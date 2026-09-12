"""Compare lexical TF-IDF retrieval with dense semantic retrieval.

Both retrievers use the same support-ticket corpus and identical queries so the
experiment compares ranking behavior, not different data or different tasks.
Lexical and embedding scores use different scales, so rankings matter more than
comparing their raw score magnitudes. Semantic retrieval complements rather
than replaces lexical retrieval: exact IDs, error codes, and rare technical
terms may still benefit from lexical matching.
"""

from collections.abc import Iterable, Mapping

from ai_engineering_lab.embeddings import TextEmbedder
from ai_engineering_lab.retrieval import RetrievedTicket, TicketRetriever
from ai_engineering_lab.semantic_retrieval import SemanticTicketRetriever


def build_demo_tickets() -> list[dict[str, str]]:
    """Return the shared historical ticket corpus for the experiment."""
    return [
        {"ticket_id": "T1", "text": "The customer payment failed during checkout."},
        {
            "ticket_id": "T2",
            "text": "The transaction was declined when the user tried to pay.",
        },
        {"ticket_id": "T3", "text": "The customer cannot sign in to the application."},
        {"ticket_id": "T4", "text": "The user forgot their password and needs a reset."},
        {"ticket_id": "T5", "text": "The customer wants a refund for a recent purchase."},
        {
            "ticket_id": "T6",
            "text": "The checkout page shows an error after submitting payment.",
        },
    ]


def build_demo_queries() -> list[str]:
    """Return identical queries for both retrieval approaches."""
    return [
        "payment failed",
        "transaction declined",
        "I cannot log into my account",
        "I need to reset my password",
        "I want my money back",
        "checkout payment error",
    ]


def build_retrievers(
    historical_tickets: Iterable[Mapping[str, str]],
    embedder: TextEmbedder,
) -> tuple[TicketRetriever, SemanticTicketRetriever]:
    """Build both retrievers over one corpus using one shared embedder."""
    tickets = list(historical_tickets)
    lexical_retriever = TicketRetriever(tickets)
    semantic_retriever = SemanticTicketRetriever(tickets, embedder=embedder)
    return lexical_retriever, semantic_retriever


def compare_query(
    lexical_retriever: TicketRetriever,
    semantic_retriever: SemanticTicketRetriever,
    query: str,
    top_k: int = 3,
) -> tuple[list[RetrievedTicket], list[RetrievedTicket]]:
    """Return lexical and semantic results for the same query and cutoff."""
    return (
        lexical_retriever.search(query, top_k=top_k),
        semantic_retriever.search(query, top_k=top_k),
    )


def _print_results(label: str, results: list[RetrievedTicket]) -> None:
    """Print ranked ticket IDs and scores in a compact comparison format."""
    print(label)
    for rank, result in enumerate(results, start=1):
        print(f"  {rank}. {result.ticket_id}  score={result.score:.4f}  {result.text}")


def main() -> None:
    """Run the lexical-versus-semantic retrieval comparison."""
    tickets = build_demo_tickets()
    queries = build_demo_queries()
    embedder = TextEmbedder()
    lexical_retriever, semantic_retriever = build_retrievers(tickets, embedder)

    # Both retrievers receive the same corpus and query set for a fair comparison.
    for query in queries:
        lexical_results, semantic_results = compare_query(
            lexical_retriever,
            semantic_retriever,
            query,
            top_k=3,
        )
        print(f"\nQuery: {query}")
        _print_results("TF-IDF lexical retrieval:", lexical_results)
        _print_results("Dense semantic retrieval:", semantic_results)


if __name__ == "__main__":
    main()
