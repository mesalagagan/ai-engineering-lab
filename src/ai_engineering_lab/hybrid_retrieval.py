"""Hybrid ticket retrieval with Reciprocal Rank Fusion."""

from collections.abc import Iterable, Mapping

from ai_engineering_lab.embeddings import TextEmbedder
from ai_engineering_lab.retrieval import RetrievedTicket, TicketRetriever
from ai_engineering_lab.semantic_retrieval import SemanticTicketRetriever


class HybridTicketRetriever:
    """Combine lexical and semantic rankings into one ticket ranking."""

    def __init__(
        self,
        historical_tickets: Iterable[Mapping[str, str]],
        embedder: TextEmbedder | None = None,
        rrf_k: int = 60,
    ) -> None:
        """Build lexical and semantic indexes over the same ticket corpus."""
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")

        self._historical_tickets = [
            {"ticket_id": ticket["ticket_id"], "text": ticket["text"]}
            for ticket in historical_tickets
        ]
        if not self._historical_tickets:
            raise ValueError("historical_tickets must not be empty")

        self._rrf_k = rrf_k
        self._lexical_retriever = TicketRetriever(self._historical_tickets)
        self._semantic_retriever = SemanticTicketRetriever(
            self._historical_tickets,
            embedder=embedder,
        )
        self._ticket_order = {
            ticket["ticket_id"]: index for index, ticket in enumerate(self._historical_tickets)
        }
        self._tickets_by_id = {ticket["ticket_id"]: ticket for ticket in self._historical_tickets}

    @property
    def lexical_retriever(self) -> TicketRetriever:
        """Return the lexical retriever used by the hybrid index."""
        return self._lexical_retriever

    @property
    def semantic_retriever(self) -> SemanticTicketRetriever:
        """Return the semantic retriever used by the hybrid index."""
        return self._semantic_retriever

    @property
    def rrf_k(self) -> int:
        """Return the Reciprocal Rank Fusion constant."""
        return self._rrf_k

    def search(self, query: str, top_k: int = 3) -> list[RetrievedTicket]:
        """Return tickets ranked by fused lexical and semantic rankings."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0:
            return []

        lexical_results = self._lexical_retriever.search(query, top_k=top_k)
        semantic_results = self._semantic_retriever.search(query, top_k=top_k)
        scores = self._fuse_rankings(lexical_results, semantic_results)
        ranked_ticket_ids = sorted(
            scores,
            key=lambda ticket_id: (
                -scores[ticket_id],
                self._ticket_order[ticket_id],
            ),
        )[:top_k]

        return [
            RetrievedTicket(
                ticket_id=ticket_id,
                text=self._tickets_by_id[ticket_id]["text"],
                score=scores[ticket_id],
            )
            for ticket_id in ranked_ticket_ids
        ]

    def _fuse_rankings(
        self,
        lexical_results: list[RetrievedTicket],
        semantic_results: list[RetrievedTicket],
    ) -> dict[str, float]:
        """Calculate one RRF score for every candidate ticket ID."""
        scores: dict[str, float] = {}
        for rank, result in enumerate(lexical_results, start=1):
            scores[result.ticket_id] = scores.get(result.ticket_id, 0.0) + (
                1 / (self._rrf_k + rank)
            )
        for rank, result in enumerate(semantic_results, start=1):
            scores[result.ticket_id] = scores.get(result.ticket_id, 0.0) + (
                1 / (self._rrf_k + rank)
            )
        return scores
