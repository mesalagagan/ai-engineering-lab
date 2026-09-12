"""Semantic ticket retrieval using dense text embeddings."""

from collections.abc import Iterable, Mapping

import numpy as np

from ai_engineering_lab.embeddings import TextEmbedder
from ai_engineering_lab.numpy_basics import vectors_to_matrix
from ai_engineering_lab.retrieval import RetrievedTicket


class SemanticTicketRetriever:
    """Index tickets with dense embeddings and search them by cosine similarity."""

    def __init__(
        self,
        historical_tickets: Iterable[Mapping[str, str]],
        embedder: TextEmbedder | None = None,
    ) -> None:
        """Embed the historical corpus once and store it as a dense matrix."""
        self._historical_tickets = [
            {"ticket_id": ticket["ticket_id"], "text": ticket["text"]}
            for ticket in historical_tickets
        ]
        if not self._historical_tickets:
            raise ValueError("historical_tickets must not be empty")

        self._embedder = embedder if embedder is not None else TextEmbedder()
        ticket_texts = [ticket["text"] for ticket in self._historical_tickets]
        # A dense matrix keeps every embedding row aligned to one ticket.
        self._embedding_matrix = vectors_to_matrix(self._embedder.embed_texts(ticket_texts))
        if self._embedding_matrix.shape[1] != self._embedder.embedding_dimension:
            raise ValueError("historical embeddings have an incompatible dimension")

    @property
    def embedder(self) -> TextEmbedder:
        """Return the embedder used for the historical corpus and queries."""
        return self._embedder

    @property
    def embedding_matrix(self) -> np.ndarray:
        """Return the dense matrix with one embedding row per ticket."""
        return self._embedding_matrix

    @property
    def embedding_dimension(self) -> int:
        """Return the number of features in each dense embedding."""
        return self._embedder.embedding_dimension

    def search(self, query: str, top_k: int = 3) -> list[RetrievedTicket]:
        """Return up to ``top_k`` tickets ranked by semantic similarity."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0:
            return []

        query_vector = np.asarray(self._embedder.embed_text(query), dtype=float)
        if query_vector.ndim != 1 or query_vector.shape[0] != self.embedding_dimension:
            raise ValueError("query vector must match the embedding dimension")

        # Matrix multiplication calculates one dot product for every ticket row.
        dot_products = self._embedding_matrix @ query_vector
        # Norms measure length; broadcasting combines each row norm with the query norm.
        ticket_norms = np.linalg.norm(self._embedding_matrix, axis=1)
        query_norm = np.linalg.norm(query_vector)
        denominator = ticket_norms * query_norm
        similarities = np.zeros(self._embedding_matrix.shape[0], dtype=float)
        np.divide(dot_products, denominator, out=similarities, where=denominator != 0.0)
        # Vectorized similarity avoids a Python loop over the historical tickets.
        ranked_indexes = np.argsort(-similarities, kind="stable")[:top_k]

        return [
            RetrievedTicket(
                ticket_id=self._historical_tickets[index]["ticket_id"],
                text=self._historical_tickets[index]["text"],
                score=float(similarities[index]),
            )
            for index in ranked_indexes
        ]
