"""NumPy-backed retrieval for historical support tickets."""

from collections.abc import Iterable, Mapping

import numpy as np

from ai_engineering_lab.numpy_basics import vectors_to_matrix
from ai_engineering_lab.retrieval import RetrievedTicket
from ai_engineering_lab.tfidf import TfidfVectorizer


class NumPyTicketRetriever:
    """Index historical tickets and search them with vectorized NumPy operations."""

    def __init__(self, historical_tickets: Iterable[Mapping[str, str]]) -> None:
        """Fit TF-IDF once and store historical vectors as a NumPy matrix."""
        self._historical_tickets = [
            {"ticket_id": ticket["ticket_id"], "text": ticket["text"]}
            for ticket in historical_tickets
        ]
        if not self._historical_tickets:
            raise ValueError("historical_tickets must not be empty")

        self._vectorizer = TfidfVectorizer()
        ticket_texts = [ticket["text"] for ticket in self._historical_tickets]
        ticket_vectors = self._vectorizer.fit_transform(ticket_texts)
        # Rows are tickets and columns are shared vocabulary features.
        self._ticket_matrix = vectors_to_matrix(ticket_vectors)

    @property
    def vectorizer(self) -> TfidfVectorizer:
        """Return the fitted vectorizer used for every query."""
        return self._vectorizer

    @property
    def ticket_matrix(self) -> np.ndarray:
        """Return the stored matrix with one row per historical ticket."""
        return self._ticket_matrix

    def search(self, query: str, top_k: int = 3) -> list[RetrievedTicket]:
        """Return up to ``top_k`` tickets ranked by vectorized cosine similarity."""
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0:
            return []

        query_vector = np.asarray(self._vectorizer.transform([query])[0], dtype=float)
        if query_vector.ndim != 1 or query_vector.shape[0] != self._ticket_matrix.shape[1]:
            raise ValueError("query and stored vectors must have compatible dimensions")

        # Matrix multiplication computes every ticket/query dot product at once.
        dot_products = self._ticket_matrix @ query_vector
        # Norms measure vector lengths; broadcasting combines each row norm with query norm.
        ticket_norms = np.linalg.norm(self._ticket_matrix, axis=1)
        query_norm = np.linalg.norm(query_vector)
        denominator = ticket_norms * query_norm
        similarities = np.zeros(self._ticket_matrix.shape[0], dtype=float)
        np.divide(dot_products, denominator, out=similarities, where=denominator != 0.0)
        # Vectorized array operations avoid the overhead of looping over tickets in Python.
        ranked_indexes = np.argsort(-similarities, kind="stable")[:top_k]

        return [
            RetrievedTicket(
                ticket_id=self._historical_tickets[index]["ticket_id"],
                text=self._historical_tickets[index]["text"],
                score=float(similarities[index]),
            )
            for index in ranked_indexes
        ]
