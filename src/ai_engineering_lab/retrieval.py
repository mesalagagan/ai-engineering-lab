"""Reusable TF-IDF retrieval for historical support tickets."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ai_engineering_lab.tfidf import TfidfVectorizer, most_similar


@dataclass(frozen=True, slots=True)
class RetrievedTicket:
    """A historical ticket returned by a similarity search."""

    ticket_id: str
    text: str
    score: float


class TicketRetriever:
    """Index historical tickets and retrieve the most similar tickets."""

    def __init__(self, historical_tickets: Iterable[Mapping[str, str]]) -> None:
        """Fit TF-IDF once and store vectors for the supplied ticket records."""
        self._historical_tickets = [
            {"ticket_id": ticket["ticket_id"], "text": ticket["text"]}
            for ticket in historical_tickets
        ]
        self._vectorizer = TfidfVectorizer()
        ticket_texts = [ticket["text"] for ticket in self._historical_tickets]
        self._ticket_vectors = self._vectorizer.fit_transform(ticket_texts)

    @property
    def vectorizer(self) -> TfidfVectorizer:
        """Return the fitted vectorizer used for all ticket searches."""
        return self._vectorizer

    def search(self, query: str, top_k: int = 3) -> list[RetrievedTicket]:
        """Return up to ``top_k`` historical tickets ranked by cosine similarity."""
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0:
            return []

        query_vector = self._vectorizer.transform([query])[0]
        ranked_matches = most_similar(query_vector, self._ticket_vectors, top_k=top_k)
        return [
            RetrievedTicket(
                ticket_id=self._historical_tickets[index]["ticket_id"],
                text=self._historical_tickets[index]["text"],
                score=score,
            )
            for index, score in ranked_matches
        ]
