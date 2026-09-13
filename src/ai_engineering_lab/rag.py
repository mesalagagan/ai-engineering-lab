"""Testable foundations for retrieval-augmented generation (RAG)."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, cast

from ai_engineering_lab.retrieval import RetrievedTicket


@dataclass(frozen=True, slots=True)
class RAGContext:
    """Immutable query, retrieved evidence, and formatted prompt context."""

    query: str
    retrieved_tickets: tuple[RetrievedTicket, ...]
    formatted_context: str


class AnswerGenerator(Protocol):
    """Callable contract for an answer generator or an LLM adapter.

    A protocol keeps this pipeline independent from any particular provider:
    production code can inject an API-backed object, while tests can inject a
    deterministic local callable.
    """

    def __call__(self, query: str, context: str) -> str:
        """Generate an answer from the query and retrieved context."""
        ...


class TicketRetrieverLike(Protocol):
    """Minimal retriever interface required by ``RAGPipeline``."""

    def search(self, query: str, top_k: int = 3) -> list[RetrievedTicket]:
        """Return ranked tickets for a query."""
        ...


def build_context(
    query: str,
    retrieved_tickets: Iterable[RetrievedTicket],
) -> RAGContext:
    """Materialize tickets and format deterministic evidence without scores."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")

    tickets = tuple(retrieved_tickets)
    formatted_context = "\n\n".join(
        f"Ticket {ticket.ticket_id}:\n{ticket.text}" for ticket in tickets
    )
    return RAGContext(
        query=query,
        retrieved_tickets=tickets,
        formatted_context=formatted_context,
    )


class RAGPipeline:
    """Retrieve supporting tickets, format context, and generate an answer."""

    def __init__(self, retriever: object, answer_generator: AnswerGenerator) -> None:
        """Store injected retrieval and generation components."""
        self._retriever = cast(TicketRetrieverLike, retriever)
        self._answer_generator = answer_generator

    @property
    def retriever(self) -> object:
        """Return the injected retriever."""
        return self._retriever

    @property
    def answer_generator(self) -> AnswerGenerator:
        """Return the injected answer generator."""
        return self._answer_generator

    def answer(self, query: str, top_k: int = 3) -> str:
        """Retrieve context and pass it to the injected answer generator."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")

        retrieved_tickets = self._retriever.search(query, top_k=top_k)
        rag_context = build_context(query, retrieved_tickets)
        return self._answer_generator(query, rag_context.formatted_context)


class ExtractiveAnswerGenerator:
    """Deterministic demo generator that returns retrieved context verbatim."""

    def __call__(self, query: str, context: str) -> str:
        """Return a local extractive answer without calling an external API."""
        del query
        if not context:
            return "I could not find supporting ticket information."
        return f"Based on the retrieved tickets:\n{context}"
