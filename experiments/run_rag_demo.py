"""Run a local RAG pipeline demo over the support-ticket corpus."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_engineering_lab.hybrid_retrieval import HybridTicketRetriever
from ai_engineering_lab.rag import ExtractiveAnswerGenerator, RAGPipeline, build_context
from experiments.compare_retrieval import build_demo_tickets


def main() -> None:
    """Retrieve ticket context and generate a deterministic demo answer."""
    query = "I want my money back"
    retriever = HybridTicketRetriever(build_demo_tickets())
    pipeline = RAGPipeline(retriever, ExtractiveAnswerGenerator())
    retrieved_tickets = retriever.search(query, top_k=3)

    print("Answer generator: ExtractiveAnswerGenerator")
    print("This is a deterministic demo substitute for an LLM; no external API is called.")
    print(f"\nQuery: {query}")
    print("Retrieved ticket IDs:")
    for ticket in retrieved_tickets:
        print(f"  {ticket.ticket_id}")

    context = build_context(query, retrieved_tickets)
    answer = pipeline.answer(query, top_k=3)
    print("\nFormatted context:")
    print(context.formatted_context)
    print("\nGenerated answer:")
    print(answer)


if __name__ == "__main__":
    main()
