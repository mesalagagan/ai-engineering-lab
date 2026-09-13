"""Run a live OpenAI-backed RAG demo over the support-ticket corpus."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_engineering_lab.hybrid_retrieval import HybridTicketRetriever
from ai_engineering_lab.openai_answer_generator import OpenAIAnswerGenerator
from ai_engineering_lab.rag import RAGPipeline
from experiments.compare_retrieval import build_demo_tickets


def main() -> None:
    """Answer one realistic ticket question with the live OpenAI API."""
    query = "My credit card payment was declined at checkout. What do the tickets suggest?"
    retriever = HybridTicketRetriever(build_demo_tickets())

    print("LIVE OPENAI API DEMO: this request may incur API cost.")
    print("OPENAI_API_KEY is read from the environment and is never printed.")
    print(f"\nUser question: {query}")

    pipeline = RAGPipeline(retriever, OpenAIAnswerGenerator())

    retrieved_tickets = retriever.search(query, top_k=3)
    print("\nRetrieved tickets:")
    for ticket in retrieved_tickets:
        print(f"  {ticket.ticket_id} (score: {ticket.score:.4f})")

    answer = pipeline.answer(query, top_k=3)
    print("\nOpenAI-generated answer:")
    print(answer)


if __name__ == "__main__":
    main()
