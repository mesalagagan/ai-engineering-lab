"""Small educational retrieval evaluation metrics."""


def precision_at_k(
    retrieved_ticket_ids: list[str],
    relevant_ticket_ids: set[str],
    k: int,
) -> float:
    """Return the fraction of the first ``k`` results that are relevant.

    The denominator is always ``k``, including when fewer than ``k`` results
    were retrieved. This follows the standard precision@k definition.
    """
    _validate_k(k)
    retrieved_at_k = retrieved_ticket_ids[:k]
    relevant_count = sum(ticket_id in relevant_ticket_ids for ticket_id in retrieved_at_k)
    return relevant_count / k


def recall_at_k(
    retrieved_ticket_ids: list[str],
    relevant_ticket_ids: set[str],
    k: int,
) -> float:
    """Return the fraction of relevant tickets found in the first ``k`` results."""
    _validate_k(k)
    if not relevant_ticket_ids:
        raise ValueError("relevant_ticket_ids must not be empty")

    retrieved_at_k = set(retrieved_ticket_ids[:k])
    relevant_count = len(retrieved_at_k & relevant_ticket_ids)
    return relevant_count / len(relevant_ticket_ids)


def _validate_k(k: int) -> None:
    """Reject invalid cutoffs shared by the retrieval metrics."""
    if k <= 0:
        raise ValueError("k must be greater than zero")
