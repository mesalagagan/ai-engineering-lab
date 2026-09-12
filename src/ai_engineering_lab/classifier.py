"""Deterministic support ticket classification."""

from dataclasses import dataclass
from enum import StrEnum


class Category(StrEnum):
    """Supported ticket categories."""

    ACCOUNT = "account"
    AUTHENTICATION = "authentication"
    BILLING = "billing"
    GENERAL = "general"
    TECHNICAL = "technical"


class Priority(StrEnum):
    """Supported ticket priorities."""

    HIGH = "high"
    LOW = "low"
    MEDIUM = "medium"


class Sentiment(StrEnum):
    """Supported ticket sentiment values."""

    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """The business-level result of classifying one support ticket."""

    category: Category
    priority: Priority
    sentiment: Sentiment


def classify_ticket(ticket_text: str) -> ClassificationResult:
    """Classify a ticket using small, deterministic keyword rules."""
    normalized_text = ticket_text.casefold()

    return ClassificationResult(
        category=_classify_category(normalized_text),
        priority=_classify_priority(normalized_text),
        sentiment=_classify_sentiment(normalized_text),
    )


def _classify_category(ticket_text: str) -> Category:
    if _contains_any(ticket_text, "password", "login", "log in", "sign in", "two-factor"):
        return Category.AUTHENTICATION
    if _contains_any(ticket_text, "invoice", "billing", "charged", "refund", "payment"):
        return Category.BILLING
    if _contains_any(ticket_text, "crash", "bug", "error", "not working", "broken"):
        return Category.TECHNICAL
    if _contains_any(ticket_text, "profile", "account", "delete my data"):
        return Category.ACCOUNT
    return Category.GENERAL


def _classify_priority(ticket_text: str) -> Priority:
    if _contains_any(ticket_text, "blocked", "urgent", "outage", "cannot access"):
        return Priority.HIGH
    if _contains_any(ticket_text, "minor", "question", "when you can"):
        return Priority.LOW
    return Priority.MEDIUM


def _classify_sentiment(ticket_text: str) -> Sentiment:
    if _contains_any(ticket_text, "angry", "frustrated", "terrible", "blocked", "cannot"):
        return Sentiment.NEGATIVE
    if _contains_any(ticket_text, "thank", "great", "helpful", "excellent"):
        return Sentiment.POSITIVE
    return Sentiment.NEUTRAL


def _contains_any(ticket_text: str, *phrases: str) -> bool:
    return any(phrase in ticket_text for phrase in phrases)
