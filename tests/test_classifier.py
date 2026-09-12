"""Tests for deterministic support ticket classification."""

from ai_engineering_lab.classifier import (
    Category,
    ClassificationResult,
    Priority,
    Sentiment,
    classify_ticket,
)


def test_classifies_blocked_password_reset() -> None:
    result = classify_ticket(
        "Customer cannot reset password and is completely blocked from logging in."
    )

    assert result == ClassificationResult(
        category=Category.AUTHENTICATION,
        priority=Priority.HIGH,
        sentiment=Sentiment.NEGATIVE,
    )


def test_classifies_billing_ticket() -> None:
    result = classify_ticket("Thank you for the helpful invoice correction.")

    assert result.category is Category.BILLING
    assert result.sentiment is Sentiment.POSITIVE
    assert result.priority is Priority.MEDIUM


def test_uses_neutral_defaults_for_unrecognized_text() -> None:
    result = classify_ticket("I have a question about the service.")

    assert result == ClassificationResult(
        category=Category.GENERAL,
        priority=Priority.LOW,
        sentiment=Sentiment.NEUTRAL,
    )


def test_classifies_technical_ticket() -> None:
    result = classify_ticket("The application crashes with an error.")

    assert result == ClassificationResult(
        category=Category.TECHNICAL,
        priority=Priority.MEDIUM,
        sentiment=Sentiment.NEUTRAL,
    )


def test_classifies_high_priority_billing_ticket() -> None:
    result = classify_ticket("I was charged twice and need an urgent refund.")

    assert result == ClassificationResult(
        category=Category.BILLING,
        priority=Priority.HIGH,
        sentiment=Sentiment.NEUTRAL,
    )


def test_classifies_positive_general_ticket() -> None:
    result = classify_ticket("The service is excellent, thank you.")

    assert result == ClassificationResult(
        category=Category.GENERAL,
        priority=Priority.MEDIUM,
        sentiment=Sentiment.POSITIVE,
    )


def test_uses_neutral_defaults_for_empty_ticket() -> None:
    result = classify_ticket("")

    assert result == ClassificationResult(
        category=Category.GENERAL,
        priority=Priority.MEDIUM,
        sentiment=Sentiment.NEUTRAL,
    )


def test_matches_uppercase_ticket_case_insensitively() -> None:
    result = classify_ticket("PASSWORD RESET IS NOT WORKING")

    assert result == ClassificationResult(
        category=Category.AUTHENTICATION,
        priority=Priority.MEDIUM,
        sentiment=Sentiment.NEUTRAL,
    )
