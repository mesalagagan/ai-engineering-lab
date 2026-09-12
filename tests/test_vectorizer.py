"""Tests for the Lesson 2 bag-of-words vectorizer."""

import pytest

from ai_engineering_lab.vectorizer import BagOfWordsVectorizer


def test_builds_case_normalized_vocabulary_and_count_vectors() -> None:
    vectorizer = BagOfWordsVectorizer()

    vectors = vectorizer.fit_transform(["Red apple", "blue apple red", "GREEN pear"])

    assert vectorizer.vocabulary == {
        "red": 0,
        "apple": 1,
        "blue": 2,
        "green": 3,
        "pear": 4,
    }
    assert vectors == [
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1],
    ]


def test_uses_whitespace_tokens_and_ignores_unknown_tokens() -> None:
    vectorizer = BagOfWordsVectorizer().fit(["Hello   WORLD"])

    assert vectorizer.transform(["world hello unknown", "  "]) == [
        [1, 1],
        [0, 0],
    ]


def test_transform_requires_a_fitted_vocabulary() -> None:
    with pytest.raises(ValueError, match="fit must be called"):
        BagOfWordsVectorizer().transform(["hello"])
