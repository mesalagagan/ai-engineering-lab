"""Tests for the Lesson 3 TF-IDF vectorizer."""

import pytest

from ai_engineering_lab.tfidf import (
    TfidfVectorizer,
    cosine_similarity,
    most_similar,
)


def test_builds_vocabulary_and_document_frequency() -> None:
    vectorizer = TfidfVectorizer().fit(["Red apple", "blue apple red", "green pear"])

    assert vectorizer.vocabulary == {
        "red": 0,
        "apple": 1,
        "blue": 2,
        "green": 3,
        "pear": 4,
    }
    assert vectorizer.document_frequency == {
        "red": 2,
        "apple": 2,
        "blue": 1,
        "green": 1,
        "pear": 1,
    }


def test_fit_transform_returns_tfidf_vectors() -> None:
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(["cat cat dog", "dog fish"])

    assert vectors[0][vectorizer.vocabulary["cat"]] == pytest.approx(
        (2 / 3) * vectorizer.idf["cat"]
    )
    assert vectors[0][vectorizer.vocabulary["dog"]] == pytest.approx(
        (1 / 3) * vectorizer.idf["dog"]
    )
    assert vectors[1][vectorizer.vocabulary["fish"]] == pytest.approx(
        (1 / 2) * vectorizer.idf["fish"]
    )


def test_repeated_terms_receive_proportionally_larger_tf_values() -> None:
    vectorizer = TfidfVectorizer().fit(["red blue"])

    vector = vectorizer.transform(["red red blue"])[0]

    assert vector[vectorizer.vocabulary["red"]] == pytest.approx(
        2 * vector[vectorizer.vocabulary["blue"]]
    )


def test_empty_documents_produce_zero_vectors() -> None:
    vectorizer = TfidfVectorizer().fit(["red blue"])

    assert vectorizer.transform(["", "   "]) == [[0.0, 0.0], [0.0, 0.0]]


def test_unknown_words_are_ignored() -> None:
    vectorizer = TfidfVectorizer().fit(["Hello world"])

    assert vectorizer.transform(["world unknown", "unknown"])[0] == [
        0.0,
        vectorizer.idf["world"] / 2,
    ]
    assert vectorizer.transform(["unknown"])[0] == [0.0, 0.0]


def test_support_ticket_query_can_rank_historical_tickets() -> None:
    tickets = [
        "password reset failed",
        "cannot reset account password",
        "invoice payment failed",
        "feature request for dashboard",
    ]
    vectorizer = TfidfVectorizer().fit(tickets)
    query_vector = vectorizer.transform(["password reset problem"])[0]

    results = most_similar(query_vector, vectorizer.transform(tickets))

    assert results[0][0] == 0
    assert results[0][1] > results[1][1]
    assert [index for index, _ in results] == [0, 1, 2]


def test_common_words_have_lower_idf_than_rare_words() -> None:
    vectorizer = TfidfVectorizer().fit(["common rare", "common", "common"])

    assert vectorizer.idf["common"] < vectorizer.idf["rare"]


def test_cosine_similarity_of_identical_vectors_is_one() -> None:
    assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_zero_vectors_do_not_cause_division_by_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
    assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0


def test_most_similar_returns_descending_similarity_order() -> None:
    results = most_similar([1.0, 0.0], [[0.5, 0.0], [0.0, 1.0], [1.0, 0.0]])

    assert [index for index, _ in results] == [0, 2, 1]
    assert [score for _, score in results] == pytest.approx([1.0, 1.0, 0.0])
