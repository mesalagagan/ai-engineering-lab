"""A small educational TF-IDF vectorizer using only the standard library."""

import re
from collections.abc import Iterable, Mapping
from math import log, sqrt

_TOKEN_PATTERN = re.compile(r"\b\w+\b")


class TfidfVectorizer:
    """Learn a vocabulary and convert documents into TF-IDF vectors."""

    def __init__(self) -> None:
        """Start with no fitted vocabulary or document statistics."""
        self._vocabulary: dict[str, int] = {}
        self._document_frequency: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._number_of_documents = 0
        self._is_fitted = False

    @property
    def vocabulary(self) -> Mapping[str, int]:
        """Return a copy of the word-to-vector-position mapping."""
        return self._vocabulary.copy()

    @property
    def document_frequency(self) -> Mapping[str, int]:
        """Return how many fitted documents contain each vocabulary word."""
        return self._document_frequency.copy()

    @property
    def idf(self) -> Mapping[str, float]:
        """Return the smoothed inverse document frequency for each word."""
        return self._idf.copy()

    def fit(self, documents: Iterable[str]) -> "TfidfVectorizer":
        """Learn vocabulary, document frequency, and smoothed IDF values."""
        documents = list(documents)
        self._vocabulary.clear()
        self._document_frequency.clear()
        self._idf.clear()
        self._number_of_documents = len(documents)

        for document in documents:
            tokens = _tokenize(document)
            for token in tokens:
                if token not in self._vocabulary:
                    self._vocabulary[token] = len(self._vocabulary)
            for token in set(tokens):
                self._document_frequency[token] = self._document_frequency.get(token, 0) + 1

        for token, frequency in self._document_frequency.items():
            self._idf[token] = log((1 + self._number_of_documents) / (1 + frequency)) + 1

        self._is_fitted = True
        return self

    def transform(self, documents: Iterable[str]) -> list[list[float]]:
        """Convert documents into normalized TF-IDF vectors.

        Term frequency is the token count divided by the total number of
        tokens in the document. Unknown tokens are excluded from the output
        vector, but still count toward the document's total token count.
        """
        if not self._is_fitted:
            raise ValueError("fit must be called before transform")

        vectors: list[list[float]] = []
        for document in documents:
            tokens = _tokenize(document)
            vector = [0.0] * len(self._vocabulary)
            if not tokens:
                vectors.append(vector)
                continue

            token_counts: dict[str, int] = {}
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1

            total_tokens = len(tokens)
            for token, count in token_counts.items():
                token_index = self._vocabulary.get(token)
                if token_index is not None:
                    term_frequency = count / total_tokens
                    vector[token_index] = term_frequency * self._idf[token]
            vectors.append(vector)

        return vectors

    def fit_transform(self, documents: Iterable[str]) -> list[list[float]]:
        """Learn document statistics and transform the same documents."""
        documents = list(documents)
        self.fit(documents)
        return self.transform(documents)


def cosine_similarity(vector_a: Iterable[float], vector_b: Iterable[float]) -> float:
    """Return the cosine similarity between two vectors."""
    vector_a = list(vector_a)
    vector_b = list(vector_b)
    if len(vector_a) != len(vector_b):
        raise ValueError("vectors must have the same length")

    dot_product = sum(value_a * value_b for value_a, value_b in zip(vector_a, vector_b))
    magnitude_a = sqrt(sum(value * value for value in vector_a))
    magnitude_b = sqrt(sum(value * value for value in vector_b))
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


def most_similar(
    query_vector: Iterable[float],
    document_vectors: Iterable[Iterable[float]],
    top_k: int = 3,
) -> list[tuple[int, float]]:
    """Return document indexes and scores, ordered from most to least similar."""
    if top_k < 0:
        raise ValueError("top_k must be non-negative")

    scores = [
        (index, cosine_similarity(query_vector, document_vector))
        for index, document_vector in enumerate(document_vectors)
    ]
    scores.sort(key=lambda result: result[1], reverse=True)
    return scores[:top_k]


def _tokenize(document: str) -> list[str]:
    """Lowercase text and extract word-like tokens."""
    return _TOKEN_PATTERN.findall(document.casefold())
