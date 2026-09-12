"""A small bag-of-words text vectorizer for Lesson 2.

The vectorizer first normalizes each sentence to lowercase and splits it on
whitespace to produce tokens. ``fit`` builds a vocabulary by assigning each
unique token a stable integer position. ``transform`` uses those positions to
count how often each vocabulary token appears in every sentence, producing a
fixed-length numerical vector. Tokens that were not seen during ``fit`` are
ignored. Because tokenization only splits on whitespace, punctuation remains
attached to a token, so ``"hello,"`` and ``"hello"`` are different tokens.
"""

from collections.abc import Iterable, Mapping


class BagOfWordsVectorizer:
    """Convert text into count vectors using a fitted vocabulary."""

    def __init__(self) -> None:
        """Start with an empty vocabulary that can be filled by ``fit``."""
        self._vocabulary: dict[str, int] = {}

    @property
    def vocabulary(self) -> Mapping[str, int]:
        """Return a copy of the word-to-vector-position mapping."""
        return self._vocabulary.copy()

    def fit(self, sentences: Iterable[str]) -> "BagOfWordsVectorizer":
        """Learn which words exist and assign each word a vector position.

        The sentences are read one at a time. Each new token is added to the
        vocabulary with the next available index. Fitting clears any previous
        vocabulary, so this method teaches the object a new set of words.
        """
        self._vocabulary.clear()

        for sentence in sentences:
            for token in _tokenize(sentence):
                if token not in self._vocabulary:
                    self._vocabulary[token] = len(self._vocabulary)

        return self

    def transform(self, sentences: Iterable[str]) -> list[list[int]]:
        """Convert sentences into vectors using the learned vocabulary.

        Each output vector has one position for every vocabulary token. The
        value at a position is the number of times that token appears in the
        sentence. Words that were not learned by ``fit`` are skipped.
        """
        if not self._vocabulary:
            raise ValueError("fit must be called before transform")

        vectors: list[list[int]] = []
        for sentence in sentences:
            vector = [0] * len(self._vocabulary)
            for token in _tokenize(sentence):
                token_index = self._vocabulary.get(token)
                if token_index is not None:
                    vector[token_index] += 1
            vectors.append(vector)

        return vectors

    def fit_transform(self, sentences: Iterable[str]) -> list[list[int]]:
        """Learn the vocabulary and convert the same sentences in one step.

        The input is saved as a list because it may be a one-use iterable,
        such as a generator. The list is needed once for ``fit`` and again for
        ``transform``.
        """
        sentences = list(sentences)
        self.fit(sentences)
        return self.transform(sentences)


def _tokenize(sentence: str) -> list[str]:
    """Prepare one sentence by lowercasing it and splitting on whitespace."""
    return sentence.casefold().split()
