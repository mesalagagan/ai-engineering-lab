"""Tests for the sentence-transformers embedding wrapper."""

import numpy as np
import pytest

from ai_engineering_lab.embeddings import TextEmbedder


@pytest.fixture(scope="module")
def embedder() -> TextEmbedder:
    """Load the default model once for this test module."""
    return TextEmbedder()


def test_default_model_loads_and_exposes_metadata(embedder: TextEmbedder) -> None:
    assert embedder.model_name == "all-MiniLM-L6-v2"
    assert isinstance(embedder.embedding_dimension, int)
    assert embedder.embedding_dimension > 0


def test_one_text_produces_float_vector_of_expected_dimension(embedder: TextEmbedder) -> None:
    vector = embedder.embed_text("password reset failed")

    assert isinstance(vector, list)
    assert len(vector) == embedder.embedding_dimension
    assert all(isinstance(value, float) for value in vector)


def test_multiple_texts_preserve_input_order(embedder: TextEmbedder) -> None:
    texts = ["password reset failed", "invoice payment failed"]

    vectors = embedder.embed_texts(texts)
    individual_vectors = [embedder.embed_text(text) for text in texts]

    assert len(vectors) == len(texts)
    for batch_vector, individual_vector in zip(vectors, individual_vectors, strict=True):
        np.testing.assert_allclose(batch_vector, individual_vector, rtol=1e-5, atol=1e-6)


def test_empty_input_returns_empty_list(embedder: TextEmbedder) -> None:
    assert embedder.embed_texts([]) == []


def test_non_string_input_raises_type_error(embedder: TextEmbedder) -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        embedder.embed_text(123)  # type: ignore[arg-type]


def test_non_string_items_raise_type_error(embedder: TextEmbedder) -> None:
    with pytest.raises(TypeError, match="every item in texts must be a string"):
        embedder.embed_texts(["valid text", 123])  # type: ignore[list-item]


def test_repeated_calls_use_the_same_loaded_model_instance(embedder: TextEmbedder) -> None:
    model = embedder._model

    embedder.embed_text("first text")
    embedder.embed_text("second text")

    assert embedder._model is model
