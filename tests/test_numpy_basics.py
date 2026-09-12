"""Tests for the educational NumPy vector helpers."""

import numpy as np
import pytest

from ai_engineering_lab.numpy_basics import (
    cosine_similarity_numpy,
    dot_product,
    vector_from_values,
    vector_norm,
    vectors_to_matrix,
)


def test_vector_from_values_converts_lists_to_float_arrays() -> None:
    vector = vector_from_values([1, 2.5, 3])

    assert isinstance(vector, np.ndarray)
    assert vector.ndim == 1
    assert vector.dtype == np.dtype(float)
    np.testing.assert_array_equal(vector, np.array([1.0, 2.5, 3.0]))


def test_dot_product() -> None:
    assert dot_product(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == pytest.approx(11.0)


def test_vector_norm() -> None:
    assert vector_norm(np.array([3.0, 4.0])) == pytest.approx(5.0)


def test_cosine_similarity_for_identical_vectors() -> None:
    vector = np.array([1.0, 2.0])

    assert cosine_similarity_numpy(vector, vector) == pytest.approx(1.0)


def test_cosine_similarity_for_orthogonal_vectors() -> None:
    assert cosine_similarity_numpy(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_similarity_returns_zero_for_zero_vectors() -> None:
    assert cosine_similarity_numpy(np.array([0.0, 0.0]), np.array([1.0, 2.0])) == pytest.approx(0.0)


def test_vector_operations_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        dot_product(np.array([1.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="equal lengths"):
        cosine_similarity_numpy(np.array([1.0]), np.array([1.0, 2.0]))


def test_vector_operations_reject_non_one_dimensional_input() -> None:
    matrix = np.array([[1.0, 2.0]])

    with pytest.raises(ValueError, match="one-dimensional"):
        dot_product(matrix, np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        vector_norm(matrix)
    with pytest.raises(ValueError, match="one-dimensional"):
        cosine_similarity_numpy(matrix, np.array([1.0, 2.0]))


def test_vectors_to_matrix_converts_vectors() -> None:
    matrix = vectors_to_matrix([[1, 2], [3, 4]])

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2, 2)
    assert matrix.dtype == np.dtype(float)
    np.testing.assert_array_equal(matrix, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_vectors_to_matrix_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="rectangular|two-dimensional"):
        vectors_to_matrix([[1, 2], [3]])

    with pytest.raises(ValueError, match="two-dimensional"):
        vectors_to_matrix([])
