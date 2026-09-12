"""Educational NumPy helpers for basic vector operations."""

from collections.abc import Iterable

import numpy as np


def vector_from_values(values: Iterable[float]) -> np.ndarray:
    """Convert numeric values into a one-dimensional float array."""
    vector = np.asarray(list(values), dtype=float)
    if vector.ndim != 1:
        raise ValueError("values must describe a one-dimensional vector")
    return vector


def dot_product(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Return the dot product of two one-dimensional, equally sized vectors."""
    _validate_vector_pair(vector_a, vector_b)
    return float(np.dot(vector_a, vector_b))


def vector_norm(vector: np.ndarray) -> float:
    """Return the Euclidean (L2) norm of a one-dimensional vector."""
    _validate_vector(vector)
    return float(np.linalg.norm(vector))


def cosine_similarity_numpy(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    """Return cosine similarity, using zero for either zero-norm vector."""
    _validate_vector_pair(vector_a, vector_b)
    norm_a = vector_norm(vector_a)
    norm_b = vector_norm(vector_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product(vector_a, vector_b) / (norm_a * norm_b)


def vectors_to_matrix(vectors: Iterable[Iterable[float]]) -> np.ndarray:
    """Convert equally sized vectors into a two-dimensional float matrix."""
    try:
        matrix = np.asarray(list(vectors), dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("vectors must form a rectangular two-dimensional matrix") from error

    if matrix.ndim != 2:
        raise ValueError("vectors must form a two-dimensional matrix")
    return matrix


def _validate_vector(vector: np.ndarray) -> None:
    """Ensure an input array is one-dimensional."""
    if vector.ndim != 1:
        raise ValueError("vectors must be one-dimensional")


def _validate_vector_pair(vector_a: np.ndarray, vector_b: np.ndarray) -> None:
    """Ensure two arrays are one-dimensional and have equal lengths."""
    _validate_vector(vector_a)
    _validate_vector(vector_b)
    if vector_a.shape[0] != vector_b.shape[0]:
        raise ValueError("vectors must have equal lengths")
