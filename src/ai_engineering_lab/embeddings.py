"""Application-level text embeddings using sentence-transformers."""

from collections.abc import Iterable

from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """Load one sentence-transformers model and generate text embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Load the named embedding model once for this embedder instance."""
        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        embedding_dimension = self._model.get_embedding_dimension()
        if embedding_dimension is None:
            raise ValueError("loaded model did not provide an embedding dimension")
        self._embedding_dimension = embedding_dimension

    @property
    def model_name(self) -> str:
        """Return the name used to load the embedding model."""
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        """Return the dimension of vectors produced by the loaded model."""
        return self._embedding_dimension

    def embed_text(self, text: str) -> list[float]:
        """Convert one string into a plain Python list of embedding floats."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self._encode([text])[0]

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        """Convert strings into embeddings in one batch, preserving input order."""
        text_list = list(texts)
        if any(not isinstance(text, str) for text in text_list):
            raise TypeError("every item in texts must be a string")
        if not text_list:
            return []
        return self._encode(text_list)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """Encode text and normalize the library result into Python lists."""
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [[float(value) for value in embedding] for embedding in embeddings]
