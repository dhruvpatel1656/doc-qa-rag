"""Text embedding backends.

* ``SentenceTransformerEmbedder`` - real semantic embeddings (default).
* ``HashingEmbedder`` - tiny, dependency-free bag-of-words embedder used for
  fast offline tests and CI (no model download needed).
"""
import hashlib
import os
import re
from typing import List, Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, texts: List[str]) -> np.ndarray:
        ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")


class HashingEmbedder:
    """Deterministic bag-of-words embedder based on the hashing trick."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.md5(token.encode()).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                out[row, index] += sign
        return _normalize(out)


class SentenceTransformerEmbedder:
    """Semantic embeddings from a Hugging Face sentence-transformers model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _normalize(vectors)


def get_embedder() -> Embedder:
    """Pick an embedder using the EMBEDDER environment variable."""
    backend = os.getenv("EMBEDDER", "sentence-transformers").lower()
    if backend == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder()
