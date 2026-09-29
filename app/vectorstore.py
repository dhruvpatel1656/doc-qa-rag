"""A small FAISS-backed vector store with JSON metadata and disk persistence."""
import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np


class FaissStore:
    def __init__(self, dim: int):
        self.dim = dim
        # Inner product on L2-normalised vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict] = []

    def __len__(self) -> int:
        return self.index.ntotal

    def add(self, vectors: np.ndarray, metadata: List[Dict]) -> None:
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata must have the same length")
        if len(vectors) == 0:
            return
        self.index.add(np.ascontiguousarray(vectors, dtype="float32"))
        self.metadata.extend(metadata)

    def search(self, query: np.ndarray, k: int = 4) -> List[Tuple[float, Dict]]:
        if len(self) == 0:
            return []
        k = min(k, len(self))
        scores, ids = self.index.search(np.ascontiguousarray(query, dtype="float32"), k)
        return [
            (float(score), self.metadata[i])
            for score, i in zip(scores[0], ids[0])
            if i != -1
        ]

    def save(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        (path / "metadata.json").write_text(json.dumps(self.metadata))

    @classmethod
    def load(cls, directory: str, dim: int) -> "FaissStore":
        path = Path(directory)
        store = cls(dim)
        index_file, meta_file = path / "index.faiss", path / "metadata.json"
        if index_file.exists() and meta_file.exists():
            store.index = faiss.read_index(str(index_file))
            store.metadata = json.loads(meta_file.read_text())
        return store
