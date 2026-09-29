"""Split raw text into overlapping word-based chunks."""
from typing import List


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> List[str]:
    """Split ``text`` into chunks of about ``chunk_size`` words.

    Consecutive chunks share ``overlap`` words so that answers that sit on a
    chunk boundary are still retrievable.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_size]
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks
