"""The retrieval-augmented generation pipeline: ingest -> retrieve -> answer."""
from typing import Dict, List

from .chunking import chunk_text
from .vectorstore import FaissStore


class RagService:
    def __init__(self, embedder, store: FaissStore, llm, persist_dir: str = None):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.persist_dir = persist_dir

    def ingest_text(self, source: str, text: str) -> int:
        chunks = chunk_text(text)
        if not chunks:
            return 0
        vectors = self.embedder.embed(chunks)
        metadata = [{"source": source, "chunk": i, "text": c} for i, c in enumerate(chunks)]
        self.store.add(vectors, metadata)
        if self.persist_dir:
            self.store.save(self.persist_dir)
        return len(chunks)

    def retrieve(self, question: str, k: int = 4) -> List[Dict]:
        query = self.embedder.embed([question])
        return [{**meta, "score": round(score, 4)} for score, meta in self.store.search(query, k)]

    def ask(self, question: str, k: int = 4) -> Dict:
        passages = self.retrieve(question, k)
        answer = self.llm.generate(question, passages)
        return {"answer": answer, "sources": passages}
