import pytest
from fastapi.testclient import TestClient

from app.chunking import chunk_text
from app.embeddings import HashingEmbedder
from app.llm import ExtractiveLLM
from app.main import create_app
from app.rag import RagService
from app.vectorstore import FaissStore

DOC = (
    "Amazon fulfillment centers sort packages by destination. "
    "The mezzanine level carries conveyor belts that feed scanning stations. "
) * 5 + (
    "The Roman aqueducts carried fresh water into cities using gravity and arched stone bridges. "
    "Engineers measured the gentle slope of each channel very carefully. "
) * 5


def make_service(tmp_path=None):
    embedder = HashingEmbedder()
    store = FaissStore(embedder.dim)
    return RagService(embedder, store, ExtractiveLLM(), persist_dir=str(tmp_path) if tmp_path else None)


def test_chunking_overlap_and_coverage():
    text = " ".join(f"w{i}" for i in range(300))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) >= 3
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]
    assert "w299" in chunks[-1]


def test_chunking_validates_arguments():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=10, overlap=10)
    assert chunk_text("   ") == []


def test_embeddings_are_normalised_and_deterministic():
    emb = HashingEmbedder()
    a = emb.embed(["hello world"])
    b = emb.embed(["hello world"])
    assert (a == b).all()
    assert abs(float((a ** 2).sum()) - 1.0) < 1e-5


def test_retrieval_returns_the_relevant_topic():
    service = make_service()
    assert service.ingest_text("mixed.txt", DOC) > 0
    result = service.retrieve("How did Roman aqueducts move water?", k=2)
    assert "aqueducts" in result[0]["text"].lower()


def test_store_persists_to_disk(tmp_path):
    service = make_service(tmp_path)
    service.ingest_text("mixed.txt", DOC)
    reloaded = FaissStore.load(str(tmp_path), HashingEmbedder().dim)
    assert len(reloaded) == len(service.store)
    assert reloaded.metadata[0]["source"] == "mixed.txt"


def test_api_upload_and_ask():
    client = TestClient(create_app(make_service()))
    assert client.get("/health").json()["chunks_indexed"] == 0

    res = client.post("/documents", files={"file": ("notes.txt", DOC.encode(), "text/plain")})
    assert res.status_code == 200 and res.json()["chunks_indexed"] > 0

    res = client.post("/ask", json={"question": "What carried water into Roman cities?"})
    body = res.json()
    assert res.status_code == 200
    assert "notes.txt" in body["answer"]
    assert body["sources"] and "score" in body["sources"][0]


def test_api_rejects_bad_input():
    client = TestClient(create_app(make_service()))
    assert client.post("/ask", json={"question": "anything here"}).status_code == 409
    bad = client.post("/documents", files={"file": ("x.exe", b"nope", "application/octet-stream")})
    assert bad.status_code == 415
    empty = client.post("/documents", files={"file": ("empty.txt", b"   ", "text/plain")})
    assert empty.status_code == 422
