"""FastAPI application exposing the document Q&A service."""
import io
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from pypdf import PdfReader

from .embeddings import get_embedder
from .llm import get_llm
from .rag import RagService
from .vectorstore import FaissStore

DATA_DIR = os.getenv("DATA_DIR", "data")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(4, ge=1, le=10)


def extract_text(filename: str, raw: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith((".txt", ".md")):
        return raw.decode("utf-8", errors="ignore")
    raise HTTPException(status_code=415, detail="Only .pdf, .txt and .md files are supported.")


def create_app(service: RagService = None) -> FastAPI:
    app = FastAPI(title="Document Q&A Assistant (RAG)", version="1.0.0")

    if service is None:
        embedder = get_embedder()
        store = FaissStore.load(DATA_DIR, embedder.dim)
        service = RagService(embedder, store, get_llm(), persist_dir=DATA_DIR)
    app.state.service = service

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks_indexed": len(app.state.service.store)}

    @app.post("/documents")
    async def upload_document(file: UploadFile = File(...)):
        text = extract_text(file.filename, await file.read())
        count = app.state.service.ingest_text(file.filename, text)
        if count == 0:
            raise HTTPException(status_code=422, detail="No readable text found in the file.")
        return {"filename": file.filename, "chunks_indexed": count}

    @app.post("/ask")
    def ask(request: AskRequest):
        if len(app.state.service.store) == 0:
            raise HTTPException(status_code=409, detail="Upload a document before asking questions.")
        return app.state.service.ask(request.question, request.top_k)

    return app
