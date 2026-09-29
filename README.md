# Document Q&A Assistant (RAG)

Upload documents (PDF, TXT, Markdown) and ask questions about them. The app finds the most relevant passages with **text embeddings and FAISS vector search**, then uses an **LLM API (Claude)** to write an answer that cites its sources. Built with **Python and FastAPI**.

## How it works

```
Upload file -> extract text -> split into overlapping chunks -> embed chunks -> store in FAISS
Question -> embed question -> FAISS finds top-k chunks -> Claude answers using only those chunks (with [1], [2] citations)
```

| Module | Job |
| --- | --- |
| `app/chunking.py` | Overlapping word-based chunking |
| `app/embeddings.py` | Sentence-transformers embeddings (plus a tiny offline embedder for tests) |
| `app/vectorstore.py` | FAISS cosine-similarity index, saved to disk |
| `app/llm.py` | Claude answer generation, with an offline fallback that returns the best passage |
| `app/rag.py` | The ingest -> retrieve -> answer pipeline |
| `app/main.py` | FastAPI endpoints |

## Run it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # optional: add ANTHROPIC_API_KEY for generated answers
uvicorn app.server:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive API page. The first start downloads the embedding model (about 90 MB).

## API

* `POST /documents` - upload a `.pdf`, `.txt` or `.md` file
* `POST /ask` - `{"question": "...", "top_k": 4}` returns an answer plus the source passages with similarity scores
* `GET /health` - service status and number of indexed chunks

## Tests

```bash
EMBEDDER=hashing python -m pytest -q     # Windows PowerShell: $env:EMBEDDER="hashing"; python -m pytest -q
```

Tests cover chunking, embeddings, retrieval quality, disk persistence, and the API (upload, ask, and error cases).

## Ideas for next steps

* Re-ranking of retrieved passages
* Per-document deletion and filtering
* A small web front end
