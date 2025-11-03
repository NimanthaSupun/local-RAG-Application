import os
import json
import uuid
import requests
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# -------------------------
# Config
# -------------------------
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
GEN_MODEL = os.getenv("GEN_MODEL", "llama3.2")

USE_ANTHROPIC = os.getenv("USE_ANTHROPIC", "false").lower() == "true"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# dimension for mxbai-embed-large (1024)
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Local RAG (Ollama + Qdrant)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set your frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Qdrant client & collection
# -------------------------
qdrant = QdrantClient(url=QDRANT_URL)

def ensure_collection():
    exists = False
    try:
        info = qdrant.get_collection(COLLECTION)
        exists = info is not None
    except Exception:
        exists = False

    if not exists:
        qdrant.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE)
        )

ensure_collection()

# -------------------------
# Schemas
# -------------------------
class IngestItem(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Optional[dict] = None

class IngestRequest(BaseModel):
    items: List[IngestItem]

class AskRequest(BaseModel):
    question: str
    top_k: int = 3

class AskResponse(BaseModel):
    answer: str
    sources: List[dict]

# -------------------------
# Ollama helpers
# -------------------------
def ollama_embed(text: str) -> List[float]:
    url = f"{OLLAMA_URL}/api/embeddings"
    r = requests.post(url, json={"model": EMBED_MODEL, "prompt": text})
    if r.status_code != 200:
        raise HTTPException(500, f"Embedding error: {r.text}")
    data = r.json()
    return data["embedding"]

def ollama_generate(prompt: str) -> str:
    url = f"{OLLAMA_URL}/api/generate"
    # stream=false simplifies handling
    r = requests.post(url, json={"model": GEN_MODEL, "prompt": prompt, "stream": False})
    if r.status_code != 200:
        raise HTTPException(500, f"Generate error: {r.text}")
    data = r.json()
    return data.get("response", "")

# Optional: Anthropic fallback
def anthropic_generate(prompt: str) -> str:
    if not (USE_ANTHROPIC and ANTHROPIC_API_KEY):
        return ""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        return ""

# -------------------------
# RAG steps
# -------------------------
def build_prompt(question: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(context_chunks)
    return (
        "You are a helpful assistant. Use the provided context to answer the question.\n"
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )

def search_context(query: str, top_k: int) -> List[qmodels.ScoredPoint]:
    qvec = ollama_embed(query)
    hits = qdrant.search(
        collection_name=COLLECTION,
        query_vector=qvec,
        limit=top_k,
        with_payload=True,
    )
    return hits

# -------------------------
# Endpoints
# -------------------------
@app.post("/ingest")
def ingest(req: IngestRequest):
    points = []
    for it in req.items:
        pid = it.id or str(uuid.uuid4())
        vec = ollama_embed(it.text)
        payload = {"text": it.text}
        if it.metadata:
            payload.update(it.metadata)
        points.append(qmodels.PointStruct(id=pid, vector=vec, payload=payload))

    qdrant.upsert(collection_name=COLLECTION, points=points)
    return {"ok": True, "count": len(points)}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    hits = search_context(req.question, req.top_k)
    chunks = [h.payload.get("text", "") for h in hits]
    sources = [
        {"id": str(h.id), "score": h.score, "metadata": {k: v for k, v in h.payload.items() if k != "text"}}
        for h in hits
    ]
    prompt = build_prompt(req.question, chunks)

    answer = ollama_generate(prompt).strip()

    # simple fallback to Anthropic if answer is too weak
    if USE_ANTHROPIC and len(answer) < 20:
        alt = anthropic_generate(prompt)
        if alt:
            answer = alt

    return AskResponse(answer=answer, sources=sources)
