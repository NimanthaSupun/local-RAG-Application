import os
import requests
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from PyPDF2 import PdfReader
import tempfile
import uuid

# -----------------------------
# CONFIG
# -----------------------------
OLLAMA_URL = "http://localhost:11434"
QDRANT_URL = "http://localhost:6333"
EMBED_MODEL = "mxbai-embed-large"
GEN_MODEL = "llama3.2"
COLLECTION_NAME = "docs"
EMBED_DIM = 1024  # embedding dimension for mxbai-embed-large

# -----------------------------
# INIT QDRANT (FIXED)
# -----------------------------
qdrant = QdrantClient(url=QDRANT_URL)

# Create collection only if it doesn't exist
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=EMBED_DIM,
            distance=qmodels.Distance.COSINE
        ),
    )

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def get_embedding(text: str):
    """Generate an embedding vector from Ollama"""
    res = requests.post(f"{OLLAMA_URL}/api/embeddings", json={
        "model": EMBED_MODEL,
        "prompt": text
    })
    res.raise_for_status()
    data = res.json()
    return data["embedding"]

def generate_answer(prompt: str):
    """Generate answer using Ollama LLM"""
    res = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": GEN_MODEL,
        "prompt": prompt,
        "stream": False
    })
    res.raise_for_status()
    data = res.json()
    return data.get("response", "").strip()

def search_similar(query: str, top_k=3):
    """Search Qdrant for similar chunks"""
    qvec = get_embedding(query)
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=qvec,
        limit=top_k,
        with_payload=True,
    )
    return hits

def ingest_text_to_qdrant(text: str, chunk_size=500):
    """Split long text into chunks and store in Qdrant"""
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    points = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        # ✅ Give each vector a unique string ID
        points.append(qmodels.PointStruct(id=str(uuid.uuid4()), vector=emb, payload={"text": chunk}))
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(chunks)

def read_file(uploaded_file):
    """Extract text from PDF or TXT file"""
    if uploaded_file.type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            reader = PdfReader(tmp.name)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = uploaded_file.read().decode("utf-8")
    return text.strip()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Local RAG with Ollama", layout="centered")
st.title("💬 Local RAG System (Ollama + Qdrant + Streamlit)")

# Sidebar for document upload
st.sidebar.header("📄 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# Ingest files
if uploaded_files:
    with st.spinner("Processing and ingesting documents..."):
        total_chunks = 0
        for file in uploaded_files:
            text = read_file(file)
            if not text:
                continue
            count = ingest_text_to_qdrant(text)
            total_chunks += count
        st.sidebar.success(f"✅ Ingested {total_chunks} chunks from {len(uploaded_files)} file(s).")

# Clear database button
if st.sidebar.button("🗑️ Clear Qdrant Data"):
    qdrant.delete_collection(collection_name=COLLECTION_NAME)
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE),
    )
    st.sidebar.warning("Qdrant collection cleared!")

# Query interface
st.markdown("---")
query = st.text_input("🔍 Ask a question about your uploaded documents:")

if st.button("Ask") and query:
    with st.spinner("Thinking..."):
        hits = search_similar(query, top_k=3)
        if not hits:
            st.warning("No relevant context found in Qdrant. Try uploading documents first.")
        else:
            context = "\n\n".join(h.payload["text"] for h in hits)
            prompt = (
                f"You are a helpful assistant. Use the context below to answer the question.\n"
                f"If the context does not contain the answer, say 'I don't know.'\n\n"
                f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            )
            answer = generate_answer(prompt)
            st.subheader("🧠 Answer:")
            st.write(answer)

            # Display sources
            st.markdown("### 📚 Context Chunks Used")
            for i, h in enumerate(hits, 1):
                st.write(f"**{i}.** (Score: {h.score:.3f}) {h.payload['text'][:250]}...")
