"""
Configuration management for Local RAG application.
Loads settings from environment variables with sensible defaults.
"""
import os
from dotenv import load_dotenv
import requests
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# -------------------------
# Ollama Configuration
# -------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
GEN_MODEL = os.getenv("GEN_MODEL", "llama3.2")

# -------------------------
# Qdrant Configuration
# -------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "docs")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))

# -------------------------
# Document Processing
# -------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# -------------------------
# Retrieval Configuration
# -------------------------
TOP_K = int(os.getenv("TOP_K", "3"))


def check_ollama_connection() -> bool:
    """Check if Ollama service is running and accessible."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def check_qdrant_connection() -> bool:
    """Check if Qdrant service is running and accessible."""
    try:
        response = requests.get(f"{QDRANT_URL}/collections", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def check_services():
    """
    Verify that all required services are running.
    Displays error messages in Streamlit UI if services are down.
    """
    ollama_ok = check_ollama_connection()
    qdrant_ok = check_qdrant_connection()
    
    if not ollama_ok:
        st.error(
            f"❌ **Ollama is not running or not accessible at `{OLLAMA_URL}`**\n\n"
            "Please start Ollama:\n"
            "```bash\n"
            "ollama serve\n"
            "```\n"
            f"Then ensure models are available:\n"
            "```bash\n"
            f"ollama pull {EMBED_MODEL}\n"
            f"ollama pull {GEN_MODEL}\n"
            "```"
        )
        st.stop()
    
    if not qdrant_ok:
        st.error(
            f"❌ **Qdrant is not running or not accessible at `{QDRANT_URL}`**\n\n"
            "Please start Qdrant:\n"
            "```bash\n"
            "docker run -p 6333:6333 -p 6334:6334 \\\n"
            "  -v $(pwd)/qdrant_storage:/qdrant/storage:z \\\n"
            "  qdrant/qdrant\n"
            "```"
        )
        st.stop()


def get_config_summary() -> dict:
    """Return a dictionary of current configuration for display purposes."""
    return {
        "Ollama URL": OLLAMA_URL,
        "Embedding Model": EMBED_MODEL,
        "Generation Model": GEN_MODEL,
        "Qdrant URL": QDRANT_URL,
        "Collection Name": COLLECTION_NAME,
        "Embedding Dimension": EMBED_DIM,
        "Chunk Size": CHUNK_SIZE,
        "Chunk Overlap": CHUNK_OVERLAP,
        "Top K Results": TOP_K,
    }
