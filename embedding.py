"""
Ollama embedding and generation functions.
Handles communication with local Ollama server for embeddings and text generation.
"""
import requests
import json
from typing import List, Iterator
from config import OLLAMA_URL, EMBED_MODEL, GEN_MODEL


def get_embedding(text: str) -> List[float]:
    """
    Generate an embedding vector from Ollama.
    
    Args:
        text: The text to embed
        
    Returns:
        List of floats representing the embedding vector
        
    Raises:
        requests.HTTPError: If the API request fails
    """
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text}
    )
    response.raise_for_status()
    data = response.json()
    return data["embedding"]


def generate_answer(prompt: str) -> str:
    """
    Generate answer using Ollama LLM (non-streaming).
    
    Args:
        prompt: The prompt to send to the LLM
        
    Returns:
        Generated text response
        
    Raises:
        requests.HTTPError: If the API request fails
    """
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": GEN_MODEL, "prompt": prompt, "stream": False}
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def generate_answer_streaming(prompt: str) -> Iterator[str]:
    """
    Generate answer using Ollama LLM with streaming response.
    
    Args:
        prompt: The prompt to send to the LLM
        
    Yields:
        Chunks of generated text as they arrive
        
    Raises:
        requests.HTTPError: If the API request fails
    """
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": GEN_MODEL, "prompt": prompt, "stream": True},
        stream=True
    )
    response.raise_for_status()
    
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line)
                chunk = data.get("response", "")
                if chunk:
                    yield chunk
                    
                # Check if generation is done
                if data.get("done", False):
                    break
            except json.JSONDecodeError:
                # Skip malformed JSON lines
                continue


def build_rag_prompt(question: str, context_chunks: List[str]) -> str:
    """
    Build a RAG prompt with context and question.
    
    Args:
        question: The user's question
        context_chunks: List of relevant text chunks to use as context
        
    Returns:
        Formatted prompt string
    """
    context = "\n\n".join(context_chunks)
    return (
        "You are a helpful assistant. Use the provided context to answer the question.\n"
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
