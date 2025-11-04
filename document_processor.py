"""
Document processing utilities.
Handles file reading, text extraction, and intelligent chunking.
"""
import tempfile
from typing import List, Dict
from datetime import datetime
from PyPDF2 import PdfReader
from config import CHUNK_SIZE, CHUNK_OVERLAP


def read_uploaded_file(uploaded_file) -> str:
    """
    Extract text from uploaded PDF or TXT file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Extracted text content
    """
    if uploaded_file.type == "application/pdf":
        return _read_pdf(uploaded_file)
    else:
        return _read_text(uploaded_file)


def _read_pdf(uploaded_file) -> str:
    """Extract text from PDF file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp.flush()
        
        reader = PdfReader(tmp.name)
        text = "\n".join(
            page.extract_text() or "" 
            for page in reader.pages
        )
    return text.strip()


def _read_text(uploaded_file) -> str:
    """Extract text from TXT file."""
    return uploaded_file.read().decode("utf-8").strip()


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum characters per chunk (uses config default if None)
        overlap: Number of characters to overlap between chunks (uses config default if None)
        
    Returns:
        List of text chunks
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if overlap is None:
        overlap = CHUNK_OVERLAP
    
    # Ensure overlap is less than chunk size
    overlap = min(overlap, chunk_size - 1)
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
        
        # Prevent infinite loop on small texts
        if start >= len(text):
            break
    
    return chunks


def create_chunk_metadata(
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    source_file_name: str,
    file_type: str
) -> Dict:
    """
    Create metadata dictionary for a text chunk.
    
    Args:
        chunk_text: The text content of the chunk
        chunk_index: Index of this chunk (0-based)
        total_chunks: Total number of chunks from the document
        source_file_name: Original filename
        file_type: MIME type of the file
        
    Returns:
        Dictionary with chunk metadata
    """
    return {
        "text": chunk_text,
        "source_file": source_file_name,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "upload_timestamp": datetime.now().isoformat(),
        "file_type": file_type,
    }


def process_document(uploaded_file, embedding_function) -> tuple[List[List[float]], List[Dict], int]:
    """
    Process a document: read, chunk, and generate embeddings.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        embedding_function: Function that takes text and returns embedding vector
        
    Returns:
        Tuple of (vectors, metadata_list, chunk_count)
    """
    # Extract text
    text = read_uploaded_file(uploaded_file)
    
    if not text:
        return [], [], 0
    
    # Chunk the text
    chunks = chunk_text(text)
    
    # Generate embeddings and metadata
    vectors = []
    metadata_list = []
    
    for i, chunk in enumerate(chunks):
        # Generate embedding
        vector = embedding_function(chunk)
        vectors.append(vector)
        
        # Create metadata
        metadata = create_chunk_metadata(
            chunk_text=chunk,
            chunk_index=i,
            total_chunks=len(chunks),
            source_file_name=uploaded_file.name,
            file_type=uploaded_file.type
        )
        metadata_list.append(metadata)
    
    return vectors, metadata_list, len(chunks)
