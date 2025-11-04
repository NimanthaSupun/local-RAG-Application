"""
Qdrant vector store operations.
Manages the vector database for storing and retrieving document embeddings.
"""
import uuid
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from config import QDRANT_URL, COLLECTION_NAME, EMBED_DIM


class VectorStore:
    """Wrapper for Qdrant vector database operations."""
    
    def __init__(self):
        """Initialize Qdrant client and ensure collection exists."""
        self.client = QdrantClient(url=QDRANT_URL)
        self.collection_name = COLLECTION_NAME
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=EMBED_DIM,
                    distance=qmodels.Distance.COSINE
                ),
            )
    
    def upsert_vectors(self, vectors: List[List[float]], payloads: List[dict]) -> int:
        """
        Insert or update vectors in the collection.
        
        Args:
            vectors: List of embedding vectors
            payloads: List of metadata dictionaries (must include 'text' key)
            
        Returns:
            Number of vectors inserted
        """
        points = []
        for vector, payload in zip(vectors, payloads):
            point_id = str(uuid.uuid4())
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            )
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return len(points)
    
    def search(self, query_vector: List[float], top_k: int = 3) -> List[qmodels.ScoredPoint]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: The query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of scored points with payloads
        """
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return hits
    
    def clear_collection(self):
        """Delete and recreate the collection (removes all data)."""
        self.client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection()
    
    def get_collection_info(self) -> dict:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
            }
        except Exception as e:
            return {"error": str(e)}
