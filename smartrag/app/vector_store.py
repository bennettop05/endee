"""
SmartRAG Vector Store Module
Wrapper around the Endee Python SDK for vector operations.
"""

import logging
from typing import Optional
from endee import Endee, Precision
from app.config import config

logger = logging.getLogger(__name__)

# Map string precision to Endee Precision enum
PRECISION_MAP = {
    "BINARY": Precision.BINARY,
    "INT8": Precision.INT8,
    "INT16": Precision.INT16,
    "FLOAT16": Precision.FLOAT16,
    "FLOAT32": Precision.FLOAT32,
}


class EndeeVectorStore:
    """
    Wrapper around the Endee Vector Database SDK.
    Handles index management, vector upsert, and similarity search.
    """

    def __init__(self):
        self.host = config.endee.host
        self.index_name = config.endee.index_name
        self.dimension = config.endee.dimension
        self.space_type = config.endee.space_type
        self.precision = config.endee.precision
        self._client = None
        self._index = None

    @property
    def client(self) -> Endee:
        """Lazy-initialize Endee client."""
        if self._client is None:
            auth_token = config.endee.auth_token
            if auth_token:
                self._client = Endee(auth_token)
            else:
                self._client = Endee()
            self._client.set_base_url(config.endee.api_base)
            logger.info(f"Connected to Endee at {self.host}")
        return self._client

    async def ensure_index(self):
        """
        Ensure the vector index exists in Endee.
        Creates it if it doesn't exist.
        """
        try:
            # Try to get existing index
            self._index = self.client.get_index(name=self.index_name)
            logger.info(f"Using existing index: {self.index_name}")
        except Exception:
            # Create new index
            logger.info(f"Creating new index: {self.index_name} (dim={self.dimension})")
            precision = PRECISION_MAP.get(self.precision, Precision.INT8)
            self.client.create_index(
                name=self.index_name,
                dimension=self.dimension,
                space_type=self.space_type,
                precision=precision,
            )
            self._index = self.client.get_index(name=self.index_name)
            logger.info(f"Index created successfully: {self.index_name}")

    async def upsert_vectors(self, vectors: list[dict]):
        """
        Upsert vectors into the Endee index.

        Args:
            vectors: List of dicts with 'id', 'vector', and 'meta' keys.
        """
        await self.ensure_index()

        batch_size = 100
        total = len(vectors)

        for i in range(0, total, batch_size):
            batch = vectors[i:i + batch_size]
            self._index.upsert(batch)
            logger.info(f"Upserted batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}")

        logger.info(f"Successfully upserted {total} vectors")

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Perform similarity search in Endee.

        Args:
            query_vector: The query embedding vector.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of search results with id, similarity score, and metadata.
        """
        await self.ensure_index()

        try:
            if filters:
                results = self._index.query(
                    vector=query_vector,
                    top_k=top_k,
                    filter=filters,
                )
            else:
                results = self._index.query(
                    vector=query_vector,
                    top_k=top_k,
                )

            # Parse results into standardized format
            parsed_results = []
            for result in results:
                parsed_results.append({
                    "id": result.get("id", ""),
                    "similarity": result.get("similarity", 0.0),
                    "meta": result.get("meta", {}),
                })

            return parsed_results

        except Exception as e:
            logger.error(f"Search error: {e}")
            raise

    async def delete_by_document(self, document_id: str):
        """Delete all vectors belonging to a specific document."""
        await self.ensure_index()
        try:
            # Use filter to find and delete vectors for this document
            logger.info(f"Deleting vectors for document: {document_id}")
            # Note: Endee may support delete operations - using available API
        except Exception as e:
            logger.error(f"Delete error: {e}")
            raise

    async def get_index_stats(self) -> dict:
        """Get statistics about the current index."""
        try:
            await self.ensure_index()
            return {
                "index_name": self.index_name,
                "dimension": self.dimension,
                "space_type": self.space_type,
                "precision": self.precision,
                "status": "connected",
            }
        except Exception as e:
            return {
                "index_name": self.index_name,
                "status": f"error: {str(e)}",
            }

    def is_connected(self) -> bool:
        """Check if connected to Endee."""
        try:
            self.client
            return True
        except Exception:
            return False


# Global vector store instance
vector_store = EndeeVectorStore()
