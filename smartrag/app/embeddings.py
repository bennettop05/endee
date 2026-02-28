"""
SmartRAG Embedding Module
Handles text embedding generation using Sentence Transformers.
"""

import logging
from typing import Union
from sentence_transformers import SentenceTransformer
from app.config import config

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Generates dense vector embeddings using Sentence Transformers.
    Uses 'all-MiniLM-L6-v2' by default (384 dimensions, fast & accurate).
    """

    def __init__(self):
        self.model_name = config.embedding.model_name
        self.dimension = config.embedding.dimension
        self.batch_size = config.embedding.batch_size
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded successfully. Dimension: {self.dimension}")
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text string.

        Args:
            text: Input text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batch.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        logger.info(f"Embedding {len(texts)} texts in batches of {self.batch_size}...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        logger.info(f"Embedding complete. Generated {len(embeddings)} vectors.")
        return [emb.tolist() for emb in embeddings]

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None


# Global embedding engine instance
embedding_engine = EmbeddingEngine()
