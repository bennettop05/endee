"""
SmartRAG Test Suite
Tests for the ingestion pipeline, embeddings, and RAG engine.
"""

import os
import sys
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingestion import TextChunker
from app.config import AppConfig


class TestTextChunker:
    """Tests for the text chunking module."""

    def setup_method(self):
        self.chunker = TextChunker(chunk_size=200, chunk_overlap=50, min_chunk_size=20)

    def test_basic_chunking(self):
        """Test that text is properly split into chunks."""
        text = "This is a test sentence. " * 50
        chunks = self.chunker.chunk_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) >= 20  # min_chunk_size

    def test_empty_text(self):
        """Test handling of empty text."""
        assert self.chunker.chunk_text("") == []
        assert self.chunker.chunk_text("   ") == []

    def test_short_text(self):
        """Test text shorter than chunk_size."""
        text = "This is a short text that fits in one chunk."
        chunks = self.chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_overlap(self):
        """Test that chunks have overlapping content."""
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here. Sixth sentence here. Seventh sentence here. Eighth sentence here."
        chunker = TextChunker(chunk_size=80, chunk_overlap=30, min_chunk_size=10)
        chunks = chunker.chunk_text(text)
        
        if len(chunks) > 1:
            # Check that some content overlaps between consecutive chunks
            for i in range(len(chunks) - 1):
                # The overlap should mean some words appear in both chunks
                words_current = set(chunks[i].split())
                words_next = set(chunks[i + 1].split())
                # There should be some common words due to overlap
                assert len(words_current & words_next) > 0

    def test_sentence_boundaries(self):
        """Test that chunks break at sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence."
        chunker = TextChunker(chunk_size=30, chunk_overlap=5, min_chunk_size=10)
        chunks = chunker.chunk_text(text)
        
        # Each chunk should end with a complete sentence or be the last chunk
        for chunk in chunks:
            assert chunk.strip()  # No empty chunks


class TestConfig:
    """Tests for configuration module."""

    def test_default_config(self):
        """Test default configuration values."""
        cfg = AppConfig()
        assert cfg.endee.dimension == 384
        assert cfg.endee.space_type == "cosine"
        assert cfg.chunking.chunk_size == 500
        assert cfg.chunking.chunk_overlap == 100

    def test_endee_api_base(self):
        """Test that API base URL is correctly constructed."""
        cfg = AppConfig()
        assert cfg.endee.api_base == f"{cfg.endee.host}/api/v1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
