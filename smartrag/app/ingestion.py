"""
SmartRAG Document Ingestion Pipeline
Handles document parsing, chunking, and ingestion into the vector store.
"""

import os
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from app.config import config
from app.models import DocumentChunk, DocumentInfo
from app.embeddings import embedding_engine
from app.vector_store import vector_store

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits documents into overlapping chunks for embedding.
    Uses a sliding window approach for context preservation.
    """

    def __init__(
        self,
        chunk_size: int = config.chunking.chunk_size,
        chunk_overlap: int = config.chunking.chunk_overlap,
        min_chunk_size: int = config.chunking.min_chunk_size
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks using sentence-aware boundaries.

        Args:
            text: The full document text.

        Returns:
            List of text chunks.
        """
        # Clean the text
        text = text.strip()
        if not text:
            return []

        # Split by sentences first for cleaner boundaries
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            if current_length + sentence_len > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk).strip()
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(chunk_text)

                # Calculate overlap — keep last few sentences for context
                overlap_chars = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    if overlap_chars + len(s) > self.chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_chars += len(s)

                current_chunk = overlap_sentences
                current_length = overlap_chars

            current_chunk.append(sentence)
            current_length += sentence_len

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)

        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using simple heuristics."""
        import re
        # Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Further split very long sentences on newlines
        result = []
        for sentence in sentences:
            if len(sentence) > self.chunk_size:
                sub_sentences = sentence.split('\n')
                result.extend([s.strip() for s in sub_sentences if s.strip()])
            else:
                result.append(sentence.strip())
        return [s for s in result if s]


class DocumentParser:
    """Parses different document formats into plain text."""

    @staticmethod
    def parse_file(file_path: str, content_type: str = "") -> str:
        """
        Parse a file and extract its text content.

        Args:
            file_path: Path to the file.
            content_type: MIME type of the file.

        Returns:
            Extracted text content.
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return DocumentParser._parse_pdf(file_path)
        elif ext == ".docx":
            return DocumentParser._parse_docx(file_path)
        elif ext in (".txt", ".md", ".csv"):
            return DocumentParser._parse_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    @staticmethod
    def _parse_pdf(file_path: str) -> str:
        """Extract text from PDF files."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except ImportError:
            logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")
            raise

    @staticmethod
    def _parse_docx(file_path: str) -> str:
        """Extract text from DOCX files."""
        try:
            from docx import Document
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except ImportError:
            logger.warning("python-docx not installed. Install with: pip install python-docx")
            raise

    @staticmethod
    def _parse_text(file_path: str) -> str:
        """Read plain text files."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline:
    Parse → Chunk → Embed → Store in Endee
    """

    def __init__(self):
        self.chunker = TextChunker()
        self.parser = DocumentParser()
        self.documents: dict[str, DocumentInfo] = {}

    async def ingest_file(
        self,
        file_path: str,
        filename: str,
        content_type: str = "",
    ) -> DocumentInfo:
        """
        Ingest a single file into the vector store.

        Args:
            file_path: Path to the file on disk.
            filename: Original filename.
            content_type: MIME type.

        Returns:
            DocumentInfo with ingestion details.
        """
        document_id = str(uuid.uuid4())[:12]
        logger.info(f"Ingesting document: {filename} (ID: {document_id})")

        # Step 1: Parse the document
        logger.info("Step 1/4: Parsing document...")
        text = self.parser.parse_file(file_path, content_type)
        if not text.strip():
            raise ValueError(f"No text content extracted from {filename}")

        # Step 2: Chunk the text
        logger.info("Step 2/4: Chunking text...")
        chunks = self.chunker.chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks from document")

        # Step 3: Generate embeddings
        logger.info("Step 3/4: Generating embeddings...")
        embeddings = embedding_engine.embed_texts(chunks)

        # Step 4: Store in Endee
        logger.info("Step 4/4: Storing in Endee vector database...")
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{document_id}_chunk_{i}"
            vectors.append({
                "id": chunk_id,
                "vector": embedding,
                "meta": {
                    "document_id": document_id,
                    "document_name": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "content": chunk,
                    "file_type": os.path.splitext(filename)[1].lower(),
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                }
            })

        await vector_store.upsert_vectors(vectors)

        # Track document info
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        doc_info = DocumentInfo(
            document_id=document_id,
            filename=filename,
            total_chunks=len(chunks),
            file_type=os.path.splitext(filename)[1].lower(),
            file_size=file_size,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            status="active"
        )
        self.documents[document_id] = doc_info
        logger.info(f"Document ingested successfully: {filename} → {len(chunks)} chunks")

        return doc_info

    async def ingest_text(
        self,
        text: str,
        document_name: str = "manual_input",
    ) -> DocumentInfo:
        """
        Ingest raw text directly (no file parsing needed).

        Args:
            text: Raw text to ingest.
            document_name: Name to assign to this text.

        Returns:
            DocumentInfo with ingestion details.
        """
        document_id = str(uuid.uuid4())[:12]
        logger.info(f"Ingesting raw text: {document_name} (ID: {document_id})")

        # Chunk the text
        chunks = self.chunker.chunk_text(text)

        # Generate embeddings
        embeddings = embedding_engine.embed_texts(chunks)

        # Store in Endee
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{document_id}_chunk_{i}"
            vectors.append({
                "id": chunk_id,
                "vector": embedding,
                "meta": {
                    "document_id": document_id,
                    "document_name": document_name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "content": chunk,
                    "file_type": "text",
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                }
            })

        await vector_store.upsert_vectors(vectors)

        doc_info = DocumentInfo(
            document_id=document_id,
            filename=document_name,
            total_chunks=len(chunks),
            file_type="text",
            file_size=len(text),
            ingested_at=datetime.now(timezone.utc).isoformat(),
            status="active"
        )
        self.documents[document_id] = doc_info
        return doc_info

    def get_documents(self) -> list[DocumentInfo]:
        """Return list of all ingested documents."""
        return list(self.documents.values())


# Global ingestion pipeline instance
ingestion_pipeline = IngestionPipeline()
