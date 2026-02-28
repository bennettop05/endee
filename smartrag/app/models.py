"""
SmartRAG Data Models
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── Document Models ──────────────────────────────────────────────────

class DocumentUpload(BaseModel):
    """Model for document upload metadata."""
    filename: str
    content_type: str
    file_size: int


class DocumentChunk(BaseModel):
    """Represents a single chunk of a document."""
    chunk_id: str
    document_id: str
    document_name: str
    content: str
    chunk_index: int
    total_chunks: int
    metadata: dict = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    """Information about an ingested document."""
    document_id: str
    filename: str
    total_chunks: int
    file_type: str
    file_size: int
    ingested_at: str
    status: str = "active"


# ── Query Models ─────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Model for a RAG query request."""
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_filter: Optional[str] = None
    include_sources: bool = True


class SourceChunk(BaseModel):
    """A source chunk returned with the answer."""
    document_name: str
    chunk_content: str
    similarity_score: float
    chunk_index: int


class QueryResponse(BaseModel):
    """Model for a RAG query response."""
    question: str
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    total_sources_found: int
    processing_time_ms: float
    model_used: str


# ── Search Models ────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """Model for a semantic search request (without LLM)."""
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=50)
    document_filter: Optional[str] = None


class SearchResult(BaseModel):
    """A single semantic search result."""
    document_name: str
    chunk_content: str
    similarity_score: float
    metadata: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Model for semantic search response."""
    query: str
    results: list[SearchResult]
    total_results: int
    processing_time_ms: float


# ── System Models ────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    endee_connected: bool
    embedding_model_loaded: bool
    llm_configured: bool
    total_documents: int
    total_chunks: int


class StatsResponse(BaseModel):
    """System statistics response."""
    total_documents: int
    total_chunks: int
    index_info: dict = Field(default_factory=dict)
    embedding_model: str
    llm_model: str
