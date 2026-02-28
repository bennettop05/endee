"""
SmartRAG FastAPI Application
REST API for document ingestion, semantic search, and RAG Q&A.
"""

import os
import shutil
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import config
from app.models import (
    QueryRequest, QueryResponse,
    SearchRequest, SearchResponse,
    DocumentInfo, HealthResponse, StatsResponse,
)
from app.vector_store import vector_store
from app.embeddings import embedding_engine
from app.ingestion import ingestion_pipeline
from app.rag_engine import rag_engine

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s │ %(name)-20s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: startup and shutdown."""
    # Startup
    logger.info("=" * 60)
    logger.info("🧠 SmartRAG — Intelligent Knowledge Base")
    logger.info("   Powered by Endee Vector Database")
    logger.info("=" * 60)

    # Ensure upload directory exists
    os.makedirs(config.upload_dir, exist_ok=True)

    # Pre-load embedding model
    logger.info("Loading embedding model...")
    _ = embedding_engine.model
    logger.info("Embedding model ready!")

    # Connect to Endee
    logger.info(f"Connecting to Endee at {config.endee.host}...")
    try:
        await vector_store.ensure_index()
        logger.info("Endee vector store ready!")
    except Exception as e:
        logger.warning(f"Could not connect to Endee: {e}")
        logger.warning("Start Endee with 'docker compose up -d' and restart SmartRAG.")

    logger.info("SmartRAG is ready! 🚀")
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("SmartRAG shutting down...")


# Create FastAPI app
app = FastAPI(
    title="SmartRAG API",
    description="Intelligent Knowledge Base & Q&A powered by Endee Vector Database and Google Gemini",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health & Info Endpoints ────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "SmartRAG API",
        "version": "1.0.0",
        "description": "Intelligent Knowledge Base powered by Endee Vector Database",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "upload": "POST /api/v1/documents/upload",
            "query": "POST /api/v1/query",
            "search": "POST /api/v1/search",
            "documents": "GET /api/v1/documents",
            "stats": "GET /api/v1/stats",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Check system health."""
    endee_connected = vector_store.is_connected()
    model_loaded = embedding_engine.is_loaded()
    llm_configured = rag_engine.is_llm_configured()
    documents = ingestion_pipeline.get_documents()

    total_chunks = sum(d.total_chunks for d in documents)

    return HealthResponse(
        status="healthy" if endee_connected else "degraded",
        endee_connected=endee_connected,
        embedding_model_loaded=model_loaded,
        llm_configured=llm_configured,
        total_documents=len(documents),
        total_chunks=total_chunks,
    )


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["Info"])
async def get_stats():
    """Get system statistics."""
    documents = ingestion_pipeline.get_documents()
    total_chunks = sum(d.total_chunks for d in documents)
    index_info = await vector_store.get_index_stats()

    return StatsResponse(
        total_documents=len(documents),
        total_chunks=total_chunks,
        index_info=index_info,
        embedding_model=config.embedding.model_name,
        llm_model=config.llm.model_name,
    )


# ── Document Endpoints ─────────────────────────────────────────────────

@app.post("/api/v1/documents/upload", response_model=DocumentInfo, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document into the knowledge base.
    Supported formats: PDF, DOCX, TXT, MD
    """
    # Validate file type
    allowed_extensions = {".pdf", ".docx", ".txt", ".md", ".csv"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_extensions)}"
        )

    # Save uploaded file
    file_path = os.path.join(config.upload_dir, file.filename or "upload.txt")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Ingest the document
        doc_info = await ingestion_pipeline.ingest_file(
            file_path=file_path,
            filename=file.filename or "upload.txt",
            content_type=file.content_type or "",
        )
        return doc_info

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/documents/text", response_model=DocumentInfo, tags=["Documents"])
async def ingest_text(text: str, name: str = "manual_input"):
    """Ingest raw text directly into the knowledge base."""
    try:
        doc_info = await ingestion_pipeline.ingest_text(text=text, document_name=name)
        return doc_info
    except Exception as e:
        logger.error(f"Text ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/documents", response_model=list[DocumentInfo], tags=["Documents"])
async def list_documents():
    """List all ingested documents."""
    return ingestion_pipeline.get_documents()


# ── RAG Query Endpoints ────────────────────────────────────────────────

@app.post("/api/v1/query", response_model=QueryResponse, tags=["RAG"])
async def rag_query(request: QueryRequest):
    """
    Ask a question and get an AI-generated answer grounded in the knowledge base.
    Uses semantic search + Google Gemini for RAG.
    """
    try:
        response = await rag_engine.query(request)
        return response
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/search", response_model=SearchResponse, tags=["Search"])
async def semantic_search(request: SearchRequest):
    """
    Perform semantic search on the knowledge base (no LLM generation).
    Returns ranked document chunks by similarity.
    """
    try:
        response = await rag_engine.search(request)
        return response
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
