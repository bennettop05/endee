<p align="center">
  <h1 align="center">🧠 SmartRAG</h1>
  <p align="center">
    <strong>Intelligent Knowledge Base & Q&A System</strong><br>
    Powered by <a href="https://github.com/endee-io/endee">Endee Vector Database</a> and Google Gemini
  </p>
  <p align="center">
    <a href="#-features">Features</a> •
    <a href="#-architecture">Architecture</a> •
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-how-endee-is-used">Endee Usage</a> •
    <a href="#-demo">Demo</a> •
    <a href="#-api-reference">API</a>
  </p>
</p>

---

## 📋 Project Overview

**SmartRAG** is an intelligent document Question & Answer system that implements **Retrieval-Augmented Generation (RAG)** using the [Endee](https://github.com/endee-io/endee) vector database as its core semantic search engine.

### Problem Statement

Organizations and individuals accumulate vast amounts of knowledge in documents (PDFs, reports, notes, articles) but struggle to efficiently retrieve specific information when needed. Traditional keyword search fails when users phrase questions differently from how information is stored. SmartRAG solves this by:

1. **Indexing documents** as semantic vector embeddings in Endee
2. **Understanding meaning** — not just keywords — when you ask questions
3. **Generating accurate answers** grounded in your actual documents, with source citations

### Use Case: RAG (Retrieval-Augmented Generation)

RAG is a powerful AI pattern that enhances LLM responses by first retrieving relevant context from a knowledge base. This approach:
- **Reduces hallucinations** — answers are grounded in real documents
- **Enables private knowledge** — works with your own documents, not just public data
- **Provides citations** — every answer comes with source references
- **Stays current** — add new documents anytime without retraining models

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **Multi-Format Upload** | Support for PDF, DOCX, TXT, MD, CSV documents |
| 🧩 **Smart Chunking** | Sentence-aware text splitting with configurable overlap |
| 🔍 **Semantic Search** | Meaning-based search powered by Endee's HNSW algorithm |
| 💬 **RAG Q&A** | AI-powered answers with source citations via Google Gemini |
| 🏷️ **Metadata Filtering** | Filter search results by document name, type, etc. |
| 📊 **Dashboard** | Real-time system monitoring and statistics |
| 🐳 **One-Command Deploy** | Full stack via Docker Compose (Endee + API + UI) |
| 🔐 **Auth Support** | Optional Endee authentication token |
| ⚡ **High Performance** | Endee handles up to 1B vectors on a single node |

---

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SmartRAG Architecture                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐     ┌──────────────────────────────────────────┐ │
│  │   Streamlit   │────▶│           FastAPI Backend               │ │
│  │   Frontend    │◀────│         (REST API Server)               │ │
│  │   (Port 8501) │     │           (Port 8000)                   │ │
│  └──────────────┘     └───────┬──────────┬──────────┬───────────┘ │
│                               │          │          │              │
│                    ┌──────────▼──┐ ┌─────▼─────┐ ┌─▼───────────┐ │
│                    │  Ingestion  │ │  Embedding │ │  RAG Engine │ │
│                    │  Pipeline   │ │   Engine   │ │  (Gemini)   │ │
│                    │             │ │(MiniLM-L6) │ │             │ │
│                    └──────┬──────┘ └─────┬──────┘ └──────┬──────┘ │
│                           │              │               │         │
│                    ┌──────▼──────────────▼───────────────▼──────┐  │
│                    │        Endee Vector Database                │  │
│                    │          (Port 8080)                        │  │
│                    │   ┌─────────────────────────────────┐      │  │
│                    │   │  HNSW Index: smartrag_docs      │      │  │
│                    │   │  Dimension: 384 (cosine)        │      │  │
│                    │   │  Precision: INT8                 │      │  │
│                    │   │  Metadata: doc_name, content,   │      │  │
│                    │   │           chunk_idx, file_type   │      │  │
│                    │   └─────────────────────────────────┘      │  │
│                    └────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### RAG Pipeline Flow

```
User Question
     │
     ▼
┌─────────────────┐
│ 1. Embed Query  │  ← Sentence Transformers (all-MiniLM-L6-v2)
│    (384-dim)    │     Converts question to semantic vector
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Vector Search│  ← Endee HNSW Similarity Search
│    in Endee     │     Finds top-K most relevant document chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Context      │  ← Retrieved chunks assembled as context
│    Assembly     │     With source document attribution
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. LLM Generate │  ← Google Gemini generates grounded answer
│    Answer       │     Strictly from provided context only
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. Response     │  ← Answer + source citations + similarity
│    with Sources │     scores returned to user
└─────────────────┘
```

### Document Ingestion Flow

```
Document Upload (PDF/DOCX/TXT)
     │
     ▼
┌─────────────────┐
│ 1. Parse        │  ← PyPDF2, python-docx, or plain text
│    Document     │     Extract raw text content
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Smart        │  ← Sentence-aware splitting
│    Chunking     │     500 chars per chunk, 100 char overlap
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Batch        │  ← Sentence Transformers batch encoding
│    Embedding    │     32 chunks per batch
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Upsert to    │  ← Endee SDK: index.upsert()
│    Endee        │     Vectors + metadata stored
└─────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vector Database** | [Endee](https://github.com/endee-io/endee) | High-performance vector storage & similarity search |
| **Backend** | Python 3.11 + FastAPI | REST API server |
| **Frontend** | Streamlit | Interactive web dashboard |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) | Text → 384-dim vector conversion |
| **LLM** | Google Gemini (`gemini-2.0-flash`) | RAG answer generation |
| **Doc Parsing** | PyPDF2, python-docx | PDF and DOCX text extraction |
| **Containerization** | Docker + Docker Compose | One-command deployment |

---

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose** — [Install Docker](https://docs.docker.com/get-docker/)
- **Python 3.10+** (for local dev, optional if using Docker)
- **Google Gemini API Key** (free) — [Get one here](https://aistudio.google.com/apikey)

### Option 1: Docker Compose (Recommended) 🐳

This starts everything — Endee, SmartRAG API, and the web UI — with one command.

```bash
# 1. Clone this repository (your fork)
git clone https://github.com/bennettop05/endee.git
cd endee/smartrag

# 2. Set your Gemini API key
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here

# 3. Launch the full stack
docker compose up -d

# 4. Open the web UI
# Frontend: http://localhost:8501
# API Docs: http://localhost:8000/docs
# Endee DB: http://localhost:8080
```

### Option 2: Local Development 🖥️

```bash
# 1. Start Endee using Docker
docker run -d \
  -p 8080:8080 \
  -v endee-data:/data \
  --name endee-server \
  endeeio/endee-server:latest

# 2. Set up Python environment
cd smartrag
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your GEMINI_API_KEY

# 5. Start the backend API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 6. (New terminal) Start the Streamlit frontend
streamlit run frontend/streamlit_app.py
```

### Option 3: Quick API Test 🧪

```bash
# Check health
curl http://localhost:8000/health

# Upload a document
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@data/sample_docs/ai_overview.txt"

# Ask a question (RAG)
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG and how does it work?", "top_k": 5}'

# Semantic search (no LLM)
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "vector database architecture", "top_k": 10}'
```

---

## 🗂️ Project Structure

```
endee/                              # Forked Endee repository
├── smartrag/                       # SmartRAG application
│   ├── app/                        # Backend application
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app with REST endpoints
│   │   ├── config.py               # Centralized configuration
│   │   ├── models.py               # Pydantic request/response models
│   │   ├── embeddings.py           # Sentence Transformers embedding engine
│   │   ├── ingestion.py            # Document parsing & chunking pipeline
│   │   ├── vector_store.py         # Endee SDK wrapper
│   │   └── rag_engine.py           # RAG query engine (Search + Gemini)
│   ├── frontend/
│   │   └── streamlit_app.py        # Streamlit web interface
│   ├── data/
│   │   └── sample_docs/            # Sample documents for demo
│   │       ├── ai_overview.txt
│   │       └── vector_databases.txt
│   ├── tests/
│   │   └── test_pipeline.py        # Unit tests
│   ├── Dockerfile                  # Application container
│   ├── docker-compose.yml          # Full stack orchestration
│   ├── requirements.txt            # Python dependencies
│   ├── .env.example                # Environment variable template
│   └── .gitignore
├── README.md                       # ← You are here
└── ... (Endee source files)
```

---

## 🔗 How Endee Is Used

Endee is the **core component** of SmartRAG, serving as the vector database that powers all semantic search operations. Here's exactly how it's integrated:

### 1. Index Creation

When the application starts, it creates (or connects to) an Endee index:

```python
from endee import Endee, Precision

client = Endee()
client.set_base_url("http://localhost:8080/api/v1")

# Create index optimized for semantic search
client.create_index(
    name="smartrag_docs",
    dimension=384,            # Matches embedding model output
    space_type="cosine",      # Best for text similarity
    precision=Precision.INT8, # Memory-efficient quantization
)
```

### 2. Document Ingestion (Upsert)

After documents are parsed, chunked, and embedded, vectors are stored in Endee with metadata:

```python
index = client.get_index(name="smartrag_docs")

# Upsert document chunks with metadata
index.upsert([
    {
        "id": "doc1_chunk_0",
        "vector": [0.021, -0.034, ...],  # 384-dim embedding
        "meta": {
            "document_name": "ai_overview.pdf",
            "content": "Artificial Intelligence is a branch of...",
            "chunk_index": 0,
            "file_type": ".pdf",
            "ingested_at": "2024-01-15T10:30:00Z"
        }
    },
    # ... more chunks
])
```

### 3. Semantic Search (Query)

When a user asks a question, the query is embedded and searched against Endee:

```python
# Embed the user's question
query_vector = embedding_model.encode("What is deep learning?")

# Search Endee for the most similar document chunks
results = index.query(
    vector=query_vector.tolist(),
    top_k=5,  # Return top 5 most similar chunks
)

# Results include similarity scores and metadata
for result in results:
    print(f"Score: {result['similarity']:.4f}")
    print(f"Document: {result['meta']['document_name']}")
    print(f"Content: {result['meta']['content']}")
```

### 4. Metadata Filtering

Endee's filtering capabilities are used to scope searches to specific documents:

```python
# Search only within a specific document
results = index.query(
    vector=query_vector.tolist(),
    top_k=5,
    filter={"document_name": {"$eq": "quarterly_report.pdf"}}
)
```

### Why Endee?

| Feature | Benefit for SmartRAG |
|---------|---------------------|
| **HNSW Algorithm** | Sub-millisecond similarity search across thousands of chunks |
| **Cosine Distance** | Best metric for comparing text embeddings |
| **INT8 Precision** | 4x less memory than FLOAT32, with minimal accuracy loss |
| **Metadata Support** | Store chunk content and document info alongside vectors |
| **Filtering** | Scope searches to specific documents or types |
| **Docker Support** | Easy deployment as part of our Docker Compose stack |
| **Python SDK** | Clean, intuitive API for all vector operations |

---

## 📡 API Reference

### Document Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/documents/upload` | Upload and ingest a document (PDF/DOCX/TXT) |
| `POST` | `/api/v1/documents/text` | Ingest raw text directly |
| `GET` | `/api/v1/documents` | List all ingested documents |

### Query Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/query` | RAG Q&A — ask a question, get an AI answer with sources |
| `POST` | `/api/v1/search` | Semantic search — find relevant chunks without LLM |

### System Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | System health check |
| `GET` | `/api/v1/stats` | System statistics and index info |

### Example: RAG Query

**Request:**
```json
{
  "question": "How does HNSW algorithm work in vector databases?",
  "top_k": 5,
  "include_sources": true
}
```

**Response:**
```json
{
  "question": "How does HNSW algorithm work in vector databases?",
  "answer": "HNSW (Hierarchical Navigable Small World) is an approximate nearest neighbor algorithm that creates a multi-layered graph structure...",
  "sources": [
    {
      "document_name": "vector_databases.txt",
      "chunk_content": "The most popular ANN algorithm is HNSW...",
      "similarity_score": 0.8923,
      "chunk_index": 3
    }
  ],
  "total_sources_found": 5,
  "processing_time_ms": 342.15,
  "model_used": "gemini-2.0-flash"
}
```

---

## 🧪 Running Tests

```bash
cd smartrag
pip install pytest
python -m pytest tests/ -v
```

---

## ⚙️ Configuration

All settings are configured via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `ENDEE_HOST` | `http://localhost:8080` | Endee server URL |
| `ENDEE_AUTH_TOKEN` | `` | Optional authentication token |
| `ENDEE_INDEX_NAME` | `smartrag_docs` | Name of the vector index |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformer model |
| `GEMINI_API_KEY` | `` | Google Gemini API key |
| `LLM_MODEL` | `gemini-2.0-flash` | Gemini model to use |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Default search results count |

---

## 🤝 Contributing

1. Fork this repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -am 'Add my feature'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project uses the Endee vector database under its original license. The SmartRAG application code is provided for educational and evaluation purposes.

---

<p align="center">
  <strong>Built with ❤️ using <a href="https://github.com/endee-io/endee">Endee Vector Database</a></strong>
</p>
