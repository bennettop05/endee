"""
SmartRAG Configuration Module
Centralized configuration management using environment variables.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EndeeConfig:
    """Endee Vector Database configuration."""
    host: str = os.getenv("ENDEE_HOST", "http://localhost:8080")
    api_base: str = field(init=False)
    auth_token: str = os.getenv("ENDEE_AUTH_TOKEN", "")
    index_name: str = os.getenv("ENDEE_INDEX_NAME", "smartrag_docs")
    dimension: int = 384  # all-MiniLM-L6-v2 output dimension
    space_type: str = "cosine"
    precision: str = "INT8"

    def __post_init__(self):
        self.api_base = f"{self.host}/api/v1"


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    dimension: int = 384
    batch_size: int = 32


@dataclass
class ChunkingConfig:
    """Document chunking configuration."""
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    min_chunk_size: int = 50


@dataclass
class LLMConfig:
    """LLM configuration for RAG."""
    provider: str = os.getenv("LLM_PROVIDER", "gemini")
    api_key: str = os.getenv("GEMINI_API_KEY", "")
    model_name: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    top_k_results: int = int(os.getenv("TOP_K_RESULTS", "5"))


@dataclass
class AppConfig:
    """Main application configuration."""
    endee: EndeeConfig = field(default_factory=EndeeConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    upload_dir: str = os.getenv("UPLOAD_DIR", "./data/uploads")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


# Global configuration instance
config = AppConfig()
