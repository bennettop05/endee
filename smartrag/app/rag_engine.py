"""
SmartRAG RAG Engine
Core Retrieval-Augmented Generation engine that combines
Endee vector search with Google Gemini for grounded Q&A.
"""

import time
import logging
from typing import Optional

import google.generativeai as genai

from app.config import config
from app.embeddings import embedding_engine
from app.vector_store import vector_store
from app.models import QueryRequest, QueryResponse, SourceChunk, SearchRequest, SearchResponse, SearchResult

logger = logging.getLogger(__name__)


RAG_SYSTEM_PROMPT = """You are SmartRAG, an intelligent question-answering assistant. Your job is to answer the user's question accurately using ONLY the provided context chunks from the knowledge base.

RULES:
1. Answer based STRICTLY on the provided context. Do not make up information.
2. If the context doesn't contain enough information to answer, say "I don't have enough information in the knowledge base to answer this question."
3. Be concise but thorough. Provide clear, well-structured answers.
4. When referencing information, mention which source document it came from.
5. Use markdown formatting for readability (headers, bullet points, bold text).
6. If the question is about comparing multiple topics, organize your answer clearly.

CONTEXT CHUNKS:
{context}

USER QUESTION: {question}

ANSWER:"""


class RAGEngine:
    """
    Retrieval-Augmented Generation engine.
    
    Pipeline:
    1. Embed the user's question
    2. Search Endee for relevant document chunks
    3. Construct a prompt with retrieved context
    4. Generate an answer using Google Gemini
    5. Return the answer with source citations
    """

    def __init__(self):
        self._llm = None
        self._configure_llm()

    def _configure_llm(self):
        """Configure the Google Gemini LLM."""
        api_key = config.llm.api_key
        if api_key:
            genai.configure(api_key=api_key)
            self._llm = genai.GenerativeModel(config.llm.model_name)
            logger.info(f"LLM configured: {config.llm.model_name}")
        else:
            logger.warning("No GEMINI_API_KEY set. RAG Q&A will not work.")

    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        Execute a full RAG query pipeline.

        Args:
            request: The query request with question and parameters.

        Returns:
            QueryResponse with answer and source citations.
        """
        start_time = time.time()

        # Step 1: Embed the question
        logger.info(f"RAG Query: '{request.question[:80]}...'")
        query_embedding = embedding_engine.embed_text(request.question)

        # Step 2: Search Endee for relevant chunks
        filters = None
        if request.document_filter:
            filters = {
                "document_name": {"$eq": request.document_filter}
            }

        search_results = await vector_store.search(
            query_vector=query_embedding,
            top_k=request.top_k,
            filters=filters,
        )

        # Step 3: Extract context from results
        sources = []
        context_parts = []

        for i, result in enumerate(search_results):
            meta = result.get("meta", {})
            content = meta.get("content", "")
            doc_name = meta.get("document_name", "Unknown")
            chunk_idx = meta.get("chunk_index", 0)
            similarity = result.get("similarity", 0.0)

            context_parts.append(
                f"[Source {i + 1}: {doc_name} (Chunk {chunk_idx + 1})]:\n{content}"
            )

            sources.append(SourceChunk(
                document_name=doc_name,
                chunk_content=content[:500],  # Truncate for response
                similarity_score=round(similarity, 4),
                chunk_index=chunk_idx,
            ))

        # Step 4: Generate answer with LLM
        if not self._llm:
            answer = self._format_sources_only(context_parts)
        else:
            context_str = "\n\n---\n\n".join(context_parts)
            prompt = RAG_SYSTEM_PROMPT.format(
                context=context_str,
                question=request.question
            )

            try:
                response = self._llm.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=config.llm.max_tokens,
                        temperature=config.llm.temperature,
                    )
                )
                answer = response.text
            except Exception as e:
                logger.error(f"LLM generation error: {e}")
                answer = f"Error generating answer: {str(e)}\n\nRelevant sources were found. Please check the source citations below."

        # Step 5: Build response
        processing_time = (time.time() - start_time) * 1000

        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources if request.include_sources else [],
            total_sources_found=len(search_results),
            processing_time_ms=round(processing_time, 2),
            model_used=config.llm.model_name if self._llm else "none",
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Perform semantic search only (no LLM generation).

        Args:
            request: The search request.

        Returns:
            SearchResponse with ranked results.
        """
        start_time = time.time()

        # Embed the query
        query_embedding = embedding_engine.embed_text(request.query)

        # Search Endee
        filters = None
        if request.document_filter:
            filters = {
                "document_name": {"$eq": request.document_filter}
            }

        search_results = await vector_store.search(
            query_vector=query_embedding,
            top_k=request.top_k,
            filters=filters,
        )

        # Format results
        results = []
        for result in search_results:
            meta = result.get("meta", {})
            results.append(SearchResult(
                document_name=meta.get("document_name", "Unknown"),
                chunk_content=meta.get("content", ""),
                similarity_score=round(result.get("similarity", 0.0), 4),
                metadata={
                    "chunk_index": meta.get("chunk_index", 0),
                    "total_chunks": meta.get("total_chunks", 0),
                    "file_type": meta.get("file_type", ""),
                    "ingested_at": meta.get("ingested_at", ""),
                },
            ))

        processing_time = (time.time() - start_time) * 1000

        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            processing_time_ms=round(processing_time, 2),
        )

    def _format_sources_only(self, context_parts: list[str]) -> str:
        """Format answer when no LLM is available."""
        if not context_parts:
            return "No relevant information found in the knowledge base."

        answer = "**LLM not configured.** Here are the most relevant passages from the knowledge base:\n\n"
        for i, part in enumerate(context_parts):
            answer += f"### Result {i + 1}\n{part}\n\n"
        return answer

    def is_llm_configured(self) -> bool:
        """Check if the LLM is configured."""
        return self._llm is not None


# Global RAG engine instance
rag_engine = RAGEngine()
