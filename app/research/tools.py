"""
To wrap the RAG as a tool that can be called by the research agent
"""

from typing import Dict, Any 
from contextlib import contextmanager
from langchain_core.tools import tool, ToolException

from app.services.rag_service import RAGService
from app.database import db
from config.logging_config import get_logger 

logger = get_logger(__name__)

_rag_service = None 

def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service 

@contextmanager
def get_db_session():
    session_gen = db.get_session()
    session = next(session_gen)
    try:
        yield session
    finally:
        session.close()

@tool #makes the function callable by the agents
def research_rag_tool(query: str) -> Dict[str, Any]:
    """Execute a RAG query to retrieve relevant information from the knowledge base.
    
    Args:
        query: The search query to execute against the knowledge base.
        
    Returns:
        A dictionary containing:
        - answer: The generated answer based on retrieved context
        - sources: List of source documents with id, title, and source path
        - retrieved_chunks: List of retrieved chunk IDs
        - chunk_ids: List of chunk IDs used in the answer
        - query: The original query
    """
    logger.info(
        "RAG tool called",
        extra={
            "tool_query": query,
            "tool_name": "research_rag_tool"
        }
    )
    try:
        rag_service = get_rag_service()
        with get_db_session() as session:
            # Use optimized top_k for deep research mode
            top_k = getattr(settings, 'deep_research_top_k', 6)
            result = rag_service.query(
                db=session,
                query=query, 
                top_k=top_k,
            )

        has_sources = len(result.get("sources", [])) > 0 
        has_chunks = len(result.get("retrieved_chunks", [])) > 0 
        is_success = has_sources and has_chunks

        if is_success:
            logger.info(
                "RAG tool completed successfully",
                extra={
                    "tool_query": query,
                    "answer_length": len(result.get("answer", "")),
                    "sources_count": len(result.get("sources", [])),
                    "chunks_count": len(result.get("retrieved_chunks", []))
                }
            )
        else:
            logger.warning(
                "RAG tool completed with no results",
                extra={
                    "tool_query": query,
                    "answer": result.get("answer", "")[:200],  # Log first 200 chars of answer
                    "sources_count": len(result.get("sources", [])),
                    "chunks_count": len(result.get("retrieved_chunks", []))
                }
            )
        return {
            "answer": result.get("answer"),
            "sources": result.get("sources", []),
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "chunk_ids": result.get("chunk_ids", []),
            "query": query 
        }
    
    except Exception as e:
        logger.error(
            "RAG tool error",
            extra={
                "tool_query": query,
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True
        )
        # Raise ToolException so LangGraph can handle it
        raise ToolException(f"Error in RAG tool: {str(e)}")

research_tools = [research_rag_tool]