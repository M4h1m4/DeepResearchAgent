from contextlib import contextmanager 
from typing import List, Dict, Any, Optional 

from mcp.server.fastmcp import FastMCP 

from app.database import db 
from app.services.rag_service import RAGService 
from app.services.deep_research_service import DeepResearchService 
from config.logging_config import get_logger 

logger = get_logger(__name__)

mcp = FastMCP(
    "Deep Research Agent",
    json_response=True,
    host="127.0.0.1",
    port=8001,
)

@contextmanager 
def _db_session():
    gen = db.get_session()
    session = next(gen)
    try:
        yield session 
    finally:
        session.close()

def _get_rag_service() -> RAGService:
    """ 
    Lazy Singleton Service that is they are created only upon the first call 
    and then the same instance is used. One Instance per one servive. 
    Why are we not doing this at startup: The server starts quickly; 
    the cost of creating the services is paid on first use instead of at import/startup.
    """
    if not hasattr(_get_rag_service, "_instance"):
        _get_rag_service._instance = RAGService()
    return _get_rag_service._instance

def _get_deep_research_service() -> DeepResearchService:
    if not hasattr(_get_deep_research_service, "_instance"):
        _get_deep_research_service._instance = DeepResearchService()
    return _get_deep_research_service._instance

@mcp.tool()
def rag_query(query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
    logger.info("MCP tool rag_query called", extra ={"query": query[:200]})

    try:
        rag = _get_rag_service()
        with _db_session() as session:
            result = rag.query(
                db=session,
                query=query,
                top_k=top_k,
                filter_dict=None,
            )
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "response_time_ms": result.get("response_time_ms", 0),
        }
    except Exception as e:
        logger.exception("MCP rag_query failed")
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "retrieved_chunks": [],
            "response_time_ms": 0,
        }

@mcp.tool()
def deep_research(query: str) -> Dict[str, Any]:
    logger.info("MCP tool deep_research called", extra={"query": query[:200]})
    try:
        service = _get_deep_research_service()
        result = service.research(query)
        sources = result.get("sources", [])
        # Normalize source dicts for JSON (e.g. SourceInfo-like)
        source_list = [
            {"id": s.get("id"), "title": s.get("title"), "source": s.get("source")}
            if isinstance(s, dict)
            else {"id": getattr(s, "id", None), "title": getattr(s, "title", ""), "source": getattr(s, "source", "")}
            for s in sources
        ]
        return {
            "answer": result.get("answer", ""),
            "sources": source_list,
            "chunk_ids": result.get("chunk_ids", []),
            "response_time_ms": result.get("response_time_ms", 0),
            "research_metadata": {
                "mode": "deep",
                "iterations": result.get("iteration_count", 0),
                "sub_queries": result.get("sub_queries", []),
                "research_plan": result.get("research_plan", ""),
            },
        }
    except Exception as e:
        logger.exception("MCP deep_research failed")
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "chunk_ids": [],
            "response_time_ms": 0,
            "research_metadata": {},
        }


def main() -> None:
    """Run MCP server. Use --http for streamable-http; default is stdio."""
    import sys
    if "--http" in sys.argv:
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
