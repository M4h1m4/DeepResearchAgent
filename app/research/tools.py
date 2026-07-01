from typing import Dict, Any, Optional
from contextvars import ContextVar
from contextlib import contextmanager
from langchain_core.tools import tool, ToolException

from app.services.rag_service import RAGService
from app.database import db
from config import settings
from config.logging_config import get_logger

logger = get_logger(__name__)

_current_model: ContextVar[Optional[str]] = ContextVar("current_model", default=None)
_current_session_id: ContextVar[Optional[str]] = ContextVar("current_session_id", default=None)


def set_request_context(model: Optional[str], session_id: Optional[str] = None) -> None:
    _current_model.set(model)
    _current_session_id.set(session_id)


_rag_service: Optional[RAGService] = None


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


@tool
def research_rag_tool(query: str) -> Dict[str, Any]:
    """Execute a RAG query to retrieve relevant information from the knowledge base.

    Args:
        query: The search query to execute against the knowledge base.

    Returns:
        A dictionary containing the answer, sources, retrieved chunk IDs, and the original query.
    """
    model = _current_model.get()
    session_id = _current_session_id.get()

    logger.info("RAG tool called", extra={"tool_query": query, "session_id": session_id})
    try:
        rag_service = get_rag_service()
        with get_db_session() as session:
            result = rag_service.query(
                db=session,
                query=query,
                top_k=settings.deep_research_top_k,
                model=model,
                session_id=session_id,
            )
        logger.info(
            "RAG tool completed",
            extra={"tool_query": query, "sources_count": len(result.get("sources", []))},
        )
        return {
            "answer": result.get("answer"),
            "sources": result.get("sources", []),
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "chunk_ids": result.get("retrieved_chunks", []),
            "query": query,
        }
    except Exception as e:
        logger.error("RAG tool error", extra={"error": str(e)}, exc_info=True)
        raise ToolException(f"Error in RAG tool: {str(e)}")


# ---------------------------------------------------------------------------
# Web search tool (Tavily)
# ---------------------------------------------------------------------------

_tavily_client = None


def _get_tavily_client():
    global _tavily_client
    if _tavily_client is not None:
        return _tavily_client
    if not settings.tavily_api_key:
        return None
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        _tavily_client = TavilySearchResults(max_results=5, tavily_api_key=settings.tavily_api_key)
        logger.info("Tavily web search client initialised")
    except Exception as e:
        logger.warning("Could not initialise Tavily", extra={"error": str(e)})
        _tavily_client = None
    return _tavily_client


@tool
def web_search_tool(query: str) -> Dict[str, Any]:
    """Search the web for current information not available in the knowledge base.

    Args:
        query: The search query to send to the web.

    Returns:
        A dictionary with web search results, a combined answer snippet, and the query.
    """
    logger.info("Web search tool called", extra={"query": query})
    client = _get_tavily_client()

    if client is None:
        logger.warning("Web search unavailable — no TAVILY_API_KEY configured")
        return {"answer": "Web search is not available (TAVILY_API_KEY not configured).", "results": [], "sources": [], "query": query}

    try:
        raw_results = client.invoke(query)
        sources, snippets = [], []
        for r in raw_results:
            if isinstance(r, dict):
                url = r.get("url", "")
                content = r.get("content", "")
                title = r.get("title") or url
                sources.append({"id": None, "title": title, "source": url, "type": "web"})
                if content:
                    snippets.append(f"[{title}]\n{content}")

        combined_answer = "\n\n".join(snippets) if snippets else "No web results found."
        logger.info("Web search complete", extra={"result_count": len(sources)})
        return {"answer": combined_answer, "results": raw_results, "sources": sources, "query": query}
    except Exception as e:
        logger.error("Web search error", extra={"error": str(e)}, exc_info=True)
        raise ToolException(f"Error in web search tool: {str(e)}")


research_tools = [research_rag_tool, web_search_tool]
