import time
from typing import Dict, Any, Optional

from app.research.graph import get_research_graph
from app.research.state import ResearchState
from app.research.tools import set_request_context
from config.logging_config import get_logger

logger = get_logger(__name__)


class DeepResearchService:
    def __init__(self):
        self.research_graph = get_research_graph()
        logger.info("DeepResearchService initialized")

    def research(
        self,
        query: str,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        start_time = time.time()
        logger.info("Starting deep research", extra={"query": query, "session_id": session_id})

        set_request_context(model, session_id)

        initial_state: ResearchState = {
            "query": query,
            "research_plan": None,
            "sub_queries": [],
            "findings": [],
            "synthesis": "",
            "knowledge_gaps": [],
            "iteration_count": 0,
            "should_continue": True,
            "final_answer": None,
            "model": model,
            "session_id": session_id,
            "guardrails": None,
        }

        if config is None:
            config = {"configurable": {"thread_id": f"research_{int(time.time())}"}}

        try:
            # stream_mode="values" yields the full accumulated state after each step
            # (reducers applied), so the final value has iteration_count, sub_queries,
            # and findings correctly populated — unlike the default per-node deltas.
            final_state = initial_state
            for state in self.research_graph.stream(initial_state, config, stream_mode="values"):
                final_state = state

            response_time_ms = int((time.time() - start_time) * 1000)

            all_sources = []
            all_chunk_ids = []
            for finding in final_state.get("findings", []):
                if isinstance(finding.get("sources"), list):
                    all_sources.extend(finding["sources"])
                if isinstance(finding.get("chunk_ids"), list):
                    all_chunk_ids.extend(finding["chunk_ids"])

            # Prefer the canonical list the final answer actually cited as [N], so
            # the UI's Sources list lines up with the inline citation numbers. Fall
            # back to deduping the findings when there's no final synthesis (errors).
            unique_sources = final_state.get("cited_sources")
            if not unique_sources:
                seen: set = set()
                unique_sources = []
                for s in all_sources:
                    if not isinstance(s, dict):
                        continue
                    # RAG sources use "id"; web sources use "source" (URL) as identity
                    key = s.get("id") or s.get("source") or s.get("url")
                    if key and key not in seen:
                        seen.add(key)
                        unique_sources.append(s)
                    elif not key:
                        unique_sources.append(s)

            # (5) The sub_queries list is accumulated across iterations via an
            # operator.add reducer, so it can carry near-identical repeats. Report a
            # de-duplicated, order-preserving list so "Deep explored N" reflects the
            # DISTINCT questions actually investigated.
            _distinct_sub_queries = list(dict.fromkeys(
                q for q in final_state.get("sub_queries", []) if q and q.strip()
            ))

            result = {
                "answer": final_state.get("final_answer") or final_state.get("synthesis", ""),
                "research_plan": final_state.get("research_plan", ""),
                "sub_queries": _distinct_sub_queries,
                "findings": final_state.get("findings", []),
                "synthesis": final_state.get("synthesis", ""),
                "sources": unique_sources,
                "chunk_ids": list(set(all_chunk_ids)),
                "iteration_count": final_state.get("iteration_count", 0),
                "response_time_ms": response_time_ms,
                "guardrails": final_state.get("guardrails"),
            }

            logger.info(
                "Deep research completed",
                extra={"iterations": result["iteration_count"], "response_time_ms": response_time_ms}
            )
            return result

        except Exception as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            logger.error("Error in deep research", extra={"error": str(e)}, exc_info=True)
            return {
                "answer": f"I encountered an error during deep research: {str(e)}",
                "research_plan": "",
                "sub_queries": [],
                "findings": [],
                "synthesis": "",
                "sources": [],
                "chunk_ids": [],
                "iteration_count": 0,
                "response_time_ms": response_time_ms,
                "error": str(e),
            }
