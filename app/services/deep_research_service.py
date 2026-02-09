import time 
from typing import Dict, Any, Optional 
from app.research.graph import get_research_graph
from app.research.state import ResearchState
from config.logging_config import get_logger 

logger = get_logger(__name__)

class DeepResearchService:
    # Service for deep research mode using LangGraph Orchestration 
    def __init__(self):
        self.research_graph = get_research_graph()
        logger.info("DeepResearch Service Initialized")
    
    def research(
        self, 
        query: str, 
        config: Optional[Dict[str, Any]] = None  #Optional LangGraph configuration ex: thread_id, etc
    ) -> Dict[str, Any]:
        start_time = time.time()
        logger.info(
            "Starting deep research", 
            extra={
                "query": query, 
                "service": "DeepResearchService"
            }
        )
        initial_state: ResearchState = {
            "query": query, 
            "research_plan": None, 
            "sub_queries": [], 
            "findings": [], 
            "synthesis": "", 
            "knowledge_gaps": [], 
            "iteration_count": 0, 
            "should_continue": True,  # Start with True to allow research to begin
            "final_answer": None
        }

        logger.info(
            "Research State Initialized", 
            extra={
                "query":initial_state["query"], 
                "iteration_count": initial_state["iteration_count"]
            }
        )
        if config is None:
            config = {
                "configurable":{
                    "thread_id": f"research_{int(time.time())}"
                }
            }
        
        try:
            final_state = None 
            #Invoke graph until completion
            #LangGraph handles the flow automatically
            for state in self.research_graph.stream(initial_state, config):
                #Log each node completion
                for node_name, node_state in state.items():
                    logger.debug(
                        "Graph node completed",
                        extra={
                            "node": node_name,
                            "query": query,
                            "iteration": node_state.get("iteration_count", 0)
                        }
                    )
                    final_state = node_state

            #Extracting the final state   
            if final_state is None:
                # Fallback: get from graph's final invocation
                final_state = initial_state
                for state in self.research_graph.stream(initial_state, config):
                    for node_state in state.values():
                        final_state = node_state
            response_time_ms = int((time.time() - start_time) * 1000)

            # Collect all sources and chunk_ids from findings
            all_sources = []
            all_chunk_ids = []
            for finding in final_state.get("findings", []):
                # Sources are already SourceInfo-like dicts from RAG service
                if isinstance(finding.get("sources"), list):
                    all_sources.extend(finding.get("sources", []))
                # Collect chunk_ids
                if isinstance(finding.get("chunk_ids"), list):
                    all_chunk_ids.extend(finding.get("chunk_ids", []))
            
            # Remove duplicate sources by id
            seen_source_ids = set()
            unique_sources = []
            for source in all_sources:
                if isinstance(source, dict) and source.get("id") not in seen_source_ids:
                    seen_source_ids.add(source.get("id"))
                    unique_sources.append(source)
            
            result = {
                "answer": final_state.get("final_answer") or final_state.get("synthesis", ""),
                "research_plan": final_state.get("research_plan", ""),
                "sub_queries": final_state.get("sub_queries", []),
                "findings": final_state.get("findings", []),
                "synthesis": final_state.get("synthesis", ""),
                "sources": unique_sources,
                "chunk_ids": list(set(all_chunk_ids)),  # Remove duplicates
                "iteration_count": final_state.get("iteration_count", 0),
                "response_time_ms": response_time_ms
            }
            
            logger.info(
                "Deep research completed",
                extra={
                    "query": query,
                    "answer_length": len(result["answer"]),
                    "iterations": result["iteration_count"],
                    "sources_count": len(result["sources"]),
                    "response_time_ms": response_time_ms
                }
            )

            return result 
        
        except Exception as e:
            
            response_time_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "Error in deep research",
                extra={
                    "query": query,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "response_time_ms": response_time_ms
                },
                exc_info=True
            )
            
            # Return error response
            return {
                "answer": f"I encountered an error during deep research: {str(e)}",
                "research_plan": "",
                "sub_queries": [],
                "findings": [],
                "synthesis": "",
                "sources": [],
                "iteration_count": 0,
                "response_time_ms": response_time_ms,
                "error": str(e)
            }



