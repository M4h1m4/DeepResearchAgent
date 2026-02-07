from typing import Literal 
from enum import Enum
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver 

from app.research.state import ResearchState 
from app.research.nodes import (
    plan_research,
    generate_sub_query,
    execute_rag_tool,
    analyze_gaps,
    synthesize_findings,
    final_synthesis
)

from config.logging_config import get_logger 

logger = get_logger(__name__)

class ResearchDecision(Enum):
    CONTINUE_RESEARCH = "continue_research"
    FINALIZE = "finalize"

def should_continue_research(state: ResearchState) -> ResearchDecision:
    should_continue = state.get("should_continue", False)
    logger.debug(
        "Research continuation decision",
        extra={
            "query": state["query"],
            "should_continue": should_continue,
            "iteration": state.get("iteration_count", 0),
            "decision": ResearchDecision.CONTINUE_RESEARCH.value if should_continue else ResearchDecision.FINALIZE.value
        }
    )
    if should_continue:
        return ResearchDecision.CONTINUE_RESEARCH
    else:
        return ResearchDecision.FINALIZE

    
def create_research_graph() -> StateGraph:
    logger.info("Creating research graph")

    workflow = StateGraph(ResearchState) #creates a state Graph 

    # the order matters for definition and not execution since the edges are added explicitly
    workflow.add_node("plan_research", plan_research)
    workflow.add_node("generate_sub_query", generate_sub_query)
    workflow.add_node("execute_rag_tool", execute_rag_tool)
    workflow.add_node("synthesize_findings", synthesize_findings)
    workflow.add_node("analyze_gaps", analyze_gaps)
    workflow.add_node("final_synthesis", final_synthesis)

    workflow.set_entry_point("plan_research")
    workflow.add_edge("plan_research", "generate_sub_query")
    workflow.add_edge("generate_sub_query", "execute_rag_tool")
    workflow.add_edge("execute_rag_tool", "synthesize_findings")
    workflow.add_edge("synthesize_findings", "analyze_gaps")

    workflow.add_conditional_edges(
        "analyze_gaps",
        should_continue_research,
        {
            ResearchDecision.CONTINUE_RESEARCH: "generate_sub_query", 
            ResearchDecision.FINALIZE: "final_synthesis"
        }
    )

    workflow.add_edge("final_synthesis", "__end__")

    memory = MemorySaver() #In memory, use FileSaver or database-backed checkpointing for production
    app = workflow.compile(checkpointer=memory) #compile graph with checkpointing for state persistence 

    logger.info("Research Graph created and compiled")
    return app 

_research_graph = None

def get_research_graph() -> StateGraph:
    #Get or create research graph instance.
    global _research_graph
    if _research_graph is None:
        _research_graph = create_research_graph()
    return _research_graph
