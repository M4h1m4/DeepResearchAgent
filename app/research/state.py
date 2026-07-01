"""
To define the information/state that passes through the research graph
tracking all the information gathered through out the research process
""" 

from typing import TypedDict, List, Dict, Optional, Any
from typing_extensions import Annotated 
import operator 

class ResearchState(TypedDict):
    query: str
    research_plan: Optional[str]

    sub_queries: Annotated[List[str], operator.add]

    findings: Annotated[List[Dict], operator.add]
    synthesis: str

    knowledge_gaps: Annotated[List[str], operator.add]

    iteration_count: int
    should_continue: bool
    final_answer: Optional[str]

    model: Optional[str]
    session_id: Optional[str]

    # Populated by final_synthesis after guardrail checks
    guardrails: Optional[Dict[str, Any]]

    # Canonical, deduplicated, ordered source list the final answer cites as [N].
    # Set by final_synthesis so the inline [N] markers line up with the Sources
    # list shown in the UI.
    cited_sources: Optional[List[Dict]]



# Annotated[List[str], operator.add] allows to merge lists when multiple nodes update the same field 
