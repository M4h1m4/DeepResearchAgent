"""
To define the information/state that passes through the research graph
tracking all the information gathered through out the research process
""" 

from typing import TypedDict, List, Dict, Optional
from typing_extensions import Annotated 
import operator 

class ResearchState(TypedDict):
    query: str #original user query
    research_plan: Optional[str]

    sub_queries : Annotated[List[str], operator.add]

    findings: Annotated[List[str], operator.add] #contains queries, answers, chunks, sources in each step
    synthesis: str # current summary of all findings 

    knowledge_gaps: Annotated[List[str], operator.add]

    iteration_count: int 
    should_continue: bool 
    final_answer: Optional[str]



# Annotated[List[str], operator.add] allows to merge lists when multiple nodes update the same field 
