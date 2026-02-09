from typing import List, Dict, Optional, Any
from app.research.state import ResearchState
from app.research.tools import research_rag_tool

from config import settings 
from config.logging_config import get_logger 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json 

logger = get_logger(__name__)

def plan_research(state : ResearchState) -> Dict[str, Any]:
    # runs at the satarting to create a research plan with sub-queries 
    logger.info(
        "Planning research",
        extra={
            "query": state["query"],
            "node": "plan_research"
        }
    )

    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.research_planning_temperature,
        openai_api_key=settings.openai_api_key
    )

    planning_prompt = ChatPromptTemplate.from_messages([
        ("system", """ You are a research planning assistant. 
        Analyze the user's query and create a research plan by breaking 
        it down into focused, specific sub-queries 
        Your task: 
        1. Identify the main topics and key aspects of the query
        2. Generate 3-5 specific sub-queries that will help answer the original query.
        3. Prioritize sub-queries by importance. 
        4. Create a brief research strategy
        Return your response as JSON with this structure:
        {{
            "research_plan": "Brief description of research strategy",
            "sub_queries": ["query1", "query2", "query3"]
        }}
        Each sub-query should be:
            - Specific and focused
            - Answerable with document search
            - Complementary to other sub-queries
            - Covering different aspects of the original query"""),
        ("human", "Original query: {query}")
    ])
    try:
        messages = planning_prompt.format_messages(query=state["query"])
        response = llm.invoke(messages)
        response_text = response.content.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        plan_data = json.loads(response_text)

        logger.info(
            "Research plan created",
            extra={
                "query": state["query"],
                "sub_queries_count": len(plan_data.get("sub_queries", [])),
                "research_plan": plan_data.get("research_plan", "")
            }
        )
        
        return {
            "research_plan": plan_data.get("research_plan", ""),
            "sub_queries": plan_data.get("sub_queries", []),
            "iteration_count": 0,
            "should_continue": True
        }

    except json.JSONDecodeError as e:
        logger.error(
            "Failed to parse research plan JSON",
            extra={
                "query": state["query"],
                "error": str(e),
                "response": response_text
            },
            exc_info=True
        )
        # Fallback: create simple plan
        return {
            "research_plan": f"Research strategy for: {state['query']}",
            "sub_queries": [state["query"]],  # Use original query as single sub-query
            "iteration_count": 0,
            "should_continue": True
        }
    except Exception as e:
        logger.error(
            "Error in research planning",
            extra={
                "query": state["query"],
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True
        )
        raise

def generate_sub_query(state: ResearchState) -> Dict[str, Any]:
    #generates sub query to stay focused and investigate on knowledge-gaps 
    logger.info(
        "Generating sub-query",
        extra={
            "query": state["query"],
            "iteration": state["iteration_count"],
            "gaps_count": len(state.get("knowledge_gaps", [])),
            "node": "generate_sub_query"
        }
    )
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.research_planning_temperature,  # Use planning temperature for sub-query generation
        openai_api_key=settings.openai_api_key
    )


    if state.get("knowledge_gaps") and len(state["knowledge_gaps"]) > 0:
        # Generate sub-query from gaps
        gap_text = "\n".join([f"- {gap}" for gap in state["knowledge_gaps"][:3]])  # Top 3 gaps
        prompt_text = f"""Based on these knowledge gaps, generate one focused sub-query to investigate:

Knowledge gaps:
{gap_text}

Original query: {state["query"]}

Generate a specific, answerable sub-query that addresses one of these gaps.
Return only the sub-query text, nothing else."""
    else:
        # Generate from remaining sub-queries in plan
        remaining_queries = state.get("sub_queries", [])
        if remaining_queries:
            # Use next planned sub-query
            next_query = remaining_queries[0]
            return {
                "sub_queries": [next_query]  # This will be processed
            }
        else:
            # Create exploratory sub-query
            prompt_text = f"""Based on the original query and findings so far, generate one focused sub-query to investigate further:

Original query: {state["query"]}
Current synthesis: {state.get("synthesis", "")[:500]}

Generate a specific, answerable sub-query that complements existing findings.
Return only the sub-query text, nothing else."""

    try:
        messages = [("system", "You are a research assistant that generates focused sub-queries."), ("human", prompt_text)]
        response = llm.invoke(messages)
        next_sub_query = response.content.strip().strip('"').strip("'")
        
        logger.info(
            "Sub-query generated",
            extra={
                "sub_query": next_sub_query,
                "iteration": state["iteration_count"]
            }
        )
        
        return {
            "sub_queries": [next_sub_query]
        }
        
    except Exception as e:
        logger.error(
            "Error generating sub-query",
            extra={
                "query": state["query"],
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True
        )
        # Fallback: use original query
        return {
            "sub_queries": [state["query"]]
        }

def execute_rag_tool(state: ResearchState) -> Dict[str, Any]:
    # executes the current sub-query using rag tool 
    if not state.get("sub_queries"):
        logger.warning(
            "No sub-queries to process",
            extra={"query": state["query"], "node": "execute_rag_tool"}
        )
        return {}
    current_sub_query = state["sub_queries"][-1]
    logger.info(
        "Executing RAG tool",
        extra={
            "sub_query": current_sub_query,
            "iteration": state["iteration_count"],
            "node": "execute_rag_tool"
        }
    )
    try:
        result = research_rag_tool.invoke({"query": current_sub_query})
        finding = {
            "query": current_sub_query,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "chunk_ids": result.get("chunk_ids", [])
        }
        
        logger.info(
            "RAG tool executed successfully",
            extra={
                "sub_query": current_sub_query,
                "answer_length": len(finding["answer"]),
                "sources_count": len(finding["sources"]),
                "chunks_count": len(finding["retrieved_chunks"])
            }
        )
        
        return {
            "findings": [finding]
        }
    except Exception as e:
        logger.error(
            "Error executing RAG tool",
            extra={
                "sub_query": current_sub_query,
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True
        )
        # Return empty finding on error
        return {
            "findings": [{
                "query": current_sub_query,
                "answer": f"Error: Could not retrieve information for this query.",
                "sources": [],
                "retrieved_chunks": [],
                "chunk_ids": []
            }]
        }

def synthesize_findings(state: ResearchState) -> Dict[str, Any]:
    #synthesize the findings into one from all rag_tool executions 
    # OPTIMIZATION: Skip synthesis if we have planned queries and only 1 finding
    # This reduces LLM calls - we'll synthesize after multiple findings or at the end
    findings_count = len(state.get("findings", []))
    remaining_queries = len(state.get("sub_queries", []))
    
    logger.info(
        "Synthesizing findings",
        extra={
            "query": state["query"],
            "findings_count": findings_count,
            "iteration": state["iteration_count"],
            "remaining_queries": remaining_queries,
            "node": "synthesize_findings"
        }
    )

    if not state.get("findings"):
        logger.warning(
            "No findings to synthesize",
            extra={"query": state["query"]}
        )
        return {
            "synthesis": "No findings available yet."
        }
    
    # OPTIMIZATION: Skip detailed synthesis if we have more queries to process
    # Just concatenate findings for now, full synthesis happens at the end
    if findings_count == 1 and remaining_queries > 0:
        logger.debug(
            "Skipping detailed synthesis - more queries to process",
            extra={"findings_count": findings_count, "remaining_queries": remaining_queries}
        )
        # Quick concatenation instead of LLM synthesis
        quick_synthesis = f"Finding: {state['findings'][0]['answer'][:500]}..."
        return {
            "synthesis": quick_synthesis
        }
    
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.deep_research_temperature,
        openai_api_key=settings.openai_api_key
    )
    findings_text = ""
    for i, finding in enumerate(state["findings"], 1):
        findings_text += f"\n\nFinding {i} (Query: {finding['query']}):\n"
        findings_text += f"Answer: {finding['answer']}\n"
        if finding.get("sources"):
            # Sources are dictionaries with id, title, source fields
            source_strings = []
            for source in finding['sources']:
                if isinstance(source, dict):
                    source_str = source.get('title', 'Unknown')
                    if source.get('source'):
                        source_str += f" ({source['source']})"
                    source_strings.append(source_str)
                else:
                    source_strings.append(str(source))
            if source_strings:
                findings_text += f"Sources: {', '.join(source_strings)}\n"
    synthesis_prompt = f"""Synthesize the following research findings into a comprehensive,
coherent answer to the original query. Combine information from all findings,
resolve any contradictions, and create a well-structured response.

Original query: {state["query"]}

Research findings:
{findings_text}

Previous synthesis (if any):
{state.get("synthesis", "None")}

Create a comprehensive synthesis that:
1. Directly addresses the original query
2. Incorporates information from all findings
3. Maintains source citations
4. Is well-structured and coherent
5. Identifies any remaining gaps or unanswered questions

Provide the synthesis now:"""
    try: 
        messages = [("system", "You are a research synthesis assistant that combines findings into comprehensive answers."), ("human", synthesis_prompt)]
        response = llm.invoke(messages)
        new_synthesis = response.content.strip()
        
        logger.info(
            "Findings synthesized",
            extra={
                "query": state["query"],
                "synthesis_length": len(new_synthesis),
                "iteration": state["iteration_count"]
            }
        )
        
        return {
            "synthesis": new_synthesis
        }
        
    except Exception as e:
        logger.error(
            "Error synthesizing findings",
            extra={
                "query": state["query"],
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True
        )
        # Fallback: concatenate findings
        fallback_synthesis = "\n\n".join([
            f"Query: {f['query']}\nAnswer: {f['answer']}"
            for f in state["findings"]
        ])
        return {
            "synthesis": fallback_synthesis
        }


def analyze_gaps(state: ResearchState) -> Dict[str, Any]:
    #The decision node to decide whether the research should continue based on the findings and the original query
    logger.info(
        "Analyzing knowledge gaps",
        extra={
            "query": state["query"],
            "iteration": state["iteration_count"],
            "node": "analyze_gaps"
        }
    )

    max_iterations = settings.max_research_iterations  # Use settings value (default 3)
    if state["iteration_count"] >= max_iterations:
        logger.info(
            "Maximum iterations reached",
            extra={
                "query": state["query"],
                "iteration_count": state["iteration_count"],
                "max_iterations": max_iterations
            }
        )
        return {
            "should_continue": False,
            "knowledge_gaps": []
        } 
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.gap_analysis_temperature,
        openai_api_key=settings.openai_api_key
    )
    gap_analysis_prompt = f"""Analyze if the current research synthesis fully addresses the original query.

    Original query: {state["query"]}

    Current synthesis:
    {state.get("synthesis", "No synthesis available yet.")}

    Task:
        1. Identify any knowledge gaps or unanswered aspects of the original query
        2. Determine if the synthesis is comprehensive enough
        3. Decide if further research is needed

        Return your response as JSON:
        {{
        "gaps": ["gap1", "gap2", ...],  // List of identified gaps (empty if none)
        "is_complete": true/false,      // Whether research is complete
        "reasoning": "Brief explanation"
        }}
        If research is complete, return empty gaps array and is_complete=true.
        If gaps exist, list them specifically and set is_complete=false."""

    try:
        messages = [("system", "You are a research gap analysis assistant."), ("human", gap_analysis_prompt)]
        response = llm.invoke(messages)
        response_text = response.content.strip()
        
        # Parse JSON (handle markdown if present)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        analysis = json.loads(response_text)
        
        gaps = analysis.get("gaps", [])
        is_complete = analysis.get("is_complete", False)
        reasoning = analysis.get("reasoning", "")
        
        logger.info(
            "Gap analysis completed",
            extra={
                "query": state["query"],
                "gaps_count": len(gaps),
                "is_complete": is_complete,
                "reasoning": reasoning,
                "iteration": state["iteration_count"]
            }
        )
        
        return {
            "knowledge_gaps": gaps,
            "should_continue": not is_complete and len(gaps) > 0,
            "iteration_count": state["iteration_count"] + 1
        }

    except (json.JSONDecodeError, KeyError) as e:
        logger.error(
            "Error parsing gap analysis",
            extra={
                "query": state["query"],
                "error": str(e),
                "response": response_text
            },
            exc_info=True
        )
        # Fallback: conservative approach - continue if early iterations
        should_continue = state["iteration_count"] < 3
        return {
            "knowledge_gaps": [],
            "should_continue": should_continue,
            "iteration_count": state["iteration_count"] + 1
        }
    except Exception as e:
        logger.error(
            "Error in gap analysis",
            extra={
                "query": state["query"],
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True
        )
        # On error, stop research to prevent infinite loops
        return {
            "knowledge_gaps": [],
            "should_continue": False,
            "iteration_count": state["iteration_count"] + 1
        }

def final_synthesis(state: ResearchState) -> Dict[str, Any]:
    #creates a comprehensive answer from the entire research 
    logger.info(
        "Creating final synthesis",
        extra={
            "query": state["query"],
            "findings_count": len(state.get("findings", [])),
            "iterations": state["iteration_count"],
            "node": "final_synthesis"
        }
    )
    
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.deep_research_temperature,
        openai_api_key=settings.openai_api_key
    )
    
    # Collect all sources from findings
    all_sources = []
    for finding in state.get("findings", []):
        all_sources.extend(finding.get("sources", []))
    unique_sources = list(set(all_sources))
    
    final_prompt = f"""Create a comprehensive, well-structured final answer to the user's query
    based on all research findings. The answer should be polished, coherent, and thoroughly
    address all aspects of the original query.

    Original query: {state["query"]}

    Research synthesis:
    {state.get("synthesis", "No synthesis available.")}

    Research findings summary:
        - Total research iterations: {state["iteration_count"]}
        - Number of sub-queries investigated: {len(state.get("sub_queries", []))}
        - Sources consulted: {len(unique_sources)}

    Create a final comprehensive answer that:
        1. Directly and thoroughly addresses the original query
        2. Incorporates insights from all research findings
        3. Is well-organized with clear structure
        4. Includes source citations where appropriate
        5. Acknowledges any limitations or remaining uncertainties

        Provide the final answer now:"""
    
    try:
        messages = [("system", "You are a research synthesis assistant that creates comprehensive final answers."), ("human", final_prompt)]
        response = llm.invoke(messages)
        final_answer = response.content.strip()
        
        logger.info(
            "Final synthesis created",
            extra={
                "query": state["query"],
                "final_answer_length": len(final_answer),
                "sources_count": len(unique_sources),
                "iterations": state["iteration_count"]
            }
        )
        
        return {
            "final_answer": final_answer
        }
        
    except Exception as e:
        logger.error(
            "Error creating final synthesis",
            extra={
                "query": state["query"],
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True
        )
        # Fallback: use current synthesis
        return {
            "final_answer": state.get("synthesis", "Unable to generate final answer.")
        }
    