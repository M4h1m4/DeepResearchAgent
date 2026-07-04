import re
from typing import List, Dict, Optional, Any
from datetime import datetime
from app.research.state import ResearchState
from app.research.tools import research_rag_tool, web_search_tool

from config import settings
from config.logging_config import get_logger
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from app.services.guardrails import PIIDetector, HallucinationGuard
import json

logger = get_logger(__name__)

_pii_detector = PIIDetector()
_hallucination_guard = HallucinationGuard()


def _today_context() -> str:
    """Grounds the LLM in the real current date so 'latest'/'recent' means now,
    not the model's training-cutoff year (which defaults answers to ~2023)."""
    now = datetime.now()
    return (
        f"IMPORTANT — Today's date is {now.strftime('%B %d, %Y')}; the current year is {now.year}. "
        f"When the user asks for 'latest', 'recent', 'current', 'new', or similar, interpret it as "
        f"{now.year} and prioritise the most up-to-date information available. "
        f"Never assume the year is 2023 or any earlier year, and do not restrict searches to past years "
        f"unless the user explicitly names one."
    )

def _llm_for_state(state: ResearchState, temperature: float) -> "ChatOpenAI":
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required. Add it to your .env file.")
    model = state.get("model") or settings.openai_model
    return ChatOpenAI(model=model, temperature=temperature, openai_api_key=settings.openai_api_key, request_timeout=60)


# ---------------------------------------------------------------------------
# Hybrid web search decision
# ---------------------------------------------------------------------------

_TEMPORAL_WORDS = {
    "latest", "recent", "recently", "current", "currently", "today", "now",
    "this year", "this month", "this week", "breaking", "just announced",
    "2024", "2025", "2026",
}

# Whole-word matching only — substring matching wrongly fired on words that
# merely contain a keyword (e.g. "now" inside "known").
_TEMPORAL_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in _TEMPORAL_WORDS) + r")\b"
)

_WEB_CONFIRM_PROMPT = """\
You are deciding whether to search the live web for a research sub-query.

Sub-query: {query}
Knowledge base result quality: {rag_quality}
Research iteration: {iteration}

Should we search the live web to get better or more current information?
Answer with YES or NO only — nothing else."""


def _should_use_web_search(query: str, rag_result: dict, state: ResearchState, llm: "ChatOpenAI") -> bool:
    """Gate 1 (rules) then Gate 2 (LLM yes/no) to decide if web search should run."""
    q_lower = query.lower()

    # Gate 1 — cheap rule checks
    has_temporal = bool(_TEMPORAL_PATTERN.search(q_lower))
    rag_empty = not rag_result.get("sources")
    rag_weak = "couldn't find" in (rag_result.get("answer") or "").lower() or \
               "no relevant" in (rag_result.get("answer") or "").lower()
    gaps_remain = (
        state.get("iteration_count", 0) >= 2
        and bool(state.get("knowledge_gaps"))
    )

    if not (has_temporal or rag_empty or rag_weak or gaps_remain):
        logger.debug("Web search gate 1: no trigger — skipping web search", extra={"query": query[:80]})
        return False

    # Temporal intent is unconditional — the user explicitly wants current information.
    # Skip Gate 2: even if the KB has relevant content it may be outdated.
    if has_temporal:
        logger.info("Web search gate: temporal intent — web search mandatory", extra={"query": query[:80]})
        return True

    logger.info(
        "Web search gate 1 triggered — asking LLM to confirm",
        extra={"query": query[:80], "rag_empty": rag_empty, "gaps_remain": gaps_remain},
    )

    # Gate 2 — LLM yes/no confirmation (only for non-temporal triggers)
    if rag_empty:
        rag_quality = "empty — no results returned"
    elif rag_weak:
        rag_quality = "weak — answer indicates no relevant information found"
    else:
        rag_quality = "partial — some results exist but may be outdated or incomplete"

    try:
        from langchain_core.messages import HumanMessage
        prompt = _WEB_CONFIRM_PROMPT.format(
            query=query, rag_quality=rag_quality, iteration=state.get("iteration_count", 0)
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        decision = response.content.strip().lower().startswith("yes")
        logger.info("Web search gate 2 LLM decision", extra={"query": query[:80], "decision": decision})
        return decision
    except Exception as e:
        logger.warning("Web search gate 2 LLM call failed — defaulting to rule result", extra={"error": str(e)})
        return rag_empty or rag_weak


def plan_research(state: ResearchState) -> Dict[str, Any]:
    logger.info("Planning research", extra={"query": state["query"], "node": "plan_research"})

    llm = _llm_for_state(state, settings.research_planning_temperature)

    planning_prompt = ChatPromptTemplate.from_messages([
        ("system", _today_context()),
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
    """Decide which sub-queries execute_rag_tool should research next.

    First pass: the planned sub-queries already exist (from plan_research), so we
    pass through and let execute_rag_tool fan them all out in parallel.
    Gap passes: turn the open knowledge gaps into a fresh batch of sub-queries
    (also researched in parallel).
    """
    findings = state.get("findings", [])
    planned = state.get("sub_queries", [])
    gaps = state.get("knowledge_gaps", [])

    logger.info(
        "Generating sub-queries",
        extra={
            "query": state["query"],
            "iteration": state["iteration_count"],
            "gaps_count": len(gaps),
            "planned_count": len(planned),
            "node": "generate_sub_query",
        },
    )

    # First pass — planned sub-queries already present; nothing to generate.
    if not findings and planned:
        return {}

    # Gap-driven pass — convert open gaps into a batch of new sub-queries.
    if gaps:
        llm = _llm_for_state(state, settings.research_planning_temperature)
        gap_text = "\n".join(f"- {gap}" for gap in gaps[:3])
        prompt_text = f"""Based on these knowledge gaps, generate up to 3 focused, specific sub-queries to investigate them.

Knowledge gaps:
{gap_text}

Original query: {state["query"]}

Return ONLY a JSON array of sub-query strings, e.g. ["sub-query 1", "sub-query 2"]. Nothing else."""
        try:
            messages = [
                ("system", f"You are a research assistant that generates focused sub-queries. {_today_context()}"),
                ("human", prompt_text),
            ]
            response = llm.invoke(messages)
            text = response.content.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            new_queries = json.loads(text)
            new_queries = [q.strip() for q in new_queries if isinstance(q, str) and q.strip()][:3]
            if new_queries:
                logger.info("Gap sub-queries generated", extra={"count": len(new_queries)})
                return {"sub_queries": new_queries}
        except Exception as e:
            logger.warning("Gap sub-query generation failed — using original query", extra={"error": str(e)})

    # Fallback: nothing planned and no usable gaps — research the original query.
    if not planned:
        return {"sub_queries": [state["query"]]}
    return {}

def _research_one_sub_query(sub_query: str, state: ResearchState, llm: "ChatOpenAI") -> Dict[str, Any]:
    """Retrieve (RAG + optional web) for a single sub-query. Runs inside a worker
    thread — kept self-contained so several can execute in parallel."""
    rag_answer = ""
    rag_sources: list = []
    rag_chunks: list = []
    rag_result: dict = {}

    try:
        rag_result = research_rag_tool.invoke({"query": sub_query})
        rag_answer = rag_result.get("answer", "")
        rag_sources = rag_result.get("sources", [])
        rag_chunks = rag_result.get("retrieved_chunks", [])
    except Exception as e:
        logger.error("RAG tool error", extra={"sub_query": sub_query[:80], "error": str(e)}, exc_info=True)

    web_answer = ""
    web_sources: list = []
    if _should_use_web_search(sub_query, rag_result, state, llm):
        try:
            web_result = web_search_tool.invoke({"query": sub_query})
            web_answer = web_result.get("answer", "")
            web_sources = web_result.get("sources", [])
        except Exception as e:
            logger.warning("Web search failed", extra={"sub_query": sub_query[:80], "error": str(e)})

    combined_answer = "\n\n".join(filter(None, [rag_answer, web_answer])) or (
        "No information found for this query."
    )
    return {
        "query": sub_query,
        "answer": combined_answer,
        "sources": rag_sources + web_sources,
        "retrieved_chunks": rag_chunks,
        "chunk_ids": rag_chunks,
    }


def execute_rag_tool(state: ResearchState) -> Dict[str, Any]:
    """Process every not-yet-researched sub-query IN PARALLEL.

    Each sub-query is independent, so instead of retrieving one-at-a-time (the old
    ~15s × N behaviour) we fan them out across worker threads and collect the
    findings together — the retrieval phase now takes ≈ the slowest single query.
    """
    findings = state.get("findings", [])
    done_queries = {f.get("query") for f in findings}
    # Preserve order, drop duplicates, skip anything already researched
    pending = list(dict.fromkeys(
        q for q in state.get("sub_queries", []) if q and q not in done_queries
    ))

    if not pending:
        logger.warning("No pending sub-queries to process", extra={"query": state["query"], "node": "execute_rag_tool"})
        return {}

    logger.info(
        "Executing retrieval in parallel",
        extra={"pending_count": len(pending), "iteration": state["iteration_count"], "node": "execute_rag_tool"},
    )

    llm = _llm_for_state(state, settings.research_planning_temperature)

    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    new_findings: List[Dict] = []
    with ThreadPoolExecutor(max_workers=min(len(pending), 6)) as executor:
        futures = []
        for sub_query in pending:
            # copy_context() snapshots the current contextvars (model + session_id set
            # by set_request_context) so the RAG tool sees them inside the worker thread.
            ctx = contextvars.copy_context()
            futures.append(executor.submit(ctx.run, _research_one_sub_query, sub_query, state, llm))
        for fut in futures:
            try:
                new_findings.append(fut.result())
            except Exception as e:
                logger.error("Sub-query retrieval failed", extra={"error": str(e)}, exc_info=True)

    logger.info(
        "Parallel retrieval complete",
        extra={"findings_added": len(new_findings), "total_sources": sum(len(f["sources"]) for f in new_findings)},
    )
    return {"findings": new_findings}

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
    
    llm = _llm_for_state(state, settings.deep_research_temperature)
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
        messages = [
            ("system", f"You are a research synthesis assistant that combines findings into comprehensive answers. {_today_context()}"),
            ("human", synthesis_prompt),
        ]
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
    llm = _llm_for_state(state, settings.gap_analysis_temperature)
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
    
    llm = _llm_for_state(state, settings.deep_research_temperature)

    # Collect all sources from findings (each source is a dict: unhashable)
    # Deduplicate by (id, title, source) so we can use a set of keys
    all_sources = []
    for finding in state.get("findings", []):
        all_sources.extend(finding.get("sources", []))
    seen_keys = set()
    unique_sources = []
    for s in all_sources:
        if isinstance(s, dict):
            key = (s.get("id"), s.get("title"), s.get("source"))
        else:
            key = (getattr(s, "id", None), getattr(s, "title", ""), getattr(s, "source", ""))
        if key not in seen_keys:
            seen_keys.add(key)
            unique_sources.append(s)
    
    # Number the deduplicated sources so the answer can cite them inline as [N],
    # mapping to the Sources list shown in the UI. The research synthesis may refer
    # to "Finding N" internally — those are not citations and must not leak into the
    # answer, so we give the model this explicit numbered list to cite instead.
    def _source_label(s) -> str:
        if isinstance(s, dict):
            title, loc = s.get("title", "Untitled"), s.get("source", "")
        else:
            title, loc = getattr(s, "title", "Untitled"), getattr(s, "source", "")
        return f"{title} ({loc})" if loc else title

    numbered_sources = "\n".join(f"[{i}] {_source_label(s)}" for i, s in enumerate(unique_sources, 1))

    final_prompt = f"""Create a comprehensive, well-structured final answer to the user's query
    based on all research findings. The answer should be polished, coherent, and thoroughly
    address all aspects of the original query.

    Original query: {state["query"]}

    Research synthesis:
    {state.get("synthesis", "No synthesis available.")}

    Sources (cite these by number):
    {numbered_sources or "No sources available."}

    Create a final comprehensive answer that:
        1. Directly and thoroughly addresses the original query
        2. Incorporates insights from all research findings
        3. Is well-organized with clear structure
        4. Cites sources inline using [N] markers that refer to the numbered Sources
           list above (e.g. "adversarial training improves robustness [1][3]"). Do NOT
           write "Finding 1" / "Finding 2" — those are internal labels, not citations.
        5. Acknowledges any limitations or remaining uncertainties

        Provide the final answer now:"""
    
    try:
        messages = [
            ("system", f"You are a research synthesis assistant that creates comprehensive final answers. {_today_context()}"),
            ("human", final_prompt),
        ]
        response = llm.invoke(messages)
        final_answer = response.content.strip()

        logger.info(
            "Final synthesis created",
            extra={
                "query": state["query"],
                "final_answer_length": len(final_answer),
                "sources_count": len(unique_sources),
                "iterations": state["iteration_count"],
            },
        )

        # Guardrail 1: PII — redact before returning
        pii_result = _pii_detector.detect_and_redact(final_answer)
        if pii_result.has_pii:
            final_answer = pii_result.redacted_text

        # Guardrail 2: Hallucination — grade the answer against the SAME material it
        # was built from. The final answer is generated from the synthesis, so check
        # against the synthesis first (a coherent, grounded distillation of every
        # finding), then the raw findings as supporting evidence. Putting the
        # synthesis first keeps it from being dropped by the judge's context-length
        # cap. Previously we graded only against the concatenated findings, which
        # both mismatched the generation source and got truncated on long runs —
        # producing a false 0.0 score even when every intermediate check was 1.0.
        synthesis_text = state.get("synthesis", "") or ""
        findings_text = "\n\n".join(
            f.get("answer", "") for f in state.get("findings", []) if f.get("answer")
        )
        all_context = "\n\n".join(part for part in (synthesis_text, findings_text) if part.strip())
        hal_result = _hallucination_guard.check(
            question=state["query"],
            answer=final_answer,
            context=all_context,
        )
        if not hal_result.is_grounded:
            final_answer = (
                f"⚠️ Note: Parts of this answer may not be fully supported by the retrieved sources "
                f"(faithfulness score: {hal_result.faithfulness_score:.2f}).\n\n{final_answer}"
            )

        return {
            "final_answer": final_answer,
            "cited_sources": unique_sources,
            "guardrails": {
                "pii_detected": pii_result.has_pii,
                "pii_entity_types": [e["type"] for e in pii_result.entities],
                "faithfulness_score": hal_result.faithfulness_score,
                "is_grounded": hal_result.is_grounded,
                "unsupported_claims": hal_result.unsupported_claims,
            },
        }

    except Exception as e:
        logger.error(
            "Error creating final synthesis",
            extra={
                "query": state["query"],
                "error": str(e),
                "error_type": type(e).__name__
            },
            exc_info=True,
        )
        return {
            "final_answer": state.get("synthesis", "Unable to generate final answer.")
        }
    