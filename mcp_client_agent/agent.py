import argparse
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# MCP server serves Streamable HTTP at /mcp by default
MCP_HTTP_URL = os.environ.get("MCP_HTTP_URL", "http://127.0.0.1:8001/mcp")
# FastAPI backend URL (for --use-rest-deep to avoid MCP client TaskGroup issues)
FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000")

# Deep research can run 2+ minutes; client must wait (default timeouts are too short)
DEEP_RESEARCH_TIMEOUT_SECONDS = 600  # 10 minutes

def parse_tool_result(result):
    #To parse the result from the tool call and structure them into JSON
    if result.is_error:
        return {
            "error": True, 
            "message": result.content[0].text if result.content else "Unknown Error",
        }
    if not result.content:
        return {"answer": "", "sources": [], "response_time_ms": 0}
    text = result.content[0].text 
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"answer": text, "sources": []}


@asynccontextmanager
async def connect_mcp_http(url: str, timeout_seconds: float = DEEP_RESEARCH_TIMEOUT_SECONDS):
    """
    Connect to MCP server via Streamable HTTP. Yields (session, connect_time_ms).
    Uses long timeouts so deep_research (2+ min) does not trigger client TaskGroup errors.
    """
    t0 = time.perf_counter()
    async with streamablehttp_client(
        url,
        timeout=timeout_seconds,
        sse_read_timeout=timeout_seconds,
    ) as (read_stream, write_stream, _): 
    #The above line opens the transport to MCP Server over streamable HTTP.
        async with ClientSession(read_stream, write_stream) as session:
        # The above line wraps the above streams into a client session. 
            await session.initialize()
            connect_time_ms = int((time.perf_counter()-t0) * 1000)
            yield session, connect_time_ms 

async def deep_research_via_rest(query: str, base_url: str, timeout_seconds: float) -> tuple[dict, int]:
    """Call FastAPI POST /api/v1/query/deep. Returns (data, tool_ms). Avoids MCP client TaskGroup issues."""
    import httpx
    url = f"{base_url.rstrip('/')}/api/v1/query/deep"
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            r = await client.post(url, json={"query": query})
            r.raise_for_status()
            body = r.json()
    except Exception as e:
        tool_ms = int((time.perf_counter() - t0) * 1000)
        return {"error": True, "message": str(e), "answer": "", "sources": []}, tool_ms
    tool_ms = int((time.perf_counter() - t0) * 1000)
    sources = body.get("sources", [])
    if sources and isinstance(sources[0], dict):
        pass
    else:
        sources = [{"id": getattr(s, "id", None), "title": getattr(s, "title", ""), "source": getattr(s, "source", "")} for s in sources]
    return {
        "answer": body.get("answer", ""),
        "sources": sources,
        "response_time_ms": body.get("response_time_ms", tool_ms),
    }, tool_ms


async def call_tool_with_timing(
    session, tool_name: str, arguments: dict, result_holder: list | None = None
):
    """
    Call an MCP tool and return (parsed_result, tool_response_time_ms).
    If result_holder is provided, store (data, tool_ms) there as soon as we have them
    so we can recover from TaskGroup raised after the result is ready.
    """
    t0 = time.perf_counter()
    read_timeout = timedelta(seconds=DEEP_RESEARCH_TIMEOUT_SECONDS) if tool_name == "deep_research" else None
    result = await session.call_tool(tool_name, arguments, read_timeout_seconds=read_timeout)
    tool_ms = int((time.perf_counter() - t0) * 1000)
    data = parse_tool_result(result)
    if result_holder is not None:
        result_holder[:] = [(data, tool_ms)]
    return data, tool_ms


async def run(
    query: str,
    mode: str = "fast",
    url: str | None = None,
    verbose: bool = True,
    use_rest_deep: bool = False,
):
    total_t0 = time.perf_counter()
    connect_ms = 0
    tool_ms = 0
    data = {}

    # Deep research via REST avoids MCP client TaskGroup errors on long runs
    if mode == "deep" and use_rest_deep:
        fastapi_url = os.environ.get("FASTAPI_URL", FASTAPI_URL)
        if verbose:
            print(f"Using REST for deep research: POST {fastapi_url}/api/v1/query/deep")
        data, tool_ms = await deep_research_via_rest(
            query, fastapi_url, DEEP_RESEARCH_TIMEOUT_SECONDS
        )
        total_ms = int((time.perf_counter() - total_t0) * 1000)
        out = {
            "answer": data.get("answer", ""),
            "sources": data.get("sources", []),
            "connect_time_ms": 0,
            "tool_response_time_ms": tool_ms,
            "total_time_ms": total_ms,
        }
        if data.get("error"):
            out["error"] = True
            out["message"] = data.get("message", "")
        if verbose:
            print("-" * 60)
            print("Response time (REST deep):", tool_ms, "ms")
            print("Response time (total):", total_ms, "ms")
            print("-" * 60)
            if out.get("error"):
                print("Error:", out.get("message", ""))
            else:
                answer = out.get("answer", "")
                preview = (answer[:400] + "...") if len(answer) > 400 else answer
                print("Answer preview:", preview)
                print("Sources count:", len(out.get("sources", [])))
        return out

    base_url = url or MCP_HTTP_URL
    result_holder = []  # mutable: if SDK raises after result is ready, we may have (data, tool_ms) here
    try:
        async with connect_mcp_http(base_url) as (session, conn_ms):
            connect_ms = conn_ms
            if verbose:
                tools_result = await session.list_tools()
                tool_names = [t.name for t in tools_result.tools]
                print(f"Connected to MCP at {base_url} (connect: {connect_ms} ms). Tools: {tool_names}")

            if mode == "deep":
                data, tool_ms = await call_tool_with_timing(
                    session, "deep_research", {"query": query}, result_holder=result_holder
                )
            else:
                data, tool_ms = await call_tool_with_timing(
                    session, "rag_query", {"query": query}, result_holder=result_holder
                )
    except Exception as e:
        total_ms = int((time.perf_counter() - total_t0) * 1000)
        # Recover result if SDK raised TaskGroup after we had stored it in result_holder
        if result_holder and len(result_holder) > 0:
            data, tool_ms = result_holder[0]
        # Server may have completed; client can raise TaskGroup/ExceptionGroup during or after response.
        # If we have a good result (from return or result_holder), treat as success.
        is_taskgroup_error = (
            "TaskGroup" in str(e)
            or "sub-exception" in str(e)
            or type(e).__name__ == "ExceptionGroup"
        )
        if is_taskgroup_error and data and not data.get("error") and data.get("answer"):
            if verbose:
                print("(MCP client teardown warning; server response was received.)")
            out = {
                "answer": data.get("answer", ""),
                "sources": data.get("sources", []),
                "connect_time_ms": connect_ms,
                "tool_response_time_ms": tool_ms,
                "total_time_ms": total_ms,
            }
            if verbose:
                print("-" * 60)
                print("Response time (MCP tool call):", tool_ms, "ms")
                print("Response time (connect + init):", connect_ms, "ms")
                print("Response time (total, this process):", total_ms, "ms")
                print("-" * 60)
                answer = out.get("answer", "")
                preview = (answer[:400] + "...") if len(answer) > 400 else answer
                print("Answer preview:", preview)
                print("Sources count:", len(out.get("sources", [])))
            return out
        if verbose:
            print(f"Error: {e}")
            print(f"Total time (failed): {total_ms} ms")
            # Diagnose root cause (MCP client often wraps the real exception)
            cause = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
            if cause is not None:
                print(f"Root cause: {type(cause).__name__}: {cause}")
            if getattr(e, "exceptions", None):
                for i, sub in enumerate(e.exceptions):
                    print(f"Sub-exception[{i}]: {type(sub).__name__}: {sub}")
        return {
            "error": True,
            "message": str(e),
            "connect_time_ms": connect_ms,
            "tool_response_time_ms": tool_ms,
            "total_time_ms": total_ms,
        }

    total_ms = int((time.perf_counter() - total_t0) * 1000)
    out = {
        "answer": data.get("answer", ""),
        "sources": data.get("sources", []),
        "connect_time_ms": connect_ms,
        "tool_response_time_ms": tool_ms,
        "total_time_ms": total_ms,
    }
    if data.get("error"):
        out["error"] = True
        out["message"] = data.get("message", "")

    if verbose:
        print("-" * 60)
        print("Response time (MCP tool call):", tool_ms, "ms")
        print("Response time (connect + init):", connect_ms, "ms")
        print("Response time (total, this process):", total_ms, "ms")
        print("-" * 60)
        if out.get("error"):
            print("Error:", out.get("message", ""))
        else:
            answer = out.get("answer", "")
            preview = (answer[:400] + "...") if len(answer) > 400 else answer
            print("Answer preview:", preview)
            print("Sources count:", len(out.get("sources", [])))

    return out

def main():
    parser = argparse.ArgumentParser(
        description="Local agent: connect to Deep Research MCP over HTTP and measure response time."
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="",
        help="Query to send (default: prompt for input).",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["fast", "deep"],
        default="fast",
        help="Use rag_query (fast) or deep_research (deep). Default: fast.",
    )
    parser.add_argument(
        "--url",
        "-u",
        type=str,
        default=None,
        help=f"MCP server URL (default: env MCP_HTTP_URL or {MCP_HTTP_URL}).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print JSON result (no human-readable timing/output).",
    )
    parser.add_argument(
        "--use-rest-deep",
        action="store_true",
        help="Use FastAPI POST /query/deep for deep research (avoids MCP client TaskGroup errors).",
    )
    args = parser.parse_args()

    query = args.query.strip()
    if not query:
        query = input("Query: ").strip()
    if not query:
        print("No query provided. Exiting.")
        return

    url = args.url or os.environ.get("MCP_HTTP_URL", MCP_HTTP_URL)
    result = asyncio.run(
        run(
            query=query,
            mode=args.mode,
            url=url,
            verbose=not args.quiet,
            use_rest_deep=args.use_rest_deep,
        )
    )
    if args.quiet:
        print(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":
    main()

