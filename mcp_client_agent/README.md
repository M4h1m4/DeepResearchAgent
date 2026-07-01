# Local Agent (HTTP) – Run FastAPI and MCP Separately

This folder contains a **local agent** that connects to the Deep Research **MCP server over HTTP** and measures **response time**. FastAPI and MCP run as **separate processes** so you can compare this setup (and its latency) with other methods later.

---

## 1. Run FastAPI and MCP separately

Use **two terminals** (or run MCP in the background).

### Terminal 1: FastAPI backend (REST API)

From the **project root** (`DeepResearchAgent`):

```bash
uv run python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- REST API: `http://127.0.0.1:8000`
- Use this for document upload, REST `/query`, evals, etc. The **local agent does not call FastAPI**; it only talks to the MCP server.

### Terminal 2: MCP server (HTTP)

From the **same project root**:

```bash
uv run python -m app.mcp.server --http
```

- MCP (Streamable HTTP): `http://127.0.0.1:8001/mcp` (server serves at path `/mcp`)
- The **agent connects here** to call `rag_query` and `deep_research`.

Summary:

| Process   | Command                                              | Port | Purpose                    |
|----------|------------------------------------------------------|------|----------------------------|
| FastAPI  | `uv run python -m uvicorn app.main:app --host 0.0.0.0 --port 8000` | 8000 | REST API (optional for agent) |
| MCP      | `uv run python -m app.mcp.server --http`             | 8001 | MCP tools (agent uses this) |

---

## 2. Create the agent script

Create **`agent.py`** in this folder (`mcp_client_agent/`) and type the code from the section **"Code: agent.py"** below. Then run it as in section 4.

---

## 3. Code: agent.py

Put this in **`mcp_client_agent/agent.py`** (you can type it manually):

```python
"""
Local agent: connects to the Deep Research MCP server over HTTP and measures response time.

Prerequisites:
  - MCP server running: uv run python -m app.mcp.server --http  (port 8001)
  - Optional: FastAPI backend running separately: uv run python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

Usage:
  uv run python mcp_client_agent/agent.py
  uv run python mcp_client_agent/agent.py --query "Your question" --mode fast
  MCP_HTTP_URL=http://127.0.0.1:8001 uv run python mcp_client_agent/agent.py
"""

import argparse
import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

# MCP server URL when run with --http (default: port 8001, path /mcp)
MCP_HTTP_URL = os.environ.get("MCP_HTTP_URL", "http://127.0.0.1:8001/mcp")


def parse_tool_result(result):
    """Extract JSON or text from MCP CallToolResult."""
    if result.is_error:
        return {
            "error": True,
            "message": result.content[0].text if result.content else "Unknown error",
        }
    if not result.content:
        return {"answer": "", "sources": [], "response_time_ms": 0}
    text = result.content[0].text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"answer": text, "sources": []}


@asynccontextmanager
async def connect_mcp_http(url: str):
    """Connect to MCP server via Streamable HTTP. Yields (session, connect_time_ms)."""
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    t0 = time.perf_counter()
    async with streamablehttp_client(url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            connect_ms = int((time.perf_counter() - t0) * 1000)
            yield session, connect_ms


async def call_tool_with_timing(session, tool_name: str, arguments: dict):
    """()
    Call an MCP tool and return (parsed_result, tool_response_time_ms).
    tool_response_time_ms is from just before call_tool to just after result received.
    """
    t0 = time.perf_counter()
    result = await session.call_tool(tool_name, arguments)
    tool_ms = int((time.perf_counter() - t0) * 1000)
    data = parse_tool_result(result)
    return data, tool_ms


async def run(
    query: str,
    mode: str = "fast",
    url: str | None = None,
    verbose: bool = True,
):
    """
    Connect to MCP over HTTP, call rag_query (fast) or deep_research (deep), report response times.|b
    Returns:
        dict with keys: answer, sources, tool_response_time_ms, connect_time_ms, total_time_ms, error?
    """
    base_url = url or MCP_HTTP_URL
    total_t0 = time.perf_counter()
    connect_ms = 0
    tool_ms = 0
    data = {}

    try:
        async with connect_mcp_http(base_url) as (session, conn_ms):
            connect_ms = conn_ms
            if verbose:
                tools_result = await session.list_tools()
                tool_names = [t.name for t in tools_result.tools]
                print(f"Connected to MCP at {base_url} (connect: {connect_ms} ms). Tools: {tool_names}")

            if mode == "deep":
                data, tool_ms = await call_tool_with_timing(
                    session, "deep_research", {"query": query}
                )
            else:
                data, tool_ms = await call_tool_with_timing(
                    session, "rag_query", {"query": query}
                )
    except Exception as e:
        total_ms = int((time.perf_counter() - total_t0) * 1000)
        if verbose:
            print(f"Error: {e}")
            print(f"Total time (failed): {total_ms} ms")
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
    args = parser.parse_args()

    query = args.query.strip()
    if not query:
        query = input("Query: ").strip()
    if not query:
        print("No query provided. Exiting.")
        return

    url = args.url or os.environ.get("MCP_HTTP_URL", MCP_HTTP_URL)
    result = asyncio.run(
        run(query=query, mode=args.mode, url=url, verbose=not args.quiet)
    )
    if args.quiet:
        print(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":
    main()
```

---

## 4. Run the local agent

From the **project root**, with the **MCP server already running** (Terminal 2):

```bash
# Default: connects to http://127.0.0.1:8001/mcp, prompts for query, uses Fast RAG
uv run python mcp_client_agent/agent.py
```

**Options:**

```bash
# Pass query and mode on the command line
uv run python mcp_client_agent/agent.py --query "What are the main findings?" --mode fast

# Deep research mode
uv run python mcp_client_agent/agent.py --query "Summarize the document." --mode deep

# Use a different MCP URL (e.g. another host)
export MCP_HTTP_URL=http://127.0.0.1:8001/mcp
uv run python mcp_client_agent/agent.py --query "Your question"

# Or pass URL explicitly
uv run python mcp_client_agent/agent.py --url http://127.0.0.1:8001/mcp --query "Your question"

# JSON-only output (for scripting / measuring)
uv run python mcp_client_agent/agent.py --query "Your question" --quiet
```

The agent uses the project’s existing `mcp` dependency; no extra install in `mcp_client_agent` is required when run with `uv run` from the repo root.

---

## 5. Response time reported

The agent prints three timings:

| Timing | Meaning |
|--------|--------|
| **Response time (MCP tool call)** | Time from start of `call_tool` to result received (what the server spent on RAG / deep research). |
| **Response time (connect + init)** | Time to open the HTTP connection and complete MCP `initialize`. |
| **Response time (total, this process)** | End-to-end time for the agent (connect + init + tool call + parsing). |

Use **MCP tool call** to compare with other methods (e.g. calling REST directly or running MCP in-process). Use **total** to see full client-side latency including HTTP/MCP overhead.

---

## 6. Quick test sequence

1. **Start MCP** (Terminal 2):  
   `uv run python -m app.mcp.server --http`

2. **Create `agent.py`** in `mcp_client_agent/` using the code in section 3, then **start the agent** (Terminal 3 or after MCP is up):  
   `uv run python mcp_client_agent/agent.py --query "What is in the knowledge base?" --mode fast`

3. Check the printed response times and answer preview.

4. (Optional) Start **FastAPI** (Terminal 1) if you need the REST API for uploads or other endpoints. The agent does not require FastAPI to be running.

---

## 5. Troubleshooting

| Issue | What to do |
|-------|------------|
| Connection refused to 8001 | Start the MCP server first: `uv run python -m app.mcp.server --http`. |
| Different port/path | Set `MCP_HTTP_URL=http://127.0.0.1:YOUR_PORT/mcp` (path must be `/mcp` unless you changed the server). |
| “No query” | Pass `--query "..."` or type the query when prompted. |
| TaskGroup / sub-exception error | The agent prints **Root cause:** when available; that points to the real error (e.g. in the MCP client or transport). If the server returned 200, the result may still be recovered and printed. |
| Chroma/PostHog telemetry errors in server log | To silence them, run the MCP server with `ANONYMIZED_TELEMETRY=False` (e.g. `ANONYMIZED_TELEMETRY=False uv run python -m app.mcp.server --http`). |

You can later try other methods (e.g. in-process MCP or REST-only) and compare **tool response time** and **total time** with this HTTP setup.
