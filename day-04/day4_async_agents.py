"""
🤖 Day 4/150 — Agentic AI Mastery Roadmap
📌 Topic: Async Python — Non-Negotiable for Agents

Key concepts:
- asyncio fundamentals — async/await, event loops
- Concurrent API calls with asyncio.gather()
- Async generators and context managers
- Simulated aiohttp for async HTTP requests
- Build: Async tool executor that runs tools simultaneously

Why async matters: Agents make PARALLEL tool calls.
Without async, your agent calls tools one-by-one = painfully slow.
With async, your agent fires all tools at once = production-ready.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
from datetime import datetime
from enum import Enum
import time
import random


# ═══════════════════════════════════════════════════════
# 1️⃣  SYNC vs ASYNC — Why Agents NEED Async
#     The #1 reason agents are slow = synchronous tool calls
# ═══════════════════════════════════════════════════════

def sync_tool_calls():
    """
    WRONG WAY: Calling tools one by one.
    Each tool waits for the previous one to finish.
    Total time = sum of all tool times.
    """
    print("\n  [SYNC] Calling 3 tools sequentially...")
    start = time.perf_counter()

    # Simulating 3 API calls (1 second each)
    time.sleep(1)  # Tool 1: Search
    print("    Tool 1 (search)    — done")
    time.sleep(1)  # Tool 2: Analyze
    print("    Tool 2 (analyze)   — done")
    time.sleep(1)  # Tool 3: Summarize
    print("    Tool 3 (summarize) — done")

    elapsed = time.perf_counter() - start
    print(f"  [SYNC] Total: {elapsed:.1f}s (slow!)")
    return elapsed


async def async_tool_calls():
    """
    RIGHT WAY: Calling all tools in parallel.
    All tools start at the same time.
    Total time = time of the SLOWEST tool.
    """
    print("\n  [ASYNC] Calling 3 tools in parallel...")
    start = time.perf_counter()

    async def tool_search():
        await asyncio.sleep(1)
        return "Found 5 results"

    async def tool_analyze():
        await asyncio.sleep(1)
        return "3 key patterns found"

    async def tool_summarize():
        await asyncio.sleep(1)
        return "Summary generated"

    # Fire all 3 tools at the same time!
    results = await asyncio.gather(
        tool_search(),
        tool_analyze(),
        tool_summarize(),
    )

    elapsed = time.perf_counter() - start
    for i, r in enumerate(results, 1):
        print(f"    Tool {i} — {r}")
    print(f"  [ASYNC] Total: {elapsed:.1f}s (3x faster!)")
    return elapsed


# ═══════════════════════════════════════════════════════
# 2️⃣  ASYNC AGENT TOOL SYSTEM
#     Production agents call tools asynchronously
#     This is how LangGraph and CrewAI execute tools
# ═══════════════════════════════════════════════════════

class ToolStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class AsyncToolResult:
    """Result from an async tool execution"""
    tool_name: str
    status: ToolStatus
    data: Any = None
    error: str | None = None
    duration_ms: float = 0


@dataclass
class AsyncTool:
    """An async tool that an agent can call"""
    name: str
    description: str
    execute: Callable[..., Awaitable[str]]


class AsyncToolExecutor:
    """
    Executes multiple tools concurrently.
    This is the engine behind every production agent.

    LangGraph does exactly this:
    - Agent decides which tools to call
    - ToolExecutor fires them all in parallel
    - Results come back as they complete
    """

    def __init__(self):
        self._tools: dict[str, AsyncTool] = {}
        self._results: list[AsyncToolResult] = []

    def register(self, tool: AsyncTool):
        self._tools[tool.name] = tool

    async def execute_single(
        self, name: str, **kwargs
    ) -> AsyncToolResult:
        """Execute a single tool with timing"""
        start = time.perf_counter()
        try:
            data = await self._tools[name].execute(**kwargs)
            duration = (time.perf_counter() - start) * 1000
            result = AsyncToolResult(
                name, ToolStatus.SUCCESS, data,
                duration_ms=round(duration, 1)
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            result = AsyncToolResult(
                name, ToolStatus.FAILED, error=str(e),
                duration_ms=round(duration, 1)
            )
        self._results.append(result)
        return result

    async def execute_parallel(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[AsyncToolResult]:
        """
        Execute multiple tools in parallel.
        This is THE key function — asyncio.gather()
        fires all tools simultaneously.
        """
        tasks = [
            self.execute_single(
                call["name"], **call.get("args", {})
            )
            for call in tool_calls
        ]
        # asyncio.gather = run all at once!
        return await asyncio.gather(*tasks)

    def get_metrics(self) -> dict:
        total = len(self._results)
        success = sum(
            1 for r in self._results
            if r.status == ToolStatus.SUCCESS
        )
        avg_ms = (
            sum(r.duration_ms for r in self._results) / total
            if total > 0 else 0
        )
        return {
            "total_calls": total,
            "success_rate": f"{success}/{total}",
            "avg_duration_ms": round(avg_ms, 1),
        }


# ═══════════════════════════════════════════════════════
# 3️⃣  ASYNC AGENT — Full Agent with Parallel Tool Calls
#     Combines the Think-Act-Observe loop with async tools
# ═══════════════════════════════════════════════════════

@dataclass
class AgentState:
    task: str = ""
    plan: list[str] = field(default_factory=list)
    tool_results: list[AsyncToolResult] = field(
        default_factory=list
    )
    final_answer: str = ""


class AsyncResearchAgent:
    """
    A production-style async agent.
    Think -> Act (parallel tools) -> Observe.
    """

    def __init__(self, name: str):
        self.name = name
        self.state = AgentState()
        self.executor = AsyncToolExecutor()
        self._register_tools()

    def _register_tools(self):
        """Register async tools"""

        async def search(query: str) -> str:
            delay = random.uniform(0.3, 0.8)
            await asyncio.sleep(delay)
            return f"Found 5 results for '{query}'"

        async def fetch_url(url: str) -> str:
            delay = random.uniform(0.2, 0.6)
            await asyncio.sleep(delay)
            return f"Fetched content from {url} (2.4KB)"

        async def analyze(data: str) -> str:
            delay = random.uniform(0.4, 0.9)
            await asyncio.sleep(delay)
            return f"Analysis: 3 key insights from '{data}'"

        async def summarize(text: str) -> str:
            delay = random.uniform(0.2, 0.5)
            await asyncio.sleep(delay)
            return f"Summary: Key points from '{text}'"

        async def fact_check(claim: str) -> str:
            delay = random.uniform(0.3, 0.7)
            await asyncio.sleep(delay)
            return f"Verified: '{claim}' is accurate (92%)"

        self.executor.register(
            AsyncTool("search", "Web search", search)
        )
        self.executor.register(
            AsyncTool("fetch", "Fetch URL", fetch_url)
        )
        self.executor.register(
            AsyncTool("analyze", "Analyze data", analyze)
        )
        self.executor.register(
            AsyncTool("summarize", "Summarize text", summarize)
        )
        self.executor.register(
            AsyncTool("fact_check", "Verify claims", fact_check)
        )

    async def think(self, task: str) -> list[dict]:
        """Plan which tools to call and with what args"""
        self.state.task = task
        print(f"\n  [THINK] Planning tools for: '{task}'")

        # Agent decides tool calls based on the task
        tool_calls = [
            {"name": "search", "args": {"query": task}},
            {"name": "fetch",
             "args": {"url": f"https://docs.ai/{task[:10]}"}},
            {"name": "analyze", "args": {"data": task}},
            {"name": "summarize", "args": {"text": task}},
            {"name": "fact_check",
             "args": {"claim": f"{task} is important"}},
        ]

        self.state.plan = [tc["name"] for tc in tool_calls]
        print(f"  [THINK] Plan: {' + '.join(self.state.plan)}")
        print(f"  [THINK] Calling {len(tool_calls)} tools "
              f"in PARALLEL...")
        return tool_calls

    async def act(
        self, tool_calls: list[dict]
    ) -> list[AsyncToolResult]:
        """Execute all tools in parallel"""
        start = time.perf_counter()

        results = await self.executor.execute_parallel(
            tool_calls
        )

        elapsed = (time.perf_counter() - start) * 1000
        self.state.tool_results = results

        print(f"\n  [ACT] Results ({elapsed:.0f}ms total):")
        for r in results:
            icon = "[+]" if r.status == ToolStatus.SUCCESS \
                else "[!]"
            print(f"    {icon} {r.tool_name}: {r.data} "
                  f"({r.duration_ms:.0f}ms)")
        return results

    async def observe(
        self, results: list[AsyncToolResult]
    ) -> str:
        """Synthesize results into final answer"""
        successful = [
            r for r in results
            if r.status == ToolStatus.SUCCESS
        ]
        answer = (
            f"Completed: {len(successful)}/{len(results)} "
            f"tools succeeded"
        )
        self.state.final_answer = answer

        metrics = self.executor.get_metrics()
        print(f"\n  [OBSERVE] {answer}")
        print(f"  [OBSERVE] Metrics: {metrics}")
        return answer

    async def run(self, task: str) -> str:
        """The async Think-Act-Observe loop"""
        print(f"\n{'='*55}")
        print(f"  Async Agent: {self.name}")
        print(f"  Task: {task}")
        print(f"{'='*55}")

        tool_calls = await self.think(task)   # Plan
        results = await self.act(tool_calls)  # Execute
        answer = await self.observe(results)  # Synthesize
        return answer


# ═══════════════════════════════════════════════════════
# 4️⃣  ASYNC GENERATOR — Streaming Agent Responses
#     Agents don't return everything at once
#     They STREAM results as they complete
# ═══════════════════════════════════════════════════════

async def stream_agent_response(task: str):
    """
    Async generator that streams agent reasoning steps.
    This is how ChatGPT streams text to you token-by-token.
    """
    steps = [
        ("thinking", f"Analyzing task: '{task}'"),
        ("planning", "Identified 3 sub-tasks"),
        ("tool_call", "Calling search API..."),
        ("tool_result", "Found 5 relevant documents"),
        ("tool_call", "Calling analyze API..."),
        ("tool_result", "Extracted 3 key insights"),
        ("synthesizing", "Combining results..."),
        ("complete", "Final answer ready"),
    ]

    for step_type, content in steps:
        await asyncio.sleep(0.2)  # Simulate processing
        yield {"type": step_type, "content": content}


# ═══════════════════════════════════════════════════════
# 5️⃣  ASYNC CONTEXT MANAGER — Resource Management
#     Agents need to manage API connections properly
# ═══════════════════════════════════════════════════════

class AsyncAgentSession:
    """
    Async context manager for agent sessions.
    Manages connections, cleanup, and resource lifecycle.
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.start_time: float = 0
        self.is_connected: bool = False

    async def __aenter__(self):
        """Setup: connect to APIs, initialize resources"""
        self.start_time = time.perf_counter()
        self.is_connected = True
        print(f"  [SESSION] {self.agent_name}: Connected")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup: close connections, log metrics"""
        duration = time.perf_counter() - self.start_time
        self.is_connected = False
        print(f"  [SESSION] {self.agent_name}: Disconnected "
              f"(session: {duration:.1f}s)")
        return False  # Don't suppress exceptions


# ═══════════════════════════════════════════════════════
#  RUN ALL DEMOS
# ═══════════════════════════════════════════════════════

async def main():
    # --- Demo 1: Sync vs Async Speed Comparison ---
    print("=" * 55)
    print("  DEMO 1: Sync vs Async (Speed Test)")
    print("=" * 55)

    sync_time = sync_tool_calls()
    async_time = await async_tool_calls()

    speedup = sync_time / async_time
    print(f"\n  Speedup: {speedup:.1f}x faster with async!")
    print(f"  Sync: {sync_time:.1f}s | Async: {async_time:.1f}s")

    # --- Demo 2: Async Agent with Parallel Tools ---
    print("\n" + "=" * 55)
    print("  DEMO 2: Async Research Agent (5 Parallel Tools)")
    print("=" * 55)

    agent = AsyncResearchAgent("Nova")
    await agent.run("LangGraph async agent patterns")

    # --- Demo 3: Streaming Responses ---
    print("\n" + "=" * 55)
    print("  DEMO 3: Async Generator (Streaming Response)")
    print("=" * 55)

    print("\n  Streaming agent response:")
    async for step in stream_agent_response("AI agents"):
        icon = {
            "thinking": "...",
            "planning": "[P]",
            "tool_call": "-->",
            "tool_result": "[+]",
            "synthesizing": "~~~",
            "complete": "[*]",
        }.get(step["type"], "   ")
        print(f"    {icon} {step['content']}")

    # --- Demo 4: Async Context Manager ---
    print("\n" + "=" * 55)
    print("  DEMO 4: Async Context Manager (Session Mgmt)")
    print("=" * 55)

    async with AsyncAgentSession("Nova") as session:
        print(f"  [SESSION] Connected: {session.is_connected}")
        agent2 = AsyncResearchAgent("Nova")
        await agent2.run("MCP Protocol overview")

    # --- Final Summary ---
    print(f"""
{'='*55}
  Day 4 — Async Patterns for Agents
{'='*55}

  [1] async/await     = Non-blocking agent execution
  [2] asyncio.gather  = Parallel tool calls (THE key)
  [3] Async generator = Streaming agent responses
  [4] Async context   = Session & resource management
  [5] Speedup         = {speedup:.0f}x faster than sync!

  Without async → tools run one-by-one → slow agent
  With async    → tools run in parallel → fast agent

  This is NON-NEGOTIABLE for production agents.
{'='*55}
""")


if __name__ == "__main__":
    asyncio.run(main())
