"""
🤖 Day 3/150 — Agentic AI Mastery Roadmap
📌 Topic: Object-Oriented Python for Agent Architecture

Key concepts:
- Classes, inheritance, composition — agents are objects with state
- Abstract base classes (ABC) — defining agent interfaces
- Dataclasses & Pydantic models — THE way to define agent state/schemas
- Design patterns: Strategy, Observer, Chain of Responsibility
- Build: A mini "agent" class with state, tools, and execute method

This is how LangGraph, CrewAI, and AutoGen define agents internally.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
from datetime import datetime
from enum import Enum


# ═══════════════════════════════════════════════════════
# 1️⃣  ABSTRACT BASE CLASS — The Agent Interface
#     Every agent framework defines a base interface
#     This is the "contract" all agents must follow
# ═══════════════════════════════════════════════════════

class AgentStatus(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting_for_input"
    COMPLETED = "completed"
    FAILED = "failed"


class BaseAgent(ABC):
    """
    Abstract Base Class for all agents.
    This is exactly how LangGraph & CrewAI define
    their agent interfaces internally.

    Every agent MUST implement:
    - think()   -> reasoning step
    - act()     -> execution step
    - observe() -> process results
    """

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.status = AgentStatus.IDLE
        self.memory: list[dict[str, Any]] = []

    @abstractmethod
    def think(self, task: str) -> str:
        """Reasoning step — plan how to solve the task"""
        pass

    @abstractmethod
    def act(self, plan: str) -> dict[str, Any]:
        """Execution step — take action based on the plan"""
        pass

    @abstractmethod
    def observe(self, result: dict[str, Any]) -> str:
        """Observation step — process and learn from results"""
        pass

    def run(self, task: str) -> str:
        """
        The Think-Act-Observe loop.
        This is THE core pattern of every AI agent.
        """
        print(f"\n{'='*55}")
        print(f"  Agent: {self.name} ({self.role})")
        print(f"  Task:  {task}")
        print(f"{'='*55}")

        # Step 1: Think
        self.status = AgentStatus.THINKING
        plan = self.think(task)
        self._remember("thought", plan)
        print(f"\n  [THINK]   {plan}")

        # Step 2: Act
        self.status = AgentStatus.EXECUTING
        result = self.act(plan)
        self._remember("action", result)
        print(f"  [ACT]     {result}")

        # Step 3: Observe
        observation = self.observe(result)
        self._remember("observation", observation)
        print(f"  [OBSERVE] {observation}")

        self.status = AgentStatus.COMPLETED
        print(f"\n  Status: {self.status.value}")
        return observation

    def _remember(self, step_type: str, content: Any):
        """Store step in agent memory"""
        self.memory.append({
            "type": step_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })


# ═══════════════════════════════════════════════════════
# 2️⃣  PYDANTIC-STYLE MODELS — Agent State & Schemas
#     Dataclasses are how agent state is defined
#     in production frameworks
# ═══════════════════════════════════════════════════════

@dataclass
class ToolResult:
    """Result from a tool execution"""
    tool_name: str
    success: bool
    data: Any
    error: str | None = None


@dataclass
class AgentState:
    """
    The complete state of an agent at any point.
    LangGraph's StateGraph uses exactly this pattern.
    """
    messages: list[str] = field(default_factory=list)
    current_task: str = ""
    plan: list[str] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    final_answer: str = ""
    step_count: int = 0
    max_steps: int = 10

    def add_message(self, role: str, content: str):
        self.messages.append(f"[{role}] {content}")
        self.step_count += 1

    def is_done(self) -> bool:
        return (
            self.final_answer != ""
            or self.step_count >= self.max_steps
        )


# ═══════════════════════════════════════════════════════
# 3️⃣  TOOL SYSTEM — Strategy Pattern
#     Tools are pluggable behaviors — classic Strategy pattern
#     This is how OpenAI function calling works under the hood
# ═══════════════════════════════════════════════════════

@dataclass
class Tool:
    """A tool that an agent can use"""
    name: str
    description: str
    function: Callable[..., str]


class ToolRegistry:
    """
    Registry of tools available to agents.
    Strategy Pattern — swap tools without changing agent code.
    """
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool
        print(f"    Registered tool: {tool.name}")

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        if tool_name not in self._tools:
            return ToolResult(tool_name, False, None,
                              f"Tool '{tool_name}' not found")
        try:
            result = self._tools[tool_name].function(**kwargs)
            return ToolResult(tool_name, True, result)
        except Exception as e:
            return ToolResult(tool_name, False, None, str(e))

    def list_tools(self) -> list[str]:
        return [
            f"{t.name}: {t.description}"
            for t in self._tools.values()
        ]


# ═══════════════════════════════════════════════════════
# 4️⃣  CONCRETE AGENT — Putting It All Together
#     A Research Agent with tools, state, and memory
# ═══════════════════════════════════════════════════════

class ResearchAgent(BaseAgent):
    """
    A concrete agent that inherits from BaseAgent.
    Uses composition to include tools and state.
    This mirrors how you'd build an agent with LangGraph.
    """

    def __init__(self, name: str):
        super().__init__(name, role="Research Specialist")
        self.tools = ToolRegistry()
        self.state = AgentState()
        self._setup_tools()

    def _setup_tools(self):
        """Register available tools"""
        print(f"\n  Setting up tools for {self.name}:")

        self.tools.register(Tool(
            name="search",
            description="Search the web for information",
            function=lambda query: f"Found 5 results for: {query}"
        ))
        self.tools.register(Tool(
            name="summarize",
            description="Summarize a block of text",
            function=lambda text:
                f"Summary: {text[:50]}..." if len(text) > 50
                else f"Summary: {text}"
        ))
        self.tools.register(Tool(
            name="analyze",
            description="Analyze data for patterns",
            function=lambda data:
                f"Analysis: Found {len(str(data))} data points"
        ))

    def think(self, task: str) -> str:
        """Plan which tools to use for this task"""
        self.state.current_task = task
        self.state.add_message("user", task)

        # Simple planning logic
        plan_steps = []
        if "search" in task.lower() or "find" in task.lower():
            plan_steps.append("search")
        if "summarize" in task.lower() or "explain" in task.lower():
            plan_steps.append("summarize")
        plan_steps.append("analyze")

        self.state.plan = plan_steps
        return f"Plan: use {' -> '.join(plan_steps)}"

    def act(self, plan: str) -> dict[str, Any]:
        """Execute tools from the plan"""
        results = {}
        for tool_name in self.state.plan:
            result = self.tools.execute(
                tool_name,
                **self._get_tool_args(tool_name)
            )
            self.state.tool_results.append(result)
            results[tool_name] = result.data
            self.state.add_message("tool", f"{tool_name}: {result.data}")
        return results

    def act(self, plan: str) -> dict[str, Any]:
        """Execute tools from the plan"""
        results = {}
        for tool_name in self.state.plan:
            result = self.tools.execute(
                tool_name,
                **self._get_tool_args(tool_name)
            )
            self.state.tool_results.append(result)
            results[tool_name] = result.data
            self.state.add_message("tool",
                                   f"{tool_name}: {result.data}")
        return results

    def observe(self, result: dict[str, Any]) -> str:
        """Process results and form final answer"""
        summary = " | ".join(
            f"{k}: {v}" for k, v in result.items()
        )
        self.state.final_answer = f"Completed: {summary}"
        self.state.add_message("agent", self.state.final_answer)
        return self.state.final_answer

    def _get_tool_args(self, tool_name: str) -> dict:
        """Get arguments for each tool"""
        task = self.state.current_task
        args_map = {
            "search": {"query": task},
            "summarize": {"text": task},
            "analyze": {"data": task},
        }
        return args_map.get(tool_name, {})


# ═══════════════════════════════════════════════════════
# 5️⃣  OBSERVER PATTERN — Agent Event System
#     Agents notify observers when events happen
#     Used in CrewAI for agent coordination
# ═══════════════════════════════════════════════════════

class AgentEventType(Enum):
    TASK_STARTED = "task_started"
    TOOL_CALLED = "tool_called"
    TASK_COMPLETED = "task_completed"
    ERROR = "error"


@dataclass
class AgentEvent:
    agent_name: str
    event_type: AgentEventType
    data: Any = None


class AgentObserver(ABC):
    """Observer interface for monitoring agents"""
    @abstractmethod
    def on_event(self, event: AgentEvent):
        pass


class AgentLogger(AgentObserver):
    """Logs all agent events — like production observability"""
    def on_event(self, event: AgentEvent):
        icon = {
            AgentEventType.TASK_STARTED: "-->",
            AgentEventType.TOOL_CALLED: "[T]",
            AgentEventType.TASK_COMPLETED: "[+]",
            AgentEventType.ERROR: "[!]",
        }
        print(f"    {icon[event.event_type]} "
              f"[{event.agent_name}] "
              f"{event.event_type.value}: {event.data}")


class AgentMetrics(AgentObserver):
    """Tracks metrics — like AgentOps in production"""
    def __init__(self):
        self.event_counts: dict[str, int] = {}

    def on_event(self, event: AgentEvent):
        key = event.event_type.value
        self.event_counts[key] = (
            self.event_counts.get(key, 0) + 1
        )

    def report(self) -> dict[str, int]:
        return self.event_counts


# ═══════════════════════════════════════════════════════
# 6️⃣  CHAIN OF RESPONSIBILITY — Agent Pipeline
#     Tasks flow through a chain of agents
#     Each agent decides: handle it or pass it on
# ═══════════════════════════════════════════════════════

class ChainableAgent(BaseAgent):
    """Agent that can be chained with other agents"""

    def __init__(self, name: str, role: str,
                 specialty: str):
        super().__init__(name, role)
        self.specialty = specialty
        self.next_agent: ChainableAgent | None = None

    def set_next(self, agent: "ChainableAgent"):
        """Chain this agent to the next one"""
        self.next_agent = agent
        return agent

    def handle(self, task: str) -> str:
        """Handle task or pass to next agent in chain"""
        if self.specialty.lower() in task.lower():
            return self.run(task)
        elif self.next_agent:
            print(f"  [{self.name}] Can't handle this "
                  f"-> passing to {self.next_agent.name}")
            return self.next_agent.handle(task)
        else:
            return f"No agent can handle: {task}"

    def think(self, task: str) -> str:
        return f"I specialize in {self.specialty}"

    def act(self, plan: str) -> dict[str, Any]:
        return {"result": f"Handled by {self.name}"}

    def observe(self, result: dict[str, Any]) -> str:
        return f"Done: {self.name} completed the task"


# ═══════════════════════════════════════════════════════
#  RUN EVERYTHING
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":

    # --- Demo 1: Research Agent with Tools ---
    print("\n" + "=" * 55)
    print("  DEMO 1: Research Agent (Inheritance + Composition)")
    print("=" * 55)

    agent = ResearchAgent("Atlas")
    result = agent.run(
        "Search and summarize info about LangGraph agents"
    )

    print(f"\n  Memory entries: {len(agent.memory)}")
    print(f"  Steps taken: {agent.state.step_count}")

    # --- Demo 2: Observer Pattern ---
    print("\n" + "=" * 55)
    print("  DEMO 2: Observer Pattern (Agent Monitoring)")
    print("=" * 55)

    logger = AgentLogger()
    metrics = AgentMetrics()

    # Simulate events
    events = [
        AgentEvent("Atlas", AgentEventType.TASK_STARTED,
                   "Research LangGraph"),
        AgentEvent("Atlas", AgentEventType.TOOL_CALLED,
                   "search('LangGraph')"),
        AgentEvent("Atlas", AgentEventType.TOOL_CALLED,
                   "summarize(results)"),
        AgentEvent("Atlas", AgentEventType.TASK_COMPLETED,
                   "Report generated"),
    ]

    print("\n  Event Log:")
    for event in events:
        logger.on_event(event)
        metrics.on_event(event)

    print(f"\n  Metrics: {metrics.report()}")

    # --- Demo 3: Chain of Responsibility ---
    print("\n" + "=" * 55)
    print("  DEMO 3: Chain of Responsibility (Agent Pipeline)")
    print("=" * 55)

    # Build the chain
    coder = ChainableAgent("CodeBot", "Developer", "code")
    researcher = ChainableAgent("Scholar", "Researcher",
                                "research")
    writer = ChainableAgent("Scribe", "Writer", "write")

    # Chain: coder -> researcher -> writer
    coder.set_next(researcher).set_next(writer)

    # Send a research task — it will flow through chain
    print("\n  Sending task: 'research about AI agents'")
    coder.handle("research about AI agents")

    # --- Final Summary ---
    print(f"""
{'='*55}
  Day 3 Key Patterns for Agent Architecture
{'='*55}

  [1] ABC (Abstract Base)   = Agent interface contracts
  [2] Inheritance            = Agent specialization
  [3] Composition            = Agent + Tools + State
  [4] Strategy Pattern       = Pluggable tool behaviors
  [5] Observer Pattern       = Agent event monitoring
  [6] Chain of Responsibility = Agent task pipelines
  [7] Dataclasses            = Agent state schemas
  [8] Enums                  = Agent status tracking

  These aren't just OOP concepts --
  they're how LangGraph, CrewAI & AutoGen are built!
{'='*55}
""")
