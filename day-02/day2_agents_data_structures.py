"""
🤖 Day 2 of 150 — Agentic AI Mastery Roadmap
📌 Topic: Data Structures & Algorithms That Agents Use

Key concepts:
- Dictionaries (nested dicts, defaultdict) — agents pass JSON data constantly
- Lists, sets, comprehensions — data transformation pipelines
- Graphs & trees basics — agent workflows ARE graphs
- Deques & queues — message passing in multi-agent systems

🔨 Build: Graph Traversal (BFS/DFS) — How LangGraph works internally
"""

from collections import defaultdict, deque
from typing import Any, Optional
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# 1️⃣  AGENT STATE — Deeply Nested Dicts
#     (This is exactly how LangGraph stores state)
# ─────────────────────────────────────────────

agent_state: dict[str, Any] = {
    "agent_id": "research-agent-001",
    "memory": {
        "short_term": ["user asked about AI agents"],
        "long_term": defaultdict(list),
    },
    "tools_available": {"search", "calculator", "code_executor"},
    "current_task": {
        "id": "task-42",
        "status": "in_progress",
        "sub_tasks": [],
    },
}

# Accessing nested state safely
task_id = agent_state.get("current_task", {}).get("id", "unknown")
print(f"Current task: {task_id}")

# Agent memory using defaultdict — never KeyError!
memory = defaultdict(list)
memory["observations"].append("User wants data analysis")
memory["observations"].append("Found 3 relevant documents")
memory["actions"].append("Called search tool")
print(f"Agent observations: {memory['observations']}")


# ─────────────────────────────────────────────
# 2️⃣  DATA TRANSFORMATION PIPELINES
#     (How agents process tool outputs)
# ─────────────────────────────────────────────

raw_tool_results = [
    {"source": "web", "content": "Python is great", "relevance": 0.9},
    {"source": "wiki", "content": "Agents use graphs", "relevance": 0.4},
    {"source": "paper", "content": "LangGraph is stateful", "relevance": 0.85},
    {"source": "blog", "content": "AI is changing", "relevance": 0.2},
]

# Filter high-relevance results + extract content (list comprehension)
high_quality = [
    r["content"]
    for r in raw_tool_results
    if r["relevance"] >= 0.8
]
# → ['Python is great', 'LangGraph is stateful']

# Unique sources (set comprehension)
sources = {r["source"] for r in raw_tool_results}
# → {'web', 'wiki', 'paper', 'blog'}

print(f"High quality results: {high_quality}")
print(f"Unique sources: {sources}")


# ─────────────────────────────────────────────
# 3️⃣  GRAPH TRAVERSAL — THIS IS HOW LANGGRAPH WORKS!
#     Every agent workflow is a directed graph
# ─────────────────────────────────────────────

@dataclass
class AgentNode:
    """A node in the agent workflow graph"""
    name: str
    action: str
    next_nodes: list[str] = field(default_factory=list)


class AgentWorkflowGraph:
    """
    Represents an agent workflow as a directed graph.
    This is conceptually how LangGraph operates internally.
    """
    def __init__(self):
        self.nodes: dict[str, AgentNode] = {}
        self.edges: dict[str, list[str]] = defaultdict(list)

    def add_node(self, node: AgentNode):
        self.nodes[node.name] = node

    def add_edge(self, from_node: str, to_node: str):
        self.edges[from_node].append(to_node)

    def bfs_execution_order(self, start: str) -> list[str]:
        """
        BFS — finds breadth-first execution order.
        Used for parallel agent task scheduling.
        """
        visited = set()
        queue = deque([start])
        execution_order = []

        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                execution_order.append(node)
                for neighbor in self.edges[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)

        return execution_order

    def dfs_execution_path(
        self, start: str, visited: Optional[set] = None
    ) -> list[str]:
        """
        DFS — finds depth-first execution path.
        Used for sequential agent reasoning chains.
        """
        if visited is None:
            visited = set()

        visited.add(start)
        path = [start]

        for neighbor in self.edges[start]:
            if neighbor not in visited:
                path.extend(
                    self.dfs_execution_path(neighbor, visited)
                )
        return path


# Build a real-world agent workflow graph
workflow = AgentWorkflowGraph()

# Nodes represent agent steps
workflow.add_node(AgentNode("start", "receive_user_query"))
workflow.add_node(AgentNode("planner", "break_into_subtasks"))
workflow.add_node(AgentNode("searcher", "search_web"))
workflow.add_node(AgentNode("analyzer", "analyze_results"))
workflow.add_node(AgentNode("responder", "generate_response"))

# Edges represent workflow transitions
workflow.add_edge("start", "planner")
workflow.add_edge("planner", "searcher")
workflow.add_edge("planner", "analyzer")   # parallel execution!
workflow.add_edge("searcher", "responder")
workflow.add_edge("analyzer", "responder")

print("\n🔵 BFS Order (parallel scheduling):")
print(" → ".join(workflow.bfs_execution_order("start")))

print("\n🟢 DFS Order (sequential chain):")
print(" → ".join(workflow.dfs_execution_path("start")))


# ─────────────────────────────────────────────
# 4️⃣  DEQUE & QUEUES — MULTI-AGENT MESSAGE PASSING
#     Agents communicate through message queues
# ─────────────────────────────────────────────

@dataclass
class AgentMessage:
    sender: str
    receiver: str
    content: str
    priority: int = 1  # higher = more urgent


class AgentMessageQueue:
    """
    Message queue for multi-agent communication.
    Agents push/pop messages to coordinate.
    """
    def __init__(self):
        self._queue: deque[AgentMessage] = deque()

    def send(self, message: AgentMessage):
        if message.priority > 1:
            self._queue.appendleft(message)  # High-priority → front
        else:
            self._queue.append(message)       # Normal → back

    def receive(self) -> Optional[AgentMessage]:
        if self._queue:
            return self._queue.popleft()
        return None

    def pending_count(self) -> int:
        return len(self._queue)


# Simulate multi-agent communication
msg_queue = AgentMessageQueue()

msg_queue.send(AgentMessage("planner", "searcher", "Search for Python docs"))
msg_queue.send(AgentMessage("planner", "analyzer", "Analyze top 5 results"))
msg_queue.send(
    AgentMessage("monitor", "planner", "⚠️ Rate limit hit!", priority=5)
)

print(f"\n📬 Messages pending: {msg_queue.pending_count()}")
while msg := msg_queue.receive():
    print(f"  [{msg.sender}] → [{msg.receiver}]: {msg.content}")


# ─────────────────────────────────────────────
# 💡 KEY INSIGHT FOR DAY 2
# ─────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════╗
║  🧠 Day 2 Key Insight                        ║
║                                              ║
║  Agent workflows = Directed Graphs           ║
║  Agent state    = Nested Dictionaries        ║
║  Agent comms    = Message Queues (Deques)    ║
║  Data pipelines = List Comprehensions        ║
║                                              ║
║  These aren't just CS concepts —             ║
║  they're the BACKBONE of LangGraph!          ║
╚══════════════════════════════════════════════╝
""")
