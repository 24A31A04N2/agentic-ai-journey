# 🤖 Day 02/150 — Data Structures & Algorithms That Agents Use

![Day](https://img.shields.io/badge/Day-02%2F150-blue?style=flat-square)
![Phase](https://img.shields.io/badge/Phase-1%3A%20Foundations-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11-3572a5?style=flat-square&logo=python)
![Tests](https://img.shields.io/badge/Code-Verified-3fb950?style=flat-square)

> **Key Insight:** Agent workflows are Directed Graphs.  
> This is literally how **LangGraph** works internally — BFS for parallel tasks, DFS for sequential chains.

---

## 📌 What I Learned Today

| Concept | Agent Application | Framework That Uses It |
|---------|-------------------|----------------------|
| `dict` & `defaultdict` | Agent state storage — JSON data passing | LangGraph `StateGraph` |
| List comprehensions | Data transformation pipelines | All agent frameworks |
| Directed Graphs + BFS/DFS | Agent workflow execution order | LangGraph, AutoGen |
| `deque` / Priority Queue | Multi-agent message passing | CrewAI, AutoGen |
| `@dataclass` | Typed agent node definitions | LangGraph nodes |

---

## 🔨 What I Built

### 1. `AgentWorkflowGraph` — Graph-based workflow engine

Models an agent workflow as a directed graph and supports two traversal strategies:

- **BFS** → Breadth-First = finds all nodes at each level first → used for **parallel task scheduling**
- **DFS** → Depth-First = follows each path to the end → used for **sequential reasoning chains**

```
Workflow Graph:
  start ──► planner ──► searcher ──► responder
                    └──► analyzer ──► responder
```

**BFS output:**  `start → planner → searcher → analyzer → responder`  
**DFS output:**  `start → planner → searcher → responder → analyzer`

### 2. `AgentMessageQueue` — Priority-aware message queue

Agents communicate by pushing/popping messages from a deque.  
High-priority messages (e.g., rate-limit alerts) jump to the front instantly.

### 3. Agent State with `defaultdict`

```python
memory = defaultdict(list)
memory["observations"].append("Found 3 relevant documents")
# Never raises KeyError — perfect for dynamic agent state
```

---

## 📂 Files

| File | Description |
|------|-------------|
| [`day2_agents_data_structures.py`](./day2_agents_data_structures.py) | Main code — all 4 concepts with full annotations |

---

## ▶️ Run It

```bash
# Clone the repo
git clone https://github.com/battulamahendra723/agentic-ai-journey.git
cd agentic-ai-journey/day-02

# Run (no dependencies needed — pure Python stdlib)
python day2_agents_data_structures.py
```

**Expected output:**

```
Current task: task-42
Agent observations: ['User wants data analysis', 'Found 3 relevant documents']
High quality results: ['Python is great', 'LangGraph is stateful']

🔵 BFS Order (parallel scheduling):
start → planner → searcher → analyzer → responder

🟢 DFS Order (sequential chain):
start → planner → searcher → responder → analyzer

📬 Messages pending: 3
  [monitor] → [planner]: ⚠️ Rate limit hit!
  [planner] → [searcher]: Search for Python docs
  [planner] → [analyzer]: Analyze top 5 results

╔══════════════════════════════════════════════╗
║  🧠 Day 2 Key Insight                        ║
║  Agent workflows = Directed Graphs           ║
║  Agent state    = Nested Dictionaries        ║
║  Agent comms    = Message Queues (Deques)    ║
╚══════════════════════════════════════════════╝
```

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| LangGraph Official Docs | https://langchain-ai.github.io/langgraph/ |
| Python `collections` module | https://docs.python.org/3/library/collections.html |
| BFS Algorithm — GeeksForGeeks | https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/ |
| Python Dataclasses — Real Python | https://realpython.com/python-data-classes/ |
| `defaultdict` Guide | https://realpython.com/python-defaultdict/ |
| LangGraph GitHub Source | https://github.com/langchain-ai/langgraph |

---

## 💡 Connection to LangGraph

LangGraph's `StateGraph` is literally a directed graph where:
- Each **node** is an agent step (a Python function)
- Each **edge** is a transition condition
- The graph can be traversed sequentially or in parallel

Understanding BFS/DFS today = understanding how LangGraph schedules work in Phase 4.

---

## 🗓️ What's Next

**Day 3:** Object-Oriented Python for Agent Architecture  
→ Classes, inheritance, ABC, Pydantic models → building a mini-agent class

---

*Part of the [150-Day Agentic AI Mastery Roadmap](../README.md)*  
*Target roles: AI Agent Engineer · LLMOps Engineer · Agent Orchestration Architect*
