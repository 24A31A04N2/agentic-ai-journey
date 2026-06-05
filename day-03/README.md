# 🤖 Day 03/150 — Object-Oriented Python for Agent Architecture

![Day](https://img.shields.io/badge/Day-03%2F150-blue?style=flat-square)
![Phase](https://img.shields.io/badge/Phase-1%3A%20Foundations-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11-3572a5?style=flat-square&logo=python)
![Patterns](https://img.shields.io/badge/Design%20Patterns-6-bc8cff?style=flat-square)
![Code](https://img.shields.io/badge/Code-Verified-3fb950?style=flat-square)

> **Key Insight:** Every AI agent framework — LangGraph, CrewAI, AutoGen — is built on classical OOP design patterns.  
> Master the patterns, master the frameworks.

---

## 📌 What I Learned Today

| OOP Concept | Agent Application | Used In |
|-------------|-------------------|---------|
| Abstract Base Class (ABC) | Defines the agent contract — `think()`, `act()`, `observe()` | LangGraph, CrewAI, AutoGen |
| Inheritance | Agent specialization — `ResearchAgent(BaseAgent)` | All agent frameworks |
| Composition | Agent = State + Tools + Memory (not just inheritance) | LangGraph `StateGraph` |
| Strategy Pattern | Pluggable tool system — swap tools without changing agent | OpenAI Function Calling |
| Observer Pattern | Agent event monitoring — logging, metrics, alerts | CrewAI, AgentOps |
| Chain of Responsibility | Task flows through a chain of specialized agents | Multi-agent pipelines |
| Dataclasses | Typed agent state schemas | LangGraph state |
| Enums | Agent status tracking (IDLE → THINKING → EXECUTING → DONE) | All frameworks |

---

## 🔨 What I Built

### 1. `BaseAgent` — Abstract Agent Interface

An ABC that defines the **Think-Act-Observe** loop:

```python
class BaseAgent(ABC):
    @abstractmethod
    def think(self, task: str) -> str: ...   # Reasoning

    @abstractmethod
    def act(self, plan: str) -> dict: ...    # Execution

    @abstractmethod
    def observe(self, result: dict) -> str: ... # Learning

    def run(self, task: str) -> str:
        plan = self.think(task)       # Step 1: Plan
        result = self.act(plan)       # Step 2: Execute
        return self.observe(result)   # Step 3: Learn
```

### 2. `ResearchAgent` — Concrete Agent with Tools

Inherits from `BaseAgent`, uses **composition** to include a `ToolRegistry` and `AgentState`:

```
ResearchAgent
├── BaseAgent (inheritance)
├── ToolRegistry (composition — Strategy Pattern)
│   ├── search tool
│   ├── summarize tool
│   └── analyze tool
└── AgentState (composition — dataclass)
    ├── messages, plan, tool_results
    └── final_answer, step_count
```

### 3. `AgentLogger` + `AgentMetrics` — Observer Pattern

```
Event Flow:
  Agent fires event → Logger prints it → Metrics counts it
  
Output:
  --> [Atlas] task_started: Research LangGraph
  [T] [Atlas] tool_called: search('LangGraph')
  [+] [Atlas] task_completed: Report generated
  Metrics: {'task_started': 1, 'tool_called': 2, 'task_completed': 1}
```

### 4. `ChainableAgent` — Chain of Responsibility

```
Chain: CodeBot → Scholar → Scribe

Task: "research about AI agents"
  [CodeBot] Can't handle → passing to Scholar
  [Scholar] I specialize in research → HANDLED!
```

---

## 📂 Files

| File | Description |
|------|-------------|
| [`day3_agent_architecture.py`](./day3_agent_architecture.py) | Full code — 6 design patterns, 3 demos, 290+ lines |

---

## ▶️ Run It

```bash
cd agentic-ai-journey/day-03

# No external dependencies — pure Python stdlib
python day3_agent_architecture.py
```

**Expected Output:**

```
DEMO 1: Research Agent (Inheritance + Composition)
  Setting up tools for Atlas:
    Registered tool: search / summarize / analyze

  Agent: Atlas (Research Specialist)
  Task:  Search and summarize info about LangGraph agents
  [THINK]   Plan: use search -> summarize -> analyze
  [ACT]     {search: Found 5 results, summarize: Summary, analyze: 48 data points}
  [OBSERVE] Completed: search + summarize + analyze

DEMO 2: Observer Pattern (Agent Monitoring)
  --> [Atlas] task_started: Research LangGraph
  [T] [Atlas] tool_called: search('LangGraph')
  [+] [Atlas] task_completed: Report generated
  Metrics: {'task_started': 1, 'tool_called': 2, 'task_completed': 1}

DEMO 3: Chain of Responsibility (Agent Pipeline)
  [CodeBot] Can't handle -> passing to Scholar
  [Scholar] I specialize in research -> Done!
```

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| Python ABC docs | https://docs.python.org/3/library/abc.html |
| Python Dataclasses | https://docs.python.org/3/library/dataclasses.html |
| Design Patterns in Python | https://refactoring.guru/design-patterns/python |
| Strategy Pattern | https://refactoring.guru/design-patterns/strategy |
| Observer Pattern | https://refactoring.guru/design-patterns/observer |
| Chain of Responsibility | https://refactoring.guru/design-patterns/chain-of-responsibility |
| Pydantic V2 Docs | https://docs.pydantic.dev/latest/ |
| LangGraph Agent Architecture | https://langchain-ai.github.io/langgraph/ |

---

## 💡 Why This Matters for Agent Engineering

```
OOP Pattern              →  What it becomes in Agents
─────────────────────────────────────────────────────
Abstract Base Class      →  Agent interface contracts
Inheritance              →  Specialized agent types
Composition              →  Agent = State + Tools + Memory
Strategy Pattern         →  Pluggable tool system
Observer Pattern         →  Agent monitoring (AgentOps)
Chain of Responsibility  →  Multi-agent task pipelines
Dataclasses              →  Typed state schemas
Enums                    →  Agent status machines
```

If you open LangGraph's source code right now, you'll find these exact patterns.

---

## 🗓️ What's Next

**Day 4:** Async Python — Non-Negotiable for Agents  
→ `asyncio`, `async/await`, `asyncio.gather()`, concurrent API calls  
→ Agents make parallel tool calls — async is mandatory

---

*Part of the [150-Day Agentic AI Mastery Roadmap](../README.md)*
