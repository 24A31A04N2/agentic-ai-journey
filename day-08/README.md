# 🤖 Day 08/150 — Error Handling & Logging for Production

![Day](https://img.shields.io/badge/Day-08%2F150-blue?style=flat-square)
![Phase](https://img.shields.io/badge/Phase-2%3A%20Advanced%20Python-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11-3572a5?style=flat-square&logo=python)
![Code](https://img.shields.io/badge/Code-Verified-3fb950?style=flat-square)

> **Key Insight:** Agents fail constantly — API timeouts, rate limits, malformed LLM output.  
> Without structured error handling, failures are silent. Without logging, you're flying blind.

---

## 📌 What I Learned Today

| Concept | Agent Application | Why It Matters |
|---------|-------------------|----------------|
| **Custom Exception Hierarchies** | `AgentError → LLMError → RateLimitError` | Catch specific failures, not generic `Exception` |
| **Structured Logging** | JSON logs + colored console output | Machine-readable for prod, human-readable for dev |
| **Context Managers** | `with managed_api_session()` | Guaranteed cleanup even on crashes |
| **Retry Engine** | Exponential/Linear/Fixed backoff | Production agents self-heal from transient failures |

---

## 🔨 What I Built

### 1. Custom Exception Hierarchy (5 Classes)
```
AgentError (base)
├── LLMError (model + status_code)
│   └── RateLimitError (retry_after)
├── ToolExecutionError (tool_name + input_data)
├── OutputParsingError (expected vs raw)
└── MaxRetriesExceededError (operation + attempts)
```
Every error carries: `error_code`, `context dict`, `timestamp`, and a `to_dict()` method for structured logging.

### 2. Dual-Mode Logging System
- **`PrettyLogFormatter`**: Colored console output with emoji level icons for development
- **`AgentLogFormatter`**: JSON structured logs for production monitoring (ELK, Datadog, etc.)

### 3. Context Managers (Sync + Async)
- **`managed_api_session()`**: Guarantees API session cleanup, tracks metrics
- **`managed_agent_execution()`**: Wraps entire agent runs with lifecycle logging

### 4. Retry Engine (3 Strategies)
- **FIXED**: Constant delay between retries
- **LINEAR**: Linearly increasing delays
- **EXPONENTIAL**: Exponential backoff + jitter (prevents thundering herd)

---

## 📂 Files

| File | Description |
|------|-------------|
| [`day8_error_handling.py`](./day8_error_handling.py) | Full code — 5 exception classes, logging system, context managers, retry engine |

---

## ▶️ Run It

```bash
cd agentic-ai-journey/day-08
python day8_error_handling.py
```

No external dependencies — pure Python stdlib.

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| Python logging docs | https://docs.python.org/3/library/logging.html |
| contextlib docs | https://docs.python.org/3/library/contextlib.html |
| Tenacity library | https://tenacity.readthedocs.io/ |
| Exception hierarchy best practices | https://docs.python.org/3/tutorial/errors.html |

---

## 🗓️ What's Next

**Day 9:** Testing — How Professionals Ship Code  
→ pytest, mocking API calls, testing async code

---

*Part of the [150-Day Agentic AI Mastery Roadmap](../README.md)*
