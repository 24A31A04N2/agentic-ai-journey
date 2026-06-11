# 🤖 Day 07/150 — Smart Research Assistant CLI

![Day](https://img.shields.io/badge/Day-07%2F150-blue?style=flat-square)
![Phase](https://img.shields.io/badge/Phase-1%3A%20Foundations-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11-3572a5?style=flat-square&logo=python)
![Type](https://img.shields.io/badge/Type-Practice%20Day-ff6b6b?style=flat-square)
![Code](https://img.shields.io/badge/Code-Verified-3fb950?style=flat-square)

> **Week 1 Integration Project** — This is NOT a new concept day.  
> This is where we PROVE we can combine everything from Days 1–6 into one production-quality tool.

---

## 📌 What This Project Integrates

| Day | Concept Used | Where in Code |
|-----|-------------|---------------|
| **Day 1** | Type hints, f-strings, modern Python | Throughout — every function is fully typed |
| **Day 2** | Data structures (dicts, lists, hash maps) | Result aggregation, latency bar charts |
| **Day 3** | OOP — Strategy Pattern | `DataSourceStrategy` ABC → `WikipediaSource`, `ArxivSource`, `NewsAPISource` |
| **Day 3** | OOP — Observer Pattern | `EventBus` + `ConsoleLogger` + `MetricsCollector` |
| **Day 4** | Async parallel execution | `asyncio.gather()` launching 3 API fetches simultaneously |
| **Day 5** | Exponential backoff + retry | `fetch_with_retry()` with `RetryConfig` dataclass |
| **Day 5** | HTTP status codes & error handling | Simulated 429, 401, timeout failures |
| **Day 6** | Pydantic-style validation | `ResearchQuery` with field constraints, `SourceResult`, `ResearchReport` |

---

## 🔨 What It Does

```
┌─────────────────────────────────────────────┐
│          SMART RESEARCH ASSISTANT            │
│          Week 1 Integration Project          │
├─────────────────────────────────────────────┤
│                                             │
│  Input: "Agentic AI"                        │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Wikipedia │  │  arXiv   │  │ NewsAPI  │  │
│  │   API     │  │   API    │  │   API    │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       │              │              │        │
│       └──────┬───────┴──────┬───────┘        │
│              │ asyncio.gather()│              │
│              ▼               ▼               │
│       ┌──────────────────────┐               │
│       │  Retry + Backoff     │               │
│       │  Validation Layer    │               │
│       │  Event Bus Logging   │               │
│       └──────────┬───────────┘               │
│                  ▼                           │
│       ┌──────────────────────┐               │
│       │  Formatted Report    │               │
│       │  with Bar Charts     │               │
│       └──────────────────────┘               │
└─────────────────────────────────────────────┘
```

---

## 📂 Files

| File | Description |
|------|-------------|
| [`smart_research_assistant.py`](./smart_research_assistant.py) | Full CLI tool — 400+ lines, 7 sections, all Week 1 concepts |

---

## ▶️ How to Run

```bash
# Default topic (Agentic AI)
cd agentic-ai-journey/day-07
python smart_research_assistant.py

# Custom topic
python smart_research_assistant.py "Large Language Models"
python smart_research_assistant.py "Retrieval Augmented Generation"
```

No external dependencies — pure Python stdlib.

---

## 📊 Sample Output

```
================================================================
  SMART RESEARCH ASSISTANT — REPORT
================================================================
  Topic:     Agentic AI
  Query ID:  a3f2c1e8
  Depth:     standard
----------------------------------------------------------------
  Total Time:        892ms
  Sources OK:        3/3
  Sources Failed:    0/3
  Avg Relevance:     87%
----------------------------------------------------------------
  LATENCY COMPARISON:
    Wikipedia       523ms  ######################## [OK]
    arXiv           741ms  ############################## [OK]
    NewsAPI         312ms  ############## [OK]
----------------------------------------------------------------
```

---

## 🗓️ What's Next

**Day 8:** Error Handling & Logging for Production  
→ Custom exception hierarchies, structured logging, context managers, tenacity retry library

---

*Part of the [150-Day Agentic AI Mastery Roadmap](../README.md)*
