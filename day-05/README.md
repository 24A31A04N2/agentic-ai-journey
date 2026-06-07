# 🤖 Day 05/150 — APIs & HTTP — The Language Agents Speak

![Day](https://img.shields.io/badge/Day-05%2F150-blue?style=flat-square)
![Phase](https://img.shields.io/badge/Phase-1%3A%20Foundations-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11-3572a5?style=flat-square&logo=python)
![Code](https://img.shields.io/badge/Code-Verified-3fb950?style=flat-square)

> **Key Insight:** Agents don't think in Python — they think in API calls.  
> Every tool call is an HTTP request. Every response is JSON to parse. Every failure needs a retry.

---

## 📌 What I Learned Today

| Concept | Agent Application | Why It Matters |
|---------|-------------------|----------------|
| HTTP anatomy (GET/POST) | Every tool call = HTTP request | Foundation of agent communication |
| Exponential backoff | Handle 429 rate limits gracefully | Agents hit rate limits constantly |
| Parallel API calls | `asyncio.gather()` + API client | 5x throughput on multi-tool agents |
| JSON processing | Parse OpenAI tool_calls from nested response | Daily skill for agent engineers |
| API key management | `.env` files, never hardcode secrets | Security baseline |

---

## 🔨 What I Built

### 1. `RetryConfig` + `@with_retry` Decorator
Exponential backoff with jitter — handles 429 and 500 errors automatically:
```
Attempt 0:  0.6s  #
Attempt 1:  1.9s  #####
Attempt 2:  4.7s  ##############
Attempt 3:  9.2s  ###########################
Attempt 4: 22.0s  #################################################################
```

### 2. `AgentAPIClient` — Production HTTP Client
- Auth headers (Bearer token)
- `@with_retry()` decorator on all methods
- `get_many()` for parallel calls via `asyncio.gather()`
- Built-in `APIMetrics` tracking

### 3. OpenAI JSON Response Parser
Extracts `tool_calls` from the real nested OpenAI chat completion response structure.

### 4. `SecureConfig` — API Key Management
Simulates `.env` file loading with masked key display.

---

## 📂 Files

| File | Description |
|------|-------------|
| [`day5_apis_http.py`](./day5_apis_http.py) | Full code — 5 concepts, 5 demos, 350+ lines |

---

## ▶️ Run It

```bash
cd agentic-ai-journey/day-05
python day5_apis_http.py
```

No external dependencies — pure Python stdlib.

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| Python httpx docs | https://www.python-httpx.org/ |
| requests library | https://requests.readthedocs.io/ |
| OpenAI API reference | https://platform.openai.com/docs/api-reference |
| python-dotenv | https://pypi.org/project/python-dotenv/ |
| Exponential backoff explained | https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/ |
| HTTP status codes | https://developer.mozilla.org/en-US/docs/Web/HTTP/Status |

---

## 🗓️ What's Next

**Day 6:** Building APIs with FastAPI  
→ Routes, Pydantic validation, streaming (SSE), WebSockets  
→ This is how agents expose their capabilities as APIs

---

*Part of the [150-Day Agentic AI Mastery Roadmap](../README.md)*
