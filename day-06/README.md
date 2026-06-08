# 🤖 Day 06/150 — Building APIs with FastAPI

![Day](https://img.shields.io/badge/Day-06%2F150-blue?style=flat-square)
![Phase](https://img.shields.io/badge/Phase-1%3A%20Foundations-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11-3572a5?style=flat-square&logo=python)
![Code](https://img.shields.io/badge/Code-Verified-3fb950?style=flat-square)

> **Key Insight:** Agents don't just call APIs — they BECOME APIs. 
> To integrate AI agents into chat interfaces, frontends, or multi-agent orchestrators, they must expose their capabilities over robust, scalable APIs. 

---

## 📌 What I Learned Today

| Concept | Agent Application | Why It Matters |
|---------|-------------------|----------------|
| **FastAPI Basics** | Core API structure & Routing | Fast, modern web framework with automatic docs |
| **Pydantic Validation** | Type safety & request parsing | Ensures the agent receives correctly formatted parameters |
| **Dependency Injection** | Injected orchestrators & DBs | Decouples agent business logic from HTTP routing |
| **Streaming Responses (SSE)** | Token and step-by-step streaming | Crucial for showing live reasoning progress, avoiding timeouts |
| **WebSockets** | Bi-directional, real-time chat | Long-lived interactive agent chat sessions |

---

## 🔨 What I Built

### 1. `AgentOrchestrator` Service
Simulates a multi-step reasoning agent workflow. It has two core functions:
- A synchronous execution model (`run_reasoning_loop`).
- An asynchronous streaming generator (`run_streaming_reasoning`) yielding Server-Sent Events (SSE).

### 2. FastAPI Gateway Router (`main.py`)
- **POST `/api/query`**: Uses Pydantic request models (`QueryInput`) with validation rules (e.g. prompt length check, max steps limit) and return schema validation (`QueryResponse`).
- **GET `/api/stream`**: Exposes the SSE endpoint returning a `StreamingResponse` so clients can read the agent's progress live.
- **WebSocket `/ws/chat/{client_id}`**: Manages live client connections, broadcasts events, and enables real-time interaction with the agent.

### 3. Async Python Client (`client.py`)
A comprehensive client showcasing how to programmatically connect to, read from, and communicate with the FastAPI server using `httpx` and `websockets`.

---

## 📂 Files

| File | Description |
|------|-------------|
| [`main.py`](./main.py) | FastAPI app — validation schemas, dependencies, REST, SSE, and WebSocket endpoints |
| [`client.py`](./client.py) | Client tester script simulating async REST, SSE streams, and WebSocket chat loops |

---

## ▶️ How to Run

### 1. Install Dependencies
Make sure you have `fastapi`, `uvicorn`, `httpx`, and `websockets` installed:
```bash
pip install fastapi uvicorn httpx websockets
```

### 2. Start the Backend API Server
Run the FastAPI application locally using uvicorn:
```bash
uvicorn main:app --reload --port 8000
```

### 3. Run the Async Client
In a separate terminal window, execute the test client:
```bash
python client.py
```

---

## 💡 Key Architectural Takeaways

1. **Why Streaming (SSE) is Mandatory for LLMs:**
   Large Language Models are slow. Waiting for an agent to finish a 10-step loop synchronously before returning a response can take 10-20 seconds, leading to bad user experience or network gateway timeouts. **Server-Sent Events (SSE)** stream chunks of data as soon as they are ready, making the interface feel immediate and responsive.
   
2. **REST vs. WebSockets for Agents:**
   - **REST (POST/GET)**: Perfect for discrete tool executions, one-off planning requests, or starting an asynchronous background run.
   - **WebSockets**: Crucial for conversational agents that maintain state and require low-latency bi-directional speech/chat feeds.

---

## 🗓️ What's Next

**Day 7:** Practice Day — Week 1 Integration Project  
→ We will build a **"Smart Research Assistant"** command-line tool that combines Async execution, multiple API fetches with retry handling, and structured data outputs using Pydantic!

---

*Part of the [150-Day Agentic AI Mastery Roadmap](../README.md)*
