# 🤖 Day 11/150 — Databases & Data Persistence — Agent Memory That Survives

![Day](https://img.shields.io/badge/Day-11%2F150-blue?style=flat-square)
![Phase](https://img.shields.io/badge/Phase-1%3A%20Foundations-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11-3572a5?style=flat-square&logo=python)
![SQLite](https://img.shields.io/badge/SQLite3-Built--In-003B57?style=flat-square&logo=sqlite)

> **Key Insight:** An agent without persistent memory is like a goldfish —
> it forgets everything between runs. Databases give agents the ability to
> remember conversations, cache tool results, and accumulate knowledge over time.

---

## 📌 What I Learned Today

| Concept | What It Does | Agent Application |
|---------|-------------|-------------------|
| **SQLite3 Context Managers** | Guarantee connections close even on errors | Prevent leaked connections in long-running agents |
| **Schema Design** | Structured tables for different memory types | Conversations, tool cache, and knowledge base tables |
| **CRUD Operations** | Create, Read, Update, Delete data | Save messages, retrieve history, update facts, delete sessions |
| **Transactions** | Atomic commit/rollback for multi-step writes | Protect agent state when multiple inserts must succeed together |
| **Query Builder** | Dynamic, parameterized query construction | Agents build queries based on user intent without SQL injection risk |
| **TTL Caching** | Time-to-live expiration for cached results | Avoid stale web search results while keeping calculator results longer |

---

## 🔨 What I Built

### AgentMemoryStore (`day11_agent_memory_db.py`)
A complete persistent memory system for AI agents with three subsystems:

- **`AgentMemoryStore`**: Single class wrapping all CRUD operations — the agent's "hippocampus"
- **`SchemaManager`**: Idempotent schema creation with indexes for fast lookups
- **`QueryBuilder`**: Dynamic, composable, parameterized query construction
- **`get_db_connection()`**: Context manager ensuring connections always close
- **`get_db_transaction()`**: Context manager with auto-rollback on exceptions

### Three Memory Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `conversations` | Chat history with session grouping | session_id, role, content, timestamp |
| `tool_results` | Cached API/tool call results with TTL | tool_name, input_hash, ttl_seconds |
| `knowledge_base` | Long-term factual memory | topic, content, source, confidence |

---

## 📂 Code Highlights

### Context Manager for Safe Connections
```python
@contextmanager
def get_db_connection(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Access columns by name
    try:
        yield conn
    finally:
        conn.close()  # Always closes, even on exceptions
```

### Transaction with Auto-Rollback
```python
@contextmanager
def get_db_transaction(conn):
    try:
        yield conn
        conn.commit()    # Success -> persist changes
    except Exception:
        conn.rollback()  # Failure -> undo everything
        raise
```

### Query Builder Pattern
```python
builder = QueryBuilder(table="knowledge_base")
builder.where("topic = ?", "python").where("confidence > ?", 0.8)
builder.order_by("confidence", "DESC").limit(5)
sql, params = builder.build()
# -> SELECT * FROM knowledge_base WHERE topic = ? AND confidence > ? ORDER BY ...
# -> params: ["python", 0.8]
```

### TTL-Based Cache Expiration
```python
age_seconds = (now - created).total_seconds()
if age_seconds > result["ttl_seconds"]:
    conn.execute("DELETE FROM tool_results WHERE id = ?", (result["id"],))
    return None  # Force agent to re-fetch
```

---

## ▶️ Run It

```bash
cd agentic-ai-journey/day-11
python day11_agent_memory_db.py
```

---

## 🧠 Why This Matters for Agents

1. **Conversation Memory**: Without saving chat history, an agent can't reference earlier messages in a session.
2. **Tool Caching**: Web searches cost time and money. Caching results with TTL saves both.
3. **Knowledge Base**: Agents can accumulate facts over time — user preferences, project context, learned information.
4. **Transaction Safety**: Multi-step agent workflows (save message + update counter) must either fully succeed or fully roll back.
5. **Dynamic Queries**: Agents filter data based on user requests — a query builder makes this safe and flexible.

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| Python sqlite3 Documentation | https://docs.python.org/3/library/sqlite3.html |
| SQLite Official Site | https://www.sqlite.org/index.html |
| Context Managers (PEP 343) | https://peps.python.org/pep-0343/ |
| Dataclasses (PEP 557) | https://peps.python.org/pep-0557/ |
| SQL Injection Prevention | https://owasp.org/www-community/attacks/SQL_Injection |
