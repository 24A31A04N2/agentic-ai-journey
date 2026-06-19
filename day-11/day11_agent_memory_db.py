"""
🤖 Day 11/150 — Databases & Data Persistence — Agent Memory That Survives
==========================================================================
Phase 1: Python & Engineering Foundations

Concepts:
  1. SQLite3 Connection Management — context managers for safe DB access
  2. Schema Design — conversations, tool_results, knowledge_base tables
  3. CRUD Operations — Create, Read, Update, Delete on agent memory
  4. Transaction Management — commit/rollback for data integrity
  5. Query Builder Pattern — dynamic, composable queries for agents
  6. AgentMemoryStore — full persistent memory class for AI agents

Why this matters for Agentic AI:
  - Agents forget EVERYTHING between runs without persistent storage.
  - Conversation history lets agents maintain context across sessions.
  - Caching tool results avoids redundant API calls (saves cost & latency).
  - A knowledge base gives agents long-term factual memory to reason over.
  - Transaction safety prevents corrupted state in multi-step agent workflows.
"""

import os
import sys
import sqlite3
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timezone
from contextlib import contextmanager


# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ============================================================================
# SECTION 1: Database Connection Manager
# ============================================================================
# Why: Raw sqlite3.connect() calls leak connections if exceptions occur.
# Context managers guarantee connections are closed even on crashes — critical
# for long-running agents that open thousands of DB connections over time.

@contextmanager
def get_db_connection(db_path: str):
    """
    Context manager for safe SQLite connections.

    Usage:
        with get_db_connection("memory.db") as conn:
            conn.execute("SELECT ...")

    Why this pattern matters:
      - Guarantees conn.close() even if an exception fires mid-query.
      - Prevents file-lock issues on the SQLite database file.
      - Agents run for hours — leaked connections accumulate and crash.
    """
    conn = sqlite3.connect(db_path)
    # Return rows as dict-like objects instead of plain tuples.
    # This lets agent code access columns by name: row["content"]
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_db_transaction(conn: sqlite3.Connection):
    """
    Context manager for atomic transactions with auto-rollback.

    Usage:
        with get_db_transaction(conn):
            conn.execute("INSERT ...")
            conn.execute("UPDATE ...")
            # Both succeed, or both roll back.

    Why this pattern matters:
      - An agent might insert a conversation AND update a counter.
      - If the counter update fails, we must undo the insert too.
      - Without transactions, partial writes corrupt agent state.
    """
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ============================================================================
# SECTION 2: Data Models (Dataclasses)
# ============================================================================
# Why: Typed data models prevent agents from passing around raw dicts/tuples.
# Dataclasses enforce structure — if a field is missing, Python raises an error
# at creation time, not deep inside a query handler.

@dataclass
class ConversationMessage:
    """A single message in an agent conversation."""
    role: str              # "user", "assistant", or "system"
    content: str           # The message text
    session_id: str        # Groups messages into conversations
    timestamp: str = ""    # ISO-8601 timestamp, auto-filled if empty
    id: Optional[int] = None  # Database primary key, None before insert

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class ToolResult:
    """Cached result from an external tool/API call."""
    tool_name: str        # e.g. "web_search", "calculator", "code_exec"
    input_hash: str       # Hash of the input for deduplication
    result_data: str      # JSON-serialized result payload
    ttl_seconds: int = 3600  # Time-to-live before cache expires (1 hour)
    created_at: str = ""
    id: Optional[int] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class KnowledgeEntry:
    """A fact or document chunk stored in the agent's knowledge base."""
    topic: str            # Category: "python", "user_preferences", "project_info"
    content: str          # The actual knowledge text
    source: str           # Where this knowledge came from
    confidence: float = 1.0  # 0.0 to 1.0 — how reliable is this fact?
    created_at: str = ""
    updated_at: str = ""
    id: Optional[int] = None

    def __post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


# ============================================================================
# SECTION 3: Schema Manager
# ============================================================================
# Why: Agents need a consistent, versioned schema. If tables don't exist,
# the agent crashes on first query. SchemaManager ensures the DB is always
# ready before any CRUD operations begin.

class SchemaManager:
    """Creates and manages the database schema for agent memory."""

    # SQL statements to create the three core agent memory tables.
    # IF NOT EXISTS makes this idempotent — safe to call on every startup.
    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS conversations (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id  TEXT NOT NULL,
        role        TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
        content     TEXT NOT NULL,
        timestamp   TEXT NOT NULL,
        created_at  TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS tool_results (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        tool_name   TEXT NOT NULL,
        input_hash  TEXT NOT NULL,
        result_data TEXT NOT NULL,
        ttl_seconds INTEGER DEFAULT 3600,
        created_at  TEXT DEFAULT (datetime('now')),
        UNIQUE(tool_name, input_hash)
    );

    CREATE TABLE IF NOT EXISTS knowledge_base (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        topic       TEXT NOT NULL,
        content     TEXT NOT NULL,
        source      TEXT NOT NULL,
        confidence  REAL DEFAULT 1.0 CHECK(confidence >= 0.0 AND confidence <= 1.0),
        created_at  TEXT DEFAULT (datetime('now')),
        updated_at  TEXT DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
    CREATE INDEX IF NOT EXISTS idx_tool_lookup  ON tool_results(tool_name, input_hash);
    CREATE INDEX IF NOT EXISTS idx_kb_topic     ON knowledge_base(topic);
    """

    @staticmethod
    def initialize(conn: sqlite3.Connection) -> None:
        """
        Create all agent memory tables if they don't exist.

        Why indexes?
          - idx_conv_session: Agents retrieve conversations BY session. Without
            an index, SQLite scans every row — O(n) instead of O(log n).
          - idx_tool_lookup: Cache hits must be instant. Composite index on
            (tool_name, input_hash) makes lookups O(1) with hashing.
          - idx_kb_topic: Knowledge retrieval filters by topic first.
        """
        conn.executescript(SchemaManager.SCHEMA_SQL)
        conn.commit()


# ============================================================================
# SECTION 4: Query Builder
# ============================================================================
# Why: Agents build queries dynamically based on user intent. Hardcoding SQL
# strings leads to injection vulnerabilities and rigid code. A query builder
# constructs parameterized queries safely and composably.

@dataclass
class QueryBuilder:
    """
    Builds parameterized SELECT queries dynamically.

    Why this pattern matters for agents:
      - An agent might filter by topic, confidence, date range, or any combo.
      - The builder lets you chain .where(), .order_by(), .limit() calls.
      - All values are parameterized (?) to prevent SQL injection attacks.
    """
    table: str
    _columns: str = "*"
    _conditions: List[str] = field(default_factory=list)
    _params: List[Any] = field(default_factory=list)
    _order: Optional[str] = None
    _limit: Optional[int] = None

    def select(self, columns: str = "*") -> "QueryBuilder":
        """Choose which columns to return."""
        self._columns = columns
        return self

    def where(self, condition: str, value: Any) -> "QueryBuilder":
        """
        Add a WHERE clause with a parameterized value.

        Example:
            builder.where("topic = ?", "python")
            builder.where("confidence > ?", 0.8)
        """
        self._conditions.append(condition)
        self._params.append(value)
        return self

    def order_by(self, column: str, direction: str = "ASC") -> "QueryBuilder":
        """Set the ORDER BY clause."""
        self._order = f"{column} {direction}"
        return self

    def limit(self, count: int) -> "QueryBuilder":
        """Limit the number of returned rows."""
        self._limit = count
        return self

    def build(self) -> Tuple[str, List[Any]]:
        """
        Compile the query into a (sql_string, params) tuple.

        Returns:
            Tuple of (SQL query string, list of parameter values)
        """
        sql = f"SELECT {self._columns} FROM {self.table}"

        if self._conditions:
            sql += " WHERE " + " AND ".join(self._conditions)

        if self._order:
            sql += f" ORDER BY {self._order}"

        if self._limit is not None:
            sql += f" LIMIT {self._limit}"

        return sql, self._params


# ============================================================================
# SECTION 5: AgentMemoryStore — The Core Persistent Memory Class
# ============================================================================
# Why: This is the single class an agent imports to remember everything.
# It wraps all CRUD operations behind clean methods so the agent never
# writes raw SQL. Think of it as the agent's "hippocampus".

class AgentMemoryStore:
    """
    Persistent memory store for AI agents using SQLite.

    Provides three memory subsystems:
      1. Conversations — chat history with session grouping
      2. Tool Results  — cached API/tool call results with TTL
      3. Knowledge Base — long-term factual memory with confidence scores

    Usage:
        store = AgentMemoryStore("agent_memory.db")
        store.save_message(ConversationMessage(...))
        history = store.get_conversation("session-123")
    """

    def __init__(self, db_path: str = ":memory:"):
        """
        Initialize the memory store.

        Args:
            db_path: Path to SQLite file, or ':memory:' for in-memory DB.
                     In-memory is great for testing; file-based for production.
        """
        self.db_path = db_path
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Create tables on first use. Idempotent — safe to call repeatedly."""
        with get_db_connection(self.db_path) as conn:
            SchemaManager.initialize(conn)

    # ----------------------------------------------------------------
    # CRUD: Conversations
    # ----------------------------------------------------------------

    def save_message(self, message: ConversationMessage) -> int:
        """
        Save a conversation message to the database.

        Returns:
            The row ID of the inserted message.

        Why agents need this:
          - Without saving messages, the agent has amnesia between API calls.
          - Session grouping lets the agent recall the RIGHT conversation.
        """
        with get_db_connection(self.db_path) as conn:
            with get_db_transaction(conn):
                cursor = conn.execute(
                    """INSERT INTO conversations (session_id, role, content, timestamp)
                       VALUES (?, ?, ?, ?)""",
                    (message.session_id, message.role, message.content, message.timestamp)
                )
                return cursor.lastrowid

    def get_conversation(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history for a session, ordered chronologically.

        Args:
            session_id: The conversation session to retrieve.
            limit: Max messages to return (prevents loading 10k messages).

        Returns:
            List of message dicts with keys: id, role, content, timestamp.
        """
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(
                """SELECT id, session_id, role, content, timestamp
                   FROM conversations
                   WHERE session_id = ?
                   ORDER BY timestamp ASC
                   LIMIT ?""",
                (session_id, limit)
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_conversation(self, session_id: str) -> int:
        """
        Delete all messages for a session.

        Returns:
            Number of messages deleted.

        Why agents need this:
          - Privacy: users may request conversation deletion.
          - Memory management: old sessions consume disk space.
        """
        with get_db_connection(self.db_path) as conn:
            with get_db_transaction(conn):
                cursor = conn.execute(
                    "DELETE FROM conversations WHERE session_id = ?",
                    (session_id,)
                )
                return cursor.rowcount

    # ----------------------------------------------------------------
    # CRUD: Tool Results Cache
    # ----------------------------------------------------------------

    def cache_tool_result(self, result: ToolResult) -> int:
        """
        Cache a tool's result for later retrieval.

        Uses INSERT OR REPLACE so re-running the same tool with the same
        input overwrites the stale cache entry automatically.

        Returns:
            The row ID of the cached result.
        """
        with get_db_connection(self.db_path) as conn:
            with get_db_transaction(conn):
                cursor = conn.execute(
                    """INSERT OR REPLACE INTO tool_results
                       (tool_name, input_hash, result_data, ttl_seconds, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (result.tool_name, result.input_hash,
                     result.result_data, result.ttl_seconds, result.created_at)
                )
                return cursor.lastrowid

    def get_cached_result(self, tool_name: str, input_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached tool result if it exists and hasn't expired.

        Returns:
            The cached result dict, or None if not found / expired.

        Why TTL matters for agents:
          - Web search results from 2 hours ago may be outdated.
          - Calculator results never expire. TTL is tool-specific.
          - Expired entries return None, forcing the agent to re-call the tool.
        """
        with get_db_connection(self.db_path) as conn:
            row = conn.execute(
                """SELECT id, tool_name, input_hash, result_data, ttl_seconds, created_at
                   FROM tool_results
                   WHERE tool_name = ? AND input_hash = ?""",
                (tool_name, input_hash)
            ).fetchone()

            if row is None:
                return None

            result = dict(row)

            # Check TTL expiration
            created = datetime.fromisoformat(result["created_at"])
            now = datetime.now(timezone.utc)
            # Handle naive datetimes from SQLite
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_seconds = (now - created).total_seconds()

            if age_seconds > result["ttl_seconds"]:
                # Cache expired — delete the stale entry
                conn.execute("DELETE FROM tool_results WHERE id = ?", (result["id"],))
                conn.commit()
                return None

            return result

    def clear_tool_cache(self, tool_name: Optional[str] = None) -> int:
        """
        Clear cached tool results. Optionally filter by tool name.

        Returns:
            Number of cache entries deleted.
        """
        with get_db_connection(self.db_path) as conn:
            with get_db_transaction(conn):
                if tool_name:
                    cursor = conn.execute(
                        "DELETE FROM tool_results WHERE tool_name = ?",
                        (tool_name,)
                    )
                else:
                    cursor = conn.execute("DELETE FROM tool_results")
                return cursor.rowcount

    # ----------------------------------------------------------------
    # CRUD: Knowledge Base
    # ----------------------------------------------------------------

    def add_knowledge(self, entry: KnowledgeEntry) -> int:
        """
        Add a knowledge entry to the agent's long-term memory.

        Returns:
            The row ID of the inserted entry.

        Why agents need a knowledge base:
          - LLM training data has a cutoff date. New facts go here.
          - User preferences ("I prefer Python over JS") persist here.
          - Project context ("the main branch is 'prod'") lives here.
        """
        with get_db_connection(self.db_path) as conn:
            with get_db_transaction(conn):
                cursor = conn.execute(
                    """INSERT INTO knowledge_base
                       (topic, content, source, confidence, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (entry.topic, entry.content, entry.source,
                     entry.confidence, entry.created_at, entry.updated_at)
                )
                return cursor.lastrowid

    def search_knowledge(self, topic: Optional[str] = None,
                         min_confidence: float = 0.0,
                         limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search the knowledge base with optional filters.

        Uses the QueryBuilder for dynamic, safe query construction.

        Args:
            topic: Filter by topic category (None = all topics).
            min_confidence: Minimum confidence threshold.
            limit: Maximum results to return.

        Returns:
            List of knowledge entry dicts.
        """
        builder = QueryBuilder(table="knowledge_base")
        builder.select("id, topic, content, source, confidence, created_at, updated_at")

        if topic:
            builder.where("topic = ?", topic)
        if min_confidence > 0.0:
            builder.where("confidence >= ?", min_confidence)

        builder.order_by("confidence", "DESC").limit(limit)

        sql, params = builder.build()

        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]

    def update_knowledge(self, entry_id: int, content: Optional[str] = None,
                         confidence: Optional[float] = None) -> bool:
        """
        Update an existing knowledge entry.

        Args:
            entry_id: The database ID of the entry to update.
            content: New content text (None = keep current).
            confidence: New confidence score (None = keep current).

        Returns:
            True if the entry was found and updated.

        Why updates matter:
          - An agent learns "Python 3.12 is latest" today, but tomorrow
            Python 3.13 releases. The agent updates, not duplicates.
        """
        updates = []
        params = []

        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(datetime.now(timezone.utc).isoformat())
        params.append(entry_id)

        sql = f"UPDATE knowledge_base SET {', '.join(updates)} WHERE id = ?"

        with get_db_connection(self.db_path) as conn:
            with get_db_transaction(conn):
                cursor = conn.execute(sql, params)
                return cursor.rowcount > 0

    def delete_knowledge(self, entry_id: int) -> bool:
        """
        Delete a knowledge entry by ID.

        Returns:
            True if the entry was found and deleted.
        """
        with get_db_connection(self.db_path) as conn:
            with get_db_transaction(conn):
                cursor = conn.execute(
                    "DELETE FROM knowledge_base WHERE id = ?",
                    (entry_id,)
                )
                return cursor.rowcount > 0

    # ----------------------------------------------------------------
    # Utility Methods
    # ----------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """
        Get memory store statistics.

        Returns:
            Dict with counts: conversations, tool_cache, knowledge_entries.
        """
        with get_db_connection(self.db_path) as conn:
            conv_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            tool_count = conn.execute("SELECT COUNT(*) FROM tool_results").fetchone()[0]
            kb_count = conn.execute("SELECT COUNT(*) FROM knowledge_base").fetchone()[0]

            return {
                "conversations": conv_count,
                "tool_cache_entries": tool_count,
                "knowledge_entries": kb_count,
            }

    def search_conversations_by_keyword(self, keyword: str,
                                        limit: int = 10) -> List[Dict[str, Any]]:
        """
        Full-text search across conversation content.

        Uses SQL LIKE for simplicity. Production agents would use
        SQLite FTS5 (Full-Text Search) or a vector database for
        semantic similarity search.
        """
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(
                """SELECT id, session_id, role, content, timestamp
                   FROM conversations
                   WHERE content LIKE ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (f"%{keyword}%", limit)
            ).fetchall()
            return [dict(row) for row in rows]


# ============================================================================
# SECTION 6: Transaction Safety Demonstration
# ============================================================================
# Why: This function shows what happens when a multi-step write fails
# midway. Without transactions, the DB ends up in an inconsistent state.

def demonstrate_transaction_safety(store: AgentMemoryStore) -> None:
    """
    Shows transaction rollback protecting data integrity.

    Scenario: Agent tries to save two related knowledge entries atomically.
    The second insert intentionally fails (confidence > 1.0 violates CHECK).
    Transaction rollback ensures NEITHER entry persists.
    """
    print("\n" + "=" * 70)
    print("  DEMO: Transaction Safety (Rollback on Failure)")
    print("=" * 70)

    with get_db_connection(store.db_path) as conn:
        before_count = conn.execute("SELECT COUNT(*) FROM knowledge_base").fetchone()[0]
        print(f"  Knowledge entries BEFORE transaction: {before_count}")

    # Attempt an atomic two-insert transaction where the second fails
    try:
        with get_db_connection(store.db_path) as conn:
            with get_db_transaction(conn):
                # First insert — valid
                conn.execute(
                    """INSERT INTO knowledge_base (topic, content, source, confidence)
                       VALUES (?, ?, ?, ?)""",
                    ("test", "This should NOT persist", "transaction_test", 0.9)
                )
                print("  [1/2] First INSERT executed (valid)...")

                # Second insert — INVALID confidence > 1.0 violates CHECK constraint
                conn.execute(
                    """INSERT INTO knowledge_base (topic, content, source, confidence)
                       VALUES (?, ?, ?, ?)""",
                    ("test", "This will fail", "transaction_test", 1.5)
                )
    except Exception as e:
        print(f"  [2/2] Second INSERT failed: {type(e).__name__}: {e}")
        print("  --> Transaction ROLLED BACK. Neither insert persisted.")

    with get_db_connection(store.db_path) as conn:
        after_count = conn.execute("SELECT COUNT(*) FROM knowledge_base").fetchone()[0]
        print(f"  Knowledge entries AFTER rollback:     {after_count}")

    if before_count == after_count:
        print("  [OK] Rollback confirmed! Data integrity preserved.")
    else:
        print("  [WARN] Counts differ — rollback may not have worked.")

    # Store before_count for verification in the outer scope
    return None


# ============================================================================
# RUNNER: Full Demo of Agent Memory Database
# ============================================================================

def run_memory_demo():
    """Runs the complete agent memory database demonstration."""
    print("=" * 70)
    print("  Day 11/150 -- Databases & Data Persistence")
    print("  Agent Memory That Survives")
    print("=" * 70)

    # Use in-memory DB for demo (no file cleanup needed)
    db_path = ":memory:"
    store = AgentMemoryStore(db_path)
    print(f"\n[+] AgentMemoryStore initialized (db={db_path})")

    # ------------------------------------------------------------------
    # DEMO 1: Conversation Memory
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DEMO 1: Conversation Memory (CRUD)")
    print("=" * 70)

    session = "session-abc-001"

    # CREATE — Save messages
    messages = [
        ConversationMessage(role="system", content="You are a helpful coding assistant.", session_id=session),
        ConversationMessage(role="user", content="How do I connect to SQLite in Python?", session_id=session),
        ConversationMessage(role="assistant", content="Use sqlite3.connect('mydb.db') with a context manager for safety.", session_id=session),
        ConversationMessage(role="user", content="What about parameterized queries?", session_id=session),
        ConversationMessage(role="assistant", content="Use ? placeholders: cursor.execute('SELECT * FROM t WHERE id=?', (val,))", session_id=session),
    ]

    print(f"\n  Saving {len(messages)} messages to session '{session}'...")
    for msg in messages:
        row_id = store.save_message(msg)
        print(f"    [+] Saved [{msg.role:>9}] id={row_id}: {msg.content[:50]}...")

    # READ — Retrieve conversation
    print(f"\n  Retrieving conversation for session '{session}':")
    history = store.get_conversation(session)
    for msg in history:
        print(f"    [{msg['role']:>9}] {msg['content'][:60]}")

    # Save a second session to show isolation
    session2 = "session-xyz-002"
    store.save_message(ConversationMessage(role="user", content="Tell me about async Python.", session_id=session2))
    store.save_message(ConversationMessage(role="assistant", content="Use asyncio with async/await for concurrent I/O.", session_id=session2))

    print(f"\n  Session '{session}' has {len(store.get_conversation(session))} messages")
    print(f"  Session '{session2}' has {len(store.get_conversation(session2))} messages")
    print("  --> Sessions are isolated! Each agent conversation stays separate.")

    # DELETE — Remove a conversation
    deleted = store.delete_conversation(session2)
    print(f"\n  Deleted session '{session2}': {deleted} messages removed")
    remaining = store.get_conversation(session2)
    print(f"  Messages remaining in '{session2}': {len(remaining)}")

    # ------------------------------------------------------------------
    # DEMO 2: Tool Results Cache
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DEMO 2: Tool Results Cache")
    print("=" * 70)

    # CREATE — Cache some tool results
    tool_results = [
        ToolResult(
            tool_name="web_search",
            input_hash="hash_python_sqlite",
            result_data=json.dumps({"title": "SQLite Python Tutorial", "url": "https://docs.python.org/3/library/sqlite3.html"}),
            ttl_seconds=3600
        ),
        ToolResult(
            tool_name="calculator",
            input_hash="hash_2plus2",
            result_data=json.dumps({"expression": "2+2", "result": 4}),
            ttl_seconds=86400  # Calculator results last 24 hours
        ),
        ToolResult(
            tool_name="web_search",
            input_hash="hash_weather_today",
            result_data=json.dumps({"weather": "Sunny, 28C"}),
            ttl_seconds=1  # Expires almost immediately for demo
        ),
    ]

    print("\n  Caching tool results...")
    for tr in tool_results:
        row_id = store.cache_tool_result(tr)
        print(f"    [+] Cached '{tr.tool_name}' (hash={tr.input_hash}) ttl={tr.ttl_seconds}s -> id={row_id}")

    # READ — Cache hit
    cached = store.get_cached_result("calculator", "hash_2plus2")
    if cached:
        data = json.loads(cached["result_data"])
        print(f"\n  Cache HIT: calculator -> {data}")
    else:
        print("\n  Cache MISS: calculator")

    # READ — Cache miss (expired TTL)
    print("\n  Waiting 2 seconds for weather cache to expire...")
    time.sleep(2)
    expired = store.get_cached_result("web_search", "hash_weather_today")
    if expired:
        print(f"  Cache HIT: weather -> {expired['result_data']}")
    else:
        print("  Cache MISS: weather result expired (TTL=1s). Agent must re-fetch!")

    # ------------------------------------------------------------------
    # DEMO 3: Knowledge Base
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DEMO 3: Knowledge Base (Long-Term Memory)")
    print("=" * 70)

    # CREATE — Add knowledge entries
    knowledge_items = [
        KnowledgeEntry(topic="python", content="Python 3.11 introduced ExceptionGroup for structured concurrency.", source="docs.python.org", confidence=0.95),
        KnowledgeEntry(topic="python", content="Use 'match' statements (structural pattern matching) from Python 3.10+.", source="PEP 634", confidence=0.90),
        KnowledgeEntry(topic="databases", content="SQLite supports up to 281 TB databases and is ACID-compliant.", source="sqlite.org", confidence=1.0),
        KnowledgeEntry(topic="user_preferences", content="User prefers dark mode and compact output format.", source="user_settings", confidence=0.85),
        KnowledgeEntry(topic="agents", content="ReAct agents alternate between Reasoning and Acting steps.", source="Yao et al., 2022", confidence=0.92),
        KnowledgeEntry(topic="agents", content="Tool-use agents need structured output to call external APIs.", source="OpenAI docs", confidence=0.88),
    ]

    print("\n  Adding knowledge entries...")
    for entry in knowledge_items:
        row_id = store.add_knowledge(entry)
        print(f"    [+] id={row_id} [{entry.topic}] confidence={entry.confidence:.2f}: {entry.content[:55]}...")

    # READ — Search by topic
    print("\n  Searching knowledge base for topic='python':")
    python_facts = store.search_knowledge(topic="python")
    for fact in python_facts:
        print(f"    [{fact['confidence']:.2f}] {fact['content']}")

    # READ — Search with confidence filter
    print("\n  Searching knowledge base for confidence >= 0.92:")
    high_conf = store.search_knowledge(min_confidence=0.92)
    for fact in high_conf:
        print(f"    [{fact['confidence']:.2f}] [{fact['topic']}] {fact['content'][:55]}...")

    # UPDATE — Update a knowledge entry
    print("\n  Updating knowledge entry id=1 with new content...")
    updated = store.update_knowledge(
        entry_id=1,
        content="Python 3.12 introduced improved error messages and f-string enhancements.",
        confidence=0.98
    )
    print(f"    Update successful: {updated}")

    # Verify the update
    updated_entries = store.search_knowledge(topic="python")
    for fact in updated_entries:
        print(f"    [{fact['confidence']:.2f}] {fact['content']}")

    # DELETE — Remove a knowledge entry
    print("\n  Deleting knowledge entry id=4 (user_preferences)...")
    deleted = store.delete_knowledge(entry_id=4)
    print(f"    Deletion successful: {deleted}")

    # ------------------------------------------------------------------
    # DEMO 4: Query Builder
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DEMO 4: Query Builder Pattern")
    print("=" * 70)

    # Show the QueryBuilder constructing different queries
    queries = [
        ("All knowledge", QueryBuilder(table="knowledge_base").select("topic, content, confidence")),
        ("High-confidence agents", QueryBuilder(table="knowledge_base")
            .select("topic, content, confidence")
            .where("topic = ?", "agents")
            .where("confidence > ?", 0.85)
            .order_by("confidence", "DESC")),
        ("Latest 3 entries", QueryBuilder(table="knowledge_base")
            .select("id, topic, content")
            .order_by("id", "DESC")
            .limit(3)),
    ]

    for label, builder in queries:
        sql, params = builder.build()
        print(f"\n  Query: {label}")
        print(f"    SQL:    {sql}")
        print(f"    Params: {params}")

        with get_db_connection(store.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
            print(f"    Results: {len(rows)} rows")
            for row in rows:
                print(f"      -> {dict(row)}")

    # ------------------------------------------------------------------
    # DEMO 5: Keyword Search
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DEMO 5: Conversation Keyword Search")
    print("=" * 70)

    results = store.search_conversations_by_keyword("parameterized")
    print(f"\n  Search for 'parameterized' found {len(results)} message(s):")
    for r in results:
        print(f"    [{r['role']}] {r['content'][:70]}...")

    results2 = store.search_conversations_by_keyword("SQLite")
    print(f"\n  Search for 'SQLite' found {len(results2)} message(s):")
    for r in results2:
        print(f"    [{r['role']}] {r['content'][:70]}...")

    # ------------------------------------------------------------------
    # DEMO 6: Transaction Safety
    # ------------------------------------------------------------------
    demonstrate_transaction_safety(store)

    # ------------------------------------------------------------------
    # DEMO 7: Memory Statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DEMO 7: Memory Store Statistics")
    print("=" * 70)

    stats = store.get_stats()
    print(f"\n  Final Memory Store Stats:")
    print(f"    Conversation messages:  {stats['conversations']}")
    print(f"    Tool cache entries:     {stats['tool_cache_entries']}")
    print(f"    Knowledge base entries: {stats['knowledge_entries']}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Day 11 Complete!")
    print("=" * 70)
    print("""
  Key Takeaways:
    1. Context managers guarantee DB connections are always closed.
    2. Transactions (commit/rollback) protect multi-step writes.
    3. Parameterized queries prevent SQL injection attacks.
    4. TTL-based caching avoids stale tool results.
    5. A QueryBuilder makes dynamic queries safe and composable.
    6. AgentMemoryStore gives agents persistent memory across restarts.

  Next: Day 12 will build on this foundation with more advanced patterns!
""")


if __name__ == "__main__":
    run_memory_demo()
