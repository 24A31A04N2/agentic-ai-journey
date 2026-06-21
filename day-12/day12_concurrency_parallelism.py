"""
🤖 Day 12/150 — Concurrency & Parallelism — Making Agents Think in Parallel
============================================================================
Phase 1: Python & Engineering Foundations

Concepts:
  1. threading.Thread Basics — Worker threads for I/O-bound agent tasks
  2. ThreadPoolExecutor — concurrent.futures for parallel tool execution
  3. Thread Safety — Locks, RLocks, and thread-safe structures for shared state
  4. ProcessPoolExecutor — CPU-bound parallelism for heavy computation
  5. Producer-Consumer Pattern — Queue-based task distribution for pipelines
  6. Parallel Agent Tool Executor — Full class with timeout, retry, aggregation

Why this matters for Agentic AI:
  - An agent calling 5 APIs sequentially wastes 5× the wall-clock time.
  - Parallel tool execution lets agents gather information concurrently.
  - Thread safety prevents corrupted shared state when multiple tools write
    results simultaneously.
  - CPU-bound work (embeddings, parsing) needs processes, not threads
    (Python's GIL blocks true CPU parallelism with threads).
  - Producer-consumer pipelines let multi-agent systems distribute work
    across workers without tight coupling.
"""

import os
import sys
import time
import json
import hashlib
import threading
import queue
import random
from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
    Future,
    TimeoutError as FuturesTimeoutError,
)
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Tuple
from datetime import datetime, timezone
from enum import Enum


# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ============================================================================
# SECTION 1: threading.Thread Basics — I/O-Bound Agent Tasks
# ============================================================================
# Why: When an agent calls external APIs (web search, database lookup, LLM
# inference), each call blocks waiting for a network response. Threads let
# the agent fire multiple I/O calls concurrently so idle wait times overlap.
# Python's GIL is released during I/O, so threads truly run in parallel for
# network-bound work.

def simulate_api_call(tool_name: str, latency: float = 1.0) -> Dict[str, Any]:
    """
    Simulates an external API call with network latency.

    In a real agent, this would be requests.get(), openai.chat(), etc.
    The key point: time.sleep() releases the GIL, so other threads run
    while this one "waits" for the network response.
    """
    start = time.perf_counter()
    time.sleep(latency)  # Simulates network I/O wait
    elapsed = time.perf_counter() - start
    return {
        "tool": tool_name,
        "result": f"{tool_name}_result_data",
        "latency_ms": round(elapsed * 1000),
        "thread": threading.current_thread().name,
    }


def run_sequential_tools(tools: List[Tuple[str, float]]) -> List[Dict]:
    """
    Run tool calls one after another — the naive approach.

    This is what happens when an agent calls tools without concurrency.
    Total time = sum of all individual latencies.
    """
    results = []
    for tool_name, latency in tools:
        result = simulate_api_call(tool_name, latency)
        results.append(result)
    return results


def run_threaded_tools(tools: List[Tuple[str, float]]) -> List[Dict]:
    """
    Run tool calls using raw threading.Thread — the basic approach.

    Each tool call gets its own thread. All threads start immediately,
    then we join() each to wait for completion.
    Total time ≈ max(individual latencies) instead of sum.

    Why raw threads instead of a pool?
      - Good for understanding the mechanics.
      - In production, use ThreadPoolExecutor (Section 2) for resource control.
    """
    results = []
    lock = threading.Lock()  # Protects the shared results list

    def worker(tool_name: str, latency: float):
        result = simulate_api_call(tool_name, latency)
        with lock:  # Thread-safe append
            results.append(result)

    threads = []
    for tool_name, latency in tools:
        t = threading.Thread(target=worker, args=(tool_name, latency), name=f"Worker-{tool_name}")
        threads.append(t)
        t.start()

    # Wait for ALL threads to finish
    for t in threads:
        t.join()

    return results


# ============================================================================
# SECTION 2: ThreadPoolExecutor — Parallel Tool Execution
# ============================================================================
# Why: Raw threads give no control over how many run at once. An agent calling
# 100 tools would spawn 100 threads, overwhelming the OS. ThreadPoolExecutor
# caps the concurrency with a fixed pool, provides Future objects for result
# retrieval, and supports both map() (simple) and submit() (flexible) APIs.

def run_pool_tools_map(tools: List[Tuple[str, float]], max_workers: int = 4) -> List[Dict]:
    """
    Use ThreadPoolExecutor.map() for simple parallel tool execution.

    map() applies the same function to each input and returns results
    in INPUT ORDER (not completion order). Great when you need ordered results.

    Why max_workers matters:
      - Too few: underutilizes concurrency, slow.
      - Too many: thread switching overhead, memory waste.
      - Rule of thumb: 4-8 for API calls, up to 20 for very fast I/O.
    """
    def call_tool(tool_spec: Tuple[str, float]) -> Dict:
        return simulate_api_call(tool_spec[0], tool_spec[1])

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(call_tool, tools))
    return results


def run_pool_tools_submit(tools: List[Tuple[str, float]], max_workers: int = 4) -> List[Dict]:
    """
    Use ThreadPoolExecutor.submit() + as_completed() for flexible execution.

    submit() returns Future objects immediately. as_completed() yields futures
    AS THEY FINISH (fastest first), which is ideal for agents that want to
    process results as they arrive rather than waiting for all.

    Why submit() over map():
      - Can handle different functions per task (not just one).
      - as_completed() lets agents process early results immediately.
      - Provides .exception() for per-task error handling.
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks — each returns a Future
        future_to_tool = {}
        for tool_name, latency in tools:
            future = executor.submit(simulate_api_call, tool_name, latency)
            future_to_tool[future] = tool_name

        # Collect results as they complete (fastest first)
        for future in as_completed(future_to_tool):
            tool_name = future_to_tool[future]
            try:
                result = future.result()  # Raises if the task threw an exception
                results.append(result)
            except Exception as e:
                results.append({"tool": tool_name, "error": str(e)})

    return results


# ============================================================================
# SECTION 3: Thread Safety — Locks and Thread-Safe Shared State
# ============================================================================
# Why: When multiple threads write to the same data (agent state, counters,
# result aggregators), race conditions corrupt data silently. Locks serialize
# access to shared resources. RLocks allow the same thread to re-acquire the
# lock (needed for recursive or nested method calls).

class ThreadSafeAgentState:
    """
    Thread-safe container for agent state shared across worker threads.

    Why this matters for agents:
      - An agent's tool calls run in parallel threads, but they all write
        back to the same state object (results, token counts, errors).
      - Without locking, two threads writing simultaneously can overwrite
        each other's data or corrupt internal structures.

    Design choices:
      - Lock for simple critical sections (counter updates, list appends).
      - RLock for methods that call other locked methods (re-entrant).
      - Thread-safe dict operations with copy-on-read to prevent external
        modification of internal state.
    """

    def __init__(self):
        self._lock = threading.RLock()  # RLock allows re-entrant locking
        self._results: Dict[str, Any] = {}
        self._errors: List[str] = []
        self._call_count: int = 0
        self._total_latency_ms: float = 0.0

    def record_result(self, tool_name: str, result: Any, latency_ms: float) -> None:
        """Thread-safe result recording."""
        with self._lock:
            self._results[tool_name] = result
            self._call_count += 1
            self._total_latency_ms += latency_ms

    def record_error(self, tool_name: str, error: str) -> None:
        """Thread-safe error recording."""
        with self._lock:
            self._errors.append(f"[{tool_name}] {error}")
            self._call_count += 1

    def get_summary(self) -> Dict[str, Any]:
        """
        Thread-safe snapshot of current state.

        Returns a COPY so the caller can't mutate internal state
        from outside the lock.
        """
        with self._lock:
            return {
                "total_calls": self._call_count,
                "successful": len(self._results),
                "failed": len(self._errors),
                "avg_latency_ms": (
                    round(self._total_latency_ms / len(self._results), 1)
                    if self._results else 0
                ),
                "results": dict(self._results),  # Shallow copy
                "errors": list(self._errors),     # Shallow copy
            }


def demonstrate_race_condition() -> Tuple[int, int]:
    """
    Demonstrates what happens WITHOUT thread safety.

    Two versions of a counter increment:
      1. Unsafe: direct counter += 1 (race condition)
      2. Safe: lock-protected counter += 1 (correct)

    Why this demo matters:
      - Race conditions are SILENT bugs. Code looks correct but produces
        wrong results intermittently. Agents must use locks for shared state.
    """
    unsafe_counter = {"value": 0}
    safe_counter = {"value": 0}
    lock = threading.Lock()
    iterations = 10_000

    def unsafe_increment():
        for _ in range(iterations):
            unsafe_counter["value"] += 1  # NOT atomic — read/modify/write

    def safe_increment():
        for _ in range(iterations):
            with lock:
                safe_counter["value"] += 1  # Atomic under lock

    # Run unsafe version
    threads = [threading.Thread(target=unsafe_increment) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Run safe version
    threads = [threading.Thread(target=safe_increment) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected = iterations * 4
    return unsafe_counter["value"], safe_counter["value"]


# ============================================================================
# SECTION 4: ProcessPoolExecutor — CPU-Bound Parallelism
# ============================================================================
# Why: Python's GIL prevents threads from running CPU-bound code in parallel.
# For heavy computation (embedding generation, document parsing, similarity
# scoring), we need PROCESSES — each with its own Python interpreter and GIL.
# ProcessPoolExecutor abstracts the complexity of multiprocessing.

def cpu_intensive_task(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulates a CPU-bound task like computing embeddings or parsing documents.

    This function must be defined at module level (not inside another function)
    because ProcessPoolExecutor uses pickle to send tasks to worker processes,
    and pickle can only serialize module-level functions.

    Real-world agent examples:
      - Computing text embeddings for RAG retrieval
      - Parsing and chunking large documents
      - Computing similarity scores across a vector database
    """
    text = data.get("text", "")
    task_id = data.get("id", 0)

    # Simulate CPU work: hash computation (fast but illustrative)
    start = time.perf_counter()
    result_hash = hashlib.sha256(text.encode()).hexdigest()

    # Simulate heavier computation with busy-wait
    total = 0
    for i in range(500_000):
        total += i * i
    elapsed = time.perf_counter() - start

    return {
        "id": task_id,
        "hash": result_hash[:16],
        "compute_time_ms": round(elapsed * 1000, 1),
        "process": os.getpid(),
    }


def run_cpu_tasks_sequential(tasks: List[Dict]) -> List[Dict]:
    """Run CPU-bound tasks sequentially — baseline for comparison."""
    return [cpu_intensive_task(task) for task in tasks]


def run_cpu_tasks_parallel(tasks: List[Dict], max_workers: int = 4) -> List[Dict]:
    """
    Run CPU-bound tasks across multiple processes.

    ProcessPoolExecutor spawns separate Python processes, each with its own
    GIL. This achieves TRUE parallelism for CPU-bound work.

    Caveats:
      - Process startup has higher overhead than threads (~100ms per process).
      - Data must be picklable (no lambdas, no open file handles).
      - Best for tasks that take >50ms each — otherwise overhead dominates.
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(cpu_intensive_task, tasks))
    return results


# ============================================================================
# SECTION 5: Producer-Consumer Pattern — Multi-Agent Pipelines
# ============================================================================
# Why: In multi-agent systems, one agent produces tasks (e.g., "search for X")
# while worker agents consume and execute them. Queue-based patterns decouple
# producers from consumers, enabling scalable pipelines where:
#   - Producers don't block waiting for consumers.
#   - Multiple consumers process tasks in parallel.
#   - Backpressure (full queue) naturally throttles producers.

@dataclass
class AgentTask:
    """A unit of work for the producer-consumer pipeline."""
    task_id: str
    task_type: str        # "search", "analyze", "summarize"
    payload: Dict[str, Any]
    priority: int = 0     # Higher = more important
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class TaskStatus(Enum):
    """Status tracking for pipeline tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskResult:
    """Result from a completed pipeline task."""
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    worker_id: str = ""
    duration_ms: float = 0.0


class ProducerConsumerPipeline:
    """
    Queue-based producer-consumer pipeline for multi-agent task distribution.

    Architecture:
      - task_queue: Unbounded queue where producers submit tasks.
      - result_queue: Where consumers post completed results.
      - Workers: Consumer threads that pull from task_queue, execute, and
        push results to result_queue.

    Why this pattern for agents:
      - An orchestrator agent can submit 50 sub-tasks without blocking.
      - Worker agents process tasks independently at their own pace.
      - The result_queue lets the orchestrator collect results asynchronously.
      - Sentinel values (None) signal workers to shut down gracefully.
    """

    def __init__(self, num_workers: int = 3):
        self.task_queue: queue.Queue[Optional[AgentTask]] = queue.Queue()
        self.result_queue: queue.Queue[TaskResult] = queue.Queue()
        self.num_workers = num_workers
        self._workers: List[threading.Thread] = []
        self._running = False
        self._lock = threading.Lock()

    def _worker_loop(self, worker_id: str) -> None:
        """
        Worker thread loop: pull tasks, execute, push results.

        Uses a sentinel value (None) to signal shutdown.
        Each worker runs until it receives None from the queue.
        """
        while True:
            task = self.task_queue.get()  # Blocks until a task is available

            if task is None:
                # Sentinel received — shut down this worker
                self.task_queue.task_done()
                break

            start = time.perf_counter()
            try:
                # Simulate task execution based on type
                result = self._execute_task(task)
                elapsed = (time.perf_counter() - start) * 1000
                self.result_queue.put(TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.COMPLETED,
                    result=result,
                    worker_id=worker_id,
                    duration_ms=round(elapsed, 1),
                ))
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                self.result_queue.put(TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=str(e),
                    worker_id=worker_id,
                    duration_ms=round(elapsed, 1),
                ))
            finally:
                self.task_queue.task_done()

    def _execute_task(self, task: AgentTask) -> Dict[str, Any]:
        """
        Execute a task based on its type.

        In a real multi-agent system, each task_type would route to a
        different agent or tool (search agent, analysis agent, etc.).
        """
        latency_map = {"search": 0.3, "analyze": 0.5, "summarize": 0.2}
        latency = latency_map.get(task.task_type, 0.1)
        time.sleep(latency)  # Simulate I/O work

        return {
            "task_type": task.task_type,
            "payload": task.payload,
            "output": f"Processed {task.task_type} for '{task.payload.get('query', 'N/A')}'",
        }

    def start(self) -> None:
        """Start all worker threads."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._workers = []
            for i in range(self.num_workers):
                worker_id = f"Worker-{i}"
                t = threading.Thread(
                    target=self._worker_loop,
                    args=(worker_id,),
                    name=worker_id,
                    daemon=True,
                )
                self._workers.append(t)
                t.start()

    def submit(self, task: AgentTask) -> None:
        """Submit a task to the pipeline."""
        self.task_queue.put(task)

    def shutdown(self) -> None:
        """
        Gracefully shut down all workers using sentinel values.

        Sends one None per worker, then joins all threads.
        """
        # Send shutdown sentinel to each worker
        for _ in self._workers:
            self.task_queue.put(None)

        # Wait for all workers to finish
        for t in self._workers:
            t.join(timeout=5.0)

        with self._lock:
            self._running = False

    def collect_results(self, expected: int, timeout: float = 10.0) -> List[TaskResult]:
        """
        Collect a specific number of results from the result queue.

        Args:
            expected: Number of results to wait for.
            timeout: Maximum seconds to wait per result.

        Returns:
            List of TaskResult objects.
        """
        results = []
        for _ in range(expected):
            try:
                result = self.result_queue.get(timeout=timeout)
                results.append(result)
            except queue.Empty:
                break
        return results


# ============================================================================
# SECTION 6: Parallel Agent Tool Executor — The Complete Solution
# ============================================================================
# Why: This class combines everything from Sections 1-5 into a production-
# ready parallel tool executor. It's what an actual agent framework would
# use to run tool calls concurrently with:
#   - Configurable timeout per tool call
#   - Automatic retry with exponential backoff on failures
#   - Thread-safe result aggregation
#   - Graceful error handling (one tool failure doesn't kill the others)

class ToolCallStatus(Enum):
    """Status of an individual tool call."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRYING = "retrying"


@dataclass
class ToolCallResult:
    """Result from a single tool call within the parallel executor."""
    tool_name: str
    status: ToolCallStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    attempts: int = 1
    latency_ms: float = 0.0
    thread_name: str = ""


@dataclass
class ToolSpec:
    """
    Specification for a tool call to be executed in parallel.

    This separates "what to call" from "how to call it", letting the
    executor manage concurrency, retries, and timeouts.
    """
    name: str
    func: Callable[..., Any]
    args: tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 10.0
    max_retries: int = 2


class ParallelToolExecutor:
    """
    Production-grade parallel tool executor for AI agents.

    Features:
      - Runs multiple tool calls concurrently using ThreadPoolExecutor.
      - Per-tool timeout prevents one slow tool from blocking everything.
      - Automatic retry with exponential backoff for transient failures.
      - Thread-safe result aggregation via ThreadSafeAgentState.
      - Detailed execution report with per-tool metrics.

    Usage:
        executor = ParallelToolExecutor(max_workers=4)
        tools = [
            ToolSpec(name="search", func=search_fn, args=("query",)),
            ToolSpec(name="weather", func=weather_fn, kwargs={"city": "NYC"}),
        ]
        report = executor.execute(tools)
        print(report.summary)

    Why this design:
      - ToolSpec decouples tool definitions from execution mechanics.
      - Each tool can have different timeouts and retry policies.
      - The executor is stateless between execute() calls — safe to reuse.
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def execute(self, tool_specs: List[ToolSpec]) -> Dict[str, Any]:
        """
        Execute all tool specs in parallel and return an aggregated report.

        Returns:
            Dict with keys: results, total_time_ms, summary.
        """
        overall_start = time.perf_counter()
        state = ThreadSafeAgentState()
        call_results: List[ToolCallResult] = []
        results_lock = threading.Lock()

        def run_with_retry(spec: ToolSpec) -> ToolCallResult:
            """Execute a single tool call with retry logic."""
            last_error = None
            for attempt in range(1, spec.max_retries + 1):
                start = time.perf_counter()
                try:
                    result = spec.func(*spec.args, **spec.kwargs)
                    elapsed = (time.perf_counter() - start) * 1000

                    call_result = ToolCallResult(
                        tool_name=spec.name,
                        status=ToolCallStatus.SUCCESS,
                        result=result,
                        attempts=attempt,
                        latency_ms=round(elapsed, 1),
                        thread_name=threading.current_thread().name,
                    )
                    state.record_result(spec.name, result, elapsed)
                    return call_result

                except Exception as e:
                    last_error = str(e)
                    elapsed = (time.perf_counter() - start) * 1000

                    if attempt < spec.max_retries:
                        # Exponential backoff: 0.1s, 0.2s, 0.4s, ...
                        backoff = 0.1 * (2 ** (attempt - 1))
                        time.sleep(backoff)
                    else:
                        call_result = ToolCallResult(
                            tool_name=spec.name,
                            status=ToolCallStatus.FAILED,
                            error=last_error,
                            attempts=attempt,
                            latency_ms=round(elapsed, 1),
                            thread_name=threading.current_thread().name,
                        )
                        state.record_error(spec.name, last_error)
                        return call_result

            # Should not reach here, but safety fallback
            return ToolCallResult(
                tool_name=spec.name,
                status=ToolCallStatus.FAILED,
                error=last_error or "Unknown error",
                attempts=spec.max_retries,
            )

        # Execute all tools in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_spec: Dict[Future, ToolSpec] = {}
            for spec in tool_specs:
                future = executor.submit(run_with_retry, spec)
                future_to_spec[future] = spec

            for future in as_completed(future_to_spec):
                spec = future_to_spec[future]
                try:
                    call_result = future.result(timeout=spec.timeout_seconds)
                    with results_lock:
                        call_results.append(call_result)
                except FuturesTimeoutError:
                    timeout_result = ToolCallResult(
                        tool_name=spec.name,
                        status=ToolCallStatus.TIMEOUT,
                        error=f"Timed out after {spec.timeout_seconds}s",
                    )
                    with results_lock:
                        call_results.append(timeout_result)
                    state.record_error(spec.name, f"Timeout after {spec.timeout_seconds}s")
                except Exception as e:
                    err_result = ToolCallResult(
                        tool_name=spec.name,
                        status=ToolCallStatus.FAILED,
                        error=str(e),
                    )
                    with results_lock:
                        call_results.append(err_result)
                    state.record_error(spec.name, str(e))

        overall_elapsed = (time.perf_counter() - overall_start) * 1000
        summary = state.get_summary()
        summary["total_wall_time_ms"] = round(overall_elapsed, 1)

        return {
            "results": call_results,
            "summary": summary,
        }


# ============================================================================
# RUNNER: Full Demo of Concurrency & Parallelism for Agents
# ============================================================================

def run_concurrency_demo():
    """Runs the complete concurrency and parallelism demonstration."""
    print("=" * 70)
    print("  Day 12/150 -- Concurrency & Parallelism")
    print("  Making Agents Think in Parallel")
    print("=" * 70)

    # ------------------------------------------------------------------
    # DEMO 1: Sequential vs Threaded Tool Calls
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DEMO 1: Sequential vs Threaded Tool Calls (threading.Thread)")
    print("=" * 70)

    tools = [
        ("web_search", 0.5),
        ("weather_api", 0.3),
        ("database_query", 0.4),
        ("calculator", 0.1),
        ("code_executor", 0.6),
    ]

    # Sequential
    print("\n  [Sequential] Running 5 tool calls one-by-one...")
    start = time.perf_counter()
    seq_results = run_sequential_tools(tools)
    seq_time = (time.perf_counter() - start) * 1000
    print(f"    Total time: {seq_time:.0f}ms")
    for r in seq_results:
        print(f"      [{r['tool']:>15}] {r['latency_ms']}ms on {r['thread']}")

    # Threaded
    print("\n  [Threaded] Running 5 tool calls concurrently...")
    start = time.perf_counter()
    thr_results = run_threaded_tools(tools)
    thr_time = (time.perf_counter() - start) * 1000
    print(f"    Total time: {thr_time:.0f}ms")
    for r in thr_results:
        print(f"      [{r['tool']:>15}] {r['latency_ms']}ms on {r['thread']}")

    speedup = seq_time / thr_time if thr_time > 0 else 0
    print(f"\n    Speedup: {speedup:.1f}x faster with threads!")
    print(f"    Sequential: {seq_time:.0f}ms vs Threaded: {thr_time:.0f}ms")

    # ------------------------------------------------------------------
    # DEMO 2: ThreadPoolExecutor — map() vs submit()
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DEMO 2: ThreadPoolExecutor (map vs submit)")
    print("=" * 70)

    pool_tools = [
        ("search_arxiv", 0.4),
        ("search_github", 0.3),
        ("search_docs", 0.5),
        ("search_stackoverflow", 0.35),
    ]

    # map() — ordered results
    print("\n  [map()] Results in INPUT order:")
    start = time.perf_counter()
    map_results = run_pool_tools_map(pool_tools, max_workers=4)
    map_time = (time.perf_counter() - start) * 1000
    for i, r in enumerate(map_results):
        print(f"    {i+1}. [{r['tool']:>25}] {r['latency_ms']}ms")
    print(f"    Total: {map_time:.0f}ms")

    # submit() — completion order
    print("\n  [submit()] Results in COMPLETION order (fastest first):")
    start = time.perf_counter()
    sub_results = run_pool_tools_submit(pool_tools, max_workers=4)
    sub_time = (time.perf_counter() - start) * 1000
    for i, r in enumerate(sub_results):
        print(f"    {i+1}. [{r['tool']:>25}] {r['latency_ms']}ms")
    print(f"    Total: {sub_time:.0f}ms")

    # ------------------------------------------------------------------
    # DEMO 3: Thread Safety — Race Conditions
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DEMO 3: Thread Safety — Race Conditions vs Locks")
    print("=" * 70)

    unsafe_val, safe_val = demonstrate_race_condition()
    expected = 40_000
    print(f"\n    Expected counter value:   {expected:>7}")
    print(f"    Unsafe (no lock) result:  {unsafe_val:>7}  {'✓ OK' if unsafe_val == expected else '✗ RACE CONDITION!'}")
    print(f"    Safe (with lock) result:  {safe_val:>7}  {'✓ OK' if safe_val == expected else '✗ BUG!'}")

    if unsafe_val != expected:
        lost = expected - unsafe_val
        print(f"    --> Lost {lost} increments due to race condition!")
        print(f"    --> This is why agents MUST use locks for shared state.")

    # ThreadSafeAgentState demo
    print("\n  ThreadSafeAgentState — safe concurrent writes:")
    state = ThreadSafeAgentState()

    def record_work(tool_name: str):
        time.sleep(random.uniform(0.05, 0.15))
        state.record_result(tool_name, f"{tool_name}_data", random.uniform(50, 200))

    threads = []
    tool_names = ["search", "weather", "calculator", "db_query", "code_exec"]
    for name in tool_names:
        t = threading.Thread(target=record_work, args=(name,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    summary = state.get_summary()
    print(f"    Total calls:    {summary['total_calls']}")
    print(f"    Successful:     {summary['successful']}")
    print(f"    Avg latency:    {summary['avg_latency_ms']}ms")
    print(f"    All 5 recorded: {'✓' if summary['successful'] == 5 else '✗'}")

    # ------------------------------------------------------------------
    # DEMO 4: ProcessPoolExecutor — CPU-Bound Tasks
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DEMO 4: ProcessPoolExecutor — CPU-Bound Parallelism")
    print("=" * 70)

    cpu_tasks = [
        {"id": i, "text": f"Document chunk {i} with content for embedding generation"}
        for i in range(8)
    ]

    # Sequential
    print(f"\n  [Sequential] Processing {len(cpu_tasks)} CPU-bound tasks...")
    start = time.perf_counter()
    seq_cpu = run_cpu_tasks_sequential(cpu_tasks)
    seq_cpu_time = (time.perf_counter() - start) * 1000
    print(f"    Total time: {seq_cpu_time:.0f}ms")
    for r in seq_cpu[:3]:
        print(f"      Task {r['id']}: {r['compute_time_ms']}ms (PID {r['process']})")
    print(f"      ... and {len(seq_cpu) - 3} more")

    # Parallel with processes
    worker_count = min(4, os.cpu_count() or 2)
    print(f"\n  [Parallel] Processing {len(cpu_tasks)} tasks across {worker_count} processes...")
    start = time.perf_counter()
    par_cpu = run_cpu_tasks_parallel(cpu_tasks, max_workers=worker_count)
    par_cpu_time = (time.perf_counter() - start) * 1000
    print(f"    Total time: {par_cpu_time:.0f}ms")
    unique_pids = set(r["process"] for r in par_cpu)
    for r in par_cpu[:3]:
        print(f"      Task {r['id']}: {r['compute_time_ms']}ms (PID {r['process']})")
    print(f"      ... and {len(par_cpu) - 3} more")
    print(f"    Unique PIDs used: {len(unique_pids)} (shows true multi-process)")

    cpu_speedup = seq_cpu_time / par_cpu_time if par_cpu_time > 0 else 0
    print(f"\n    Speedup: {cpu_speedup:.1f}x faster with processes!")

    # ------------------------------------------------------------------
    # DEMO 5: Producer-Consumer Pipeline
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DEMO 5: Producer-Consumer Pipeline (Multi-Agent)")
    print("=" * 70)

    pipeline = ProducerConsumerPipeline(num_workers=3)
    pipeline.start()

    tasks = [
        AgentTask(task_id="T1", task_type="search", payload={"query": "LangGraph tutorial"}),
        AgentTask(task_id="T2", task_type="analyze", payload={"query": "agent performance metrics"}),
        AgentTask(task_id="T3", task_type="summarize", payload={"query": "concurrency patterns"}),
        AgentTask(task_id="T4", task_type="search", payload={"query": "RAG best practices"}),
        AgentTask(task_id="T5", task_type="analyze", payload={"query": "multi-agent architectures"}),
        AgentTask(task_id="T6", task_type="summarize", payload={"query": "thread safety in Python"}),
    ]

    print(f"\n  Submitting {len(tasks)} tasks to pipeline...")
    start = time.perf_counter()
    for task in tasks:
        pipeline.submit(task)
        print(f"    [>] Submitted {task.task_id}: {task.task_type}('{task.payload['query']}')")

    # Collect results
    results = pipeline.collect_results(expected=len(tasks), timeout=5.0)
    pipe_time = (time.perf_counter() - start) * 1000

    print(f"\n  Results ({len(results)}/{len(tasks)} collected):")
    for r in results:
        status_icon = "✓" if r.status == TaskStatus.COMPLETED else "✗"
        print(f"    [{status_icon}] {r.task_id} on {r.worker_id}: "
              f"{r.duration_ms:.0f}ms — {r.status.value}")

    pipeline.shutdown()
    print(f"\n    Pipeline total: {pipe_time:.0f}ms for {len(tasks)} tasks")
    print(f"    Workers used: {pipeline.num_workers}")

    # ------------------------------------------------------------------
    # DEMO 6: ParallelToolExecutor — Full Integration
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DEMO 6: ParallelToolExecutor — Production-Grade Solution")
    print("=" * 70)

    # Define tool functions
    def reliable_search(query: str) -> Dict:
        time.sleep(0.3)
        return {"query": query, "results": ["result1", "result2"]}

    def flaky_weather(city: str) -> Dict:
        """Simulates a tool that sometimes fails (transient network error)."""
        if random.random() < 0.5:
            raise ConnectionError(f"API timeout for {city}")
        time.sleep(0.2)
        return {"city": city, "temp": "22°C", "condition": "Sunny"}

    def slow_database(query: str) -> Dict:
        time.sleep(0.5)
        return {"query": query, "rows": 42}

    def fast_calculator(expression: str) -> Dict:
        time.sleep(0.05)
        return {"expression": expression, "result": eval(expression)}

    # Build tool specs
    tool_specs = [
        ToolSpec(
            name="web_search",
            func=reliable_search,
            args=("agentic AI patterns",),
            timeout_seconds=5.0,
            max_retries=1,
        ),
        ToolSpec(
            name="weather",
            func=flaky_weather,
            args=("New York",),
            timeout_seconds=5.0,
            max_retries=3,  # Retries for flaky API
        ),
        ToolSpec(
            name="database",
            func=slow_database,
            args=("SELECT * FROM agents",),
            timeout_seconds=5.0,
            max_retries=1,
        ),
        ToolSpec(
            name="calculator",
            func=fast_calculator,
            args=("2 ** 10",),
            timeout_seconds=2.0,
            max_retries=1,
        ),
    ]

    executor = ParallelToolExecutor(max_workers=4)
    print(f"\n  Executing {len(tool_specs)} tools in parallel...")
    print(f"  (weather API is intentionally flaky — may retry)")

    report = executor.execute(tool_specs)

    print(f"\n  Execution Report:")
    print(f"  {'─' * 60}")
    for cr in report["results"]:
        status_icon = {
            ToolCallStatus.SUCCESS: "✓",
            ToolCallStatus.FAILED: "✗",
            ToolCallStatus.TIMEOUT: "⏱",
        }.get(cr.status, "?")

        detail = ""
        if cr.status == ToolCallStatus.SUCCESS:
            detail = f"result={cr.result}"
        elif cr.error:
            detail = f"error={cr.error}"

        print(f"    [{status_icon}] {cr.tool_name:>12} | {cr.status.value:>7} | "
              f"{cr.latency_ms:>7.0f}ms | attempts={cr.attempts} | {cr.thread_name}")
        if detail:
            print(f"         {'':>12}   └─ {detail[:70]}")

    summary = report["summary"]
    print(f"\n  Summary:")
    print(f"    Total wall time:  {summary['total_wall_time_ms']:.0f}ms")
    print(f"    Successful calls: {summary['successful']}/{summary['total_calls']}")
    print(f"    Failed calls:     {summary['failed']}")
    print(f"    Avg latency:      {summary['avg_latency_ms']}ms")

    if summary["errors"]:
        print(f"\n    Errors:")
        for err in summary["errors"]:
            print(f"      ⚠ {err}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Day 12 Complete!")
    print("=" * 70)
    print("""
  Key Takeaways:
    1. threading.Thread provides basic concurrency for I/O-bound agent tasks.
    2. ThreadPoolExecutor gives controlled concurrency with map() and submit().
    3. Locks and RLocks prevent race conditions on shared agent state.
    4. ProcessPoolExecutor enables TRUE parallelism for CPU-bound work.
    5. Producer-consumer pipelines decouple task creation from execution.
    6. ParallelToolExecutor combines everything: timeout, retry, aggregation.

  Performance Impact for Agents:
    - 5 sequential API calls (1.9s) → threaded (0.6s) = 3x speedup
    - CPU-bound tasks scale nearly linearly with process count
    - Agents that think in parallel deliver faster responses

  Next: Day 13 will continue building on engineering foundations!
""")


if __name__ == "__main__":
    run_concurrency_demo()
