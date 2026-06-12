"""
🤖 Day 08/150 — Error Handling & Logging for Production
========================================================
Week 2: Advanced Python + Dev Tools

Concepts:
  1. Custom Exception Hierarchies — structured error types for agents
  2. Python Logging Module — structured logging with levels & formatters
  3. Context Managers — safe resource management (files, connections, sessions)
  4. Retry Patterns with Tenacity — production-grade retry decorators

Why this matters for Agentic AI:
  - Agents fail constantly: API timeouts, rate limits, malformed LLM output
  - Without structured error handling, failures are silent and undebuggable
  - Without logging, you're flying blind in production
  - Without context managers, resource leaks crash long-running agents
"""

import asyncio
import logging
import time
import random
import sys
import json
import traceback
from abc import ABC
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ============================================================
# SECTION 1: Custom Exception Hierarchy
# ============================================================
# Why: Agents need STRUCTURED error types.
# "Exception" tells you nothing.
# "ToolExecutionError(tool='web_search', reason='timeout')" tells you everything.

class AgentError(Exception):
    """Base exception for all agent-related errors.
    Every agent error carries: message, error_code, context dict."""

    def __init__(self, message: str, error_code: str = "AGENT_ERROR",
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": str(self),
            "context": self.context,
            "timestamp": self.timestamp
        }


class LLMError(AgentError):
    """Errors from LLM API calls — timeouts, rate limits, bad responses."""

    def __init__(self, message: str, model: str = "unknown",
                 status_code: Optional[int] = None, **kwargs):
        super().__init__(
            message,
            error_code="LLM_ERROR",
            context={"model": model, "status_code": status_code, **kwargs}
        )
        self.model = model
        self.status_code = status_code


class RateLimitError(LLMError):
    """429 Too Many Requests — needs backoff."""

    def __init__(self, model: str, retry_after: float = 0):
        super().__init__(
            f"Rate limit exceeded for {model}. Retry after {retry_after}s",
            model=model,
            status_code=429,
            retry_after=retry_after
        )
        self.retry_after = retry_after
        self.error_code = "RATE_LIMIT"


class ToolExecutionError(AgentError):
    """Errors when an agent tool fails — web_search, code_exec, etc."""

    def __init__(self, tool_name: str, reason: str,
                 input_data: Optional[Dict] = None):
        super().__init__(
            f"Tool '{tool_name}' failed: {reason}",
            error_code="TOOL_EXEC_ERROR",
            context={"tool": tool_name, "reason": reason, "input": input_data}
        )
        self.tool_name = tool_name


class OutputParsingError(AgentError):
    """LLM returned malformed/unparseable output."""

    def __init__(self, expected_format: str, raw_output: str):
        super().__init__(
            f"Failed to parse LLM output. Expected: {expected_format}",
            error_code="PARSE_ERROR",
            context={
                "expected": expected_format,
                "raw_output": raw_output[:200]  # Truncate for safety
            }
        )


class MaxRetriesExceededError(AgentError):
    """All retry attempts exhausted."""

    def __init__(self, operation: str, attempts: int):
        super().__init__(
            f"Max retries ({attempts}) exceeded for: {operation}",
            error_code="MAX_RETRIES",
            context={"operation": operation, "attempts": attempts}
        )


# ============================================================
# SECTION 2: Structured Logging System
# ============================================================
# Why: print() is for demos. logging is for production.
# Agents need structured, leveled, filterable logs.

class AgentLogFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs.
    Production agents need machine-readable logs for monitoring."""

    LEVEL_ICONS = {
        "DEBUG": "🔍",
        "INFO": "📋",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "🔥",
    }

    def format(self, record: logging.LogRecord) -> str:
        icon = self.LEVEL_ICONS.get(record.levelname, "📌")
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3],
            "level": record.levelname,
            "icon": icon,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach extra context if provided
        if hasattr(record, "agent_context"):
            log_entry["context"] = record.agent_context

        # Attach exception info if present
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, ensure_ascii=False)


class PrettyLogFormatter(logging.Formatter):
    """Human-readable colored formatter for development/console output."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[1;31m", # Bold Red
    }
    RESET = "\033[0m"
    ICONS = {
        "DEBUG": "🔍", "INFO": "✅", "WARNING": "⚠️",
        "ERROR": "❌", "CRITICAL": "🔥"
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        icon = self.ICONS.get(record.levelname, "📌")
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]

        base = f"  {color}[{ts}] {icon} {record.levelname:<8}{self.RESET} {record.getMessage()}"

        if hasattr(record, "agent_context"):
            ctx = record.agent_context
            base += f" {color}| ctx={json.dumps(ctx)}{self.RESET}"

        if record.exc_info and record.exc_info[0]:
            base += f"\n    {color}└── {record.exc_info[0].__name__}: {record.exc_info[1]}{self.RESET}"

        return base


def setup_agent_logger(name: str = "agent", level: int = logging.DEBUG) -> logging.Logger:
    """Creates a production-ready logger with both console and structured output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers on re-initialization
    if logger.handlers:
        logger.handlers.clear()

    # Console handler — pretty output for humans
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(PrettyLogFormatter())
    logger.addHandler(console_handler)

    return logger


# ============================================================
# SECTION 3: Context Managers — Resource Safety
# ============================================================
# Why: Agents open files, connections, API sessions.
# If they crash mid-execution, resources MUST be cleaned up.
# Context managers guarantee cleanup even on exceptions.

@dataclass
class ResourceMetrics:
    """Tracks resource usage across context manager lifecycle."""
    resources_opened: int = 0
    resources_closed: int = 0
    errors_caught: int = 0
    total_time_ms: float = 0.0


resource_metrics = ResourceMetrics()


@contextmanager
def managed_api_session(session_name: str, logger: logging.Logger):
    """Context manager for API sessions — guarantees cleanup.
    Usage: with managed_api_session('openai', logger) as session: ..."""

    logger.info(f"Opening API session: {session_name}",
                extra={"agent_context": {"session": session_name}})
    resource_metrics.resources_opened += 1
    start_time = time.time()

    session = {"name": session_name, "status": "active", "created": time.time()}

    try:
        yield session
    except AgentError as e:
        resource_metrics.errors_caught += 1
        logger.error(f"Agent error in session '{session_name}': {e}",
                     extra={"agent_context": e.to_dict()})
        raise
    except Exception as e:
        resource_metrics.errors_caught += 1
        logger.critical(f"Unexpected error in session '{session_name}': {e}",
                        exc_info=True)
        raise
    finally:
        elapsed = (time.time() - start_time) * 1000
        resource_metrics.resources_closed += 1
        resource_metrics.total_time_ms += elapsed
        session["status"] = "closed"
        logger.info(f"Closed API session: {session_name} ({elapsed:.0f}ms)",
                     extra={"agent_context": {"session": session_name,
                                              "duration_ms": round(elapsed, 2)}})


@asynccontextmanager
async def managed_agent_execution(agent_name: str, logger: logging.Logger):
    """Async context manager for agent execution lifecycle.
    Wraps the entire agent run with setup/teardown logging."""

    logger.info(f"Agent '{agent_name}' execution started",
                extra={"agent_context": {"agent": agent_name, "phase": "start"}})
    start_time = time.time()

    try:
        yield {"agent": agent_name, "start_time": start_time}
    except AgentError as e:
        elapsed = (time.time() - start_time) * 1000
        logger.error(f"Agent '{agent_name}' failed after {elapsed:.0f}ms",
                     extra={"agent_context": {**e.to_dict(), "duration_ms": elapsed}})
        raise
    finally:
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Agent '{agent_name}' execution finished ({elapsed:.0f}ms)",
                    extra={"agent_context": {"agent": agent_name,
                                             "phase": "end",
                                             "duration_ms": round(elapsed, 2)}})


# ============================================================
# SECTION 4: Retry Engine (Production-Grade)
# ============================================================
# Why: Tenacity-inspired retry decorator.
# Agents hit rate limits, network glitches, transient failures.
# A good retry engine is the difference between demo and production.

class RetryStrategy(Enum):
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"


@dataclass
class RetryConfig:
    """Production retry configuration — inspired by tenacity library."""
    max_attempts: int = 4
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: bool = True
    retry_on: tuple = (RateLimitError, ToolExecutionError, ConnectionError)


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay based on retry strategy."""
    if config.strategy == RetryStrategy.FIXED:
        delay = config.base_delay
    elif config.strategy == RetryStrategy.LINEAR:
        delay = config.base_delay * (attempt + 1)
    else:  # EXPONENTIAL
        delay = config.base_delay * (2 ** attempt)

    # Add jitter to prevent thundering herd
    if config.jitter:
        delay += random.uniform(0, delay * 0.3)

    return min(delay, config.max_delay)


async def retry_with_logging(
    func,
    args: tuple = (),
    kwargs: dict = None,
    config: RetryConfig = None,
    logger: logging.Logger = None,
    operation_name: str = "operation"
) -> Any:
    """Execute a function with retry logic and structured logging.
    Inspired by tenacity but built for agent error hierarchies."""

    config = config or RetryConfig()
    kwargs = kwargs or {}
    logger = logger or logging.getLogger("agent")

    last_error = None

    for attempt in range(config.max_attempts):
        try:
            logger.debug(f"Attempt {attempt + 1}/{config.max_attempts}: {operation_name}",
                        extra={"agent_context": {"attempt": attempt + 1,
                                                 "max": config.max_attempts}})

            result = await func(*args, **kwargs)

            if attempt > 0:
                logger.info(f"Succeeded on attempt {attempt + 1}: {operation_name}",
                           extra={"agent_context": {"attempts_needed": attempt + 1}})
            return result

        except config.retry_on as e:
            last_error = e
            delay = calculate_delay(attempt, config)

            if attempt < config.max_attempts - 1:
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...",
                    extra={"agent_context": {
                        "error": str(e),
                        "delay": round(delay, 2),
                        "attempt": attempt + 1
                    }}
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"All {config.max_attempts} attempts failed for: {operation_name}",
                    extra={"agent_context": {"final_error": str(e)}},
                    exc_info=True
                )

        except Exception as e:
            logger.critical(f"Non-retryable error in {operation_name}: {e}",
                          exc_info=True)
            raise

    raise MaxRetriesExceededError(operation_name, config.max_attempts)


# ============================================================
# SECTION 5: Demo — Putting It All Together
# ============================================================

async def simulate_llm_call(prompt: str) -> Dict[str, Any]:
    """Simulates an LLM API call that can fail in various ways."""
    await asyncio.sleep(random.uniform(0.1, 0.4))

    roll = random.random()

    if roll < 0.25:
        raise RateLimitError(model="gpt-4o", retry_after=2.0)
    elif roll < 0.35:
        raise ToolExecutionError(
            tool_name="web_search",
            reason="Connection timeout after 5000ms",
            input_data={"query": prompt}
        )
    elif roll < 0.40:
        raise OutputParsingError(
            expected_format="JSON with 'action' and 'thought' keys",
            raw_output="Sorry, I can't help with that..."
        )

    return {
        "model": "gpt-4o",
        "response": f"Processed: {prompt}",
        "tokens_used": random.randint(100, 500),
        "latency_ms": random.randint(200, 800)
    }


async def simulate_tool_call(tool_name: str) -> Dict[str, Any]:
    """Simulates an agent tool execution."""
    await asyncio.sleep(random.uniform(0.1, 0.3))

    if random.random() < 0.3:
        raise ToolExecutionError(
            tool_name=tool_name,
            reason="External service unavailable",
            input_data={"tool": tool_name}
        )

    return {"tool": tool_name, "status": "success", "result": f"{tool_name} completed"}


async def run_production_agent():
    """Full demo: exception hierarchy + logging + context managers + retry."""

    logger = setup_agent_logger("production-agent")

    print("\n" + "=" * 64)
    print("  DAY 8: ERROR HANDLING & LOGGING FOR PRODUCTION")
    print("  Week 2 — Advanced Python + Dev Tools")
    print("=" * 64 + "\n")

    # ── Demo 1: Exception Hierarchy ──
    print("─" * 64)
    print("  DEMO 1: Custom Exception Hierarchy")
    print("─" * 64)

    errors_to_demo = [
        RateLimitError(model="gpt-4o", retry_after=2.5),
        ToolExecutionError("web_search", "DNS resolution failed",
                          {"query": "latest AI news"}),
        OutputParsingError("JSON", "I'm sorry, I cannot..."),
        MaxRetriesExceededError("llm_call", 4),
    ]

    for err in errors_to_demo:
        logger.error(f"Demo error: {err.error_code}",
                    extra={"agent_context": err.to_dict()})

    # ── Demo 2: Context Managers ──
    print("\n" + "─" * 64)
    print("  DEMO 2: Context Managers — Safe Resource Management")
    print("─" * 64)

    # Successful session
    with managed_api_session("openai-gpt4", logger) as session:
        await asyncio.sleep(0.2)  # Simulate work
        logger.info(f"Session active: {session['name']}")

    # Session with handled error
    try:
        with managed_api_session("anthropic-claude", logger) as session:
            await asyncio.sleep(0.1)
            raise RateLimitError(model="claude-3", retry_after=5.0)
    except RateLimitError:
        logger.warning("Rate limit handled gracefully via context manager")

    print(f"\n  Resource Metrics:")
    print(f"    Opened:  {resource_metrics.resources_opened}")
    print(f"    Closed:  {resource_metrics.resources_closed}")
    print(f"    Errors:  {resource_metrics.errors_caught}")
    print(f"    Time:    {resource_metrics.total_time_ms:.0f}ms")

    # ── Demo 3: Retry Engine ──
    print("\n" + "─" * 64)
    print("  DEMO 3: Retry Engine — Production-Grade Resilience")
    print("─" * 64)

    retry_config = RetryConfig(
        max_attempts=4,
        strategy=RetryStrategy.EXPONENTIAL,
        base_delay=0.5,
        max_delay=10.0,
        jitter=True,
        retry_on=(RateLimitError, ToolExecutionError)
    )

    # Attempt LLM call with retries
    try:
        result = await retry_with_logging(
            simulate_llm_call,
            args=("Explain quantum computing",),
            config=retry_config,
            logger=logger,
            operation_name="llm_call(gpt-4o)"
        )
        logger.info(f"LLM call succeeded: {result['tokens_used']} tokens")
    except (MaxRetriesExceededError, OutputParsingError) as e:
        logger.error(f"LLM call ultimately failed: {e}")

    # Attempt tool call with retries
    try:
        result = await retry_with_logging(
            simulate_tool_call,
            args=("web_search",),
            config=retry_config,
            logger=logger,
            operation_name="tool_call(web_search)"
        )
        logger.info(f"Tool call succeeded: {result}")
    except MaxRetriesExceededError as e:
        logger.error(f"Tool call ultimately failed: {e}")

    # ── Demo 4: Async Agent Execution Context ──
    print("\n" + "─" * 64)
    print("  DEMO 4: Async Agent Lifecycle Context Manager")
    print("─" * 64)

    async with managed_agent_execution("ResearchAgent-v1", logger) as ctx:
        logger.info("Agent planning phase...")
        await asyncio.sleep(0.2)
        logger.info("Agent executing tools...")
        await asyncio.sleep(0.3)
        logger.info("Agent synthesizing results...")
        await asyncio.sleep(0.1)

    # ── Summary ──
    print("\n" + "=" * 64)
    print("  PRODUCTION ERROR HANDLING SUMMARY")
    print("=" * 64)
    print(f"""
  Exception Classes Built:     5
    AgentError (base)          → structured error_code + context
    LLMError                   → model + status_code tracking
    RateLimitError             → retry_after field for backoff
    ToolExecutionError         → tool_name + input_data capture
    OutputParsingError         → expected vs raw output diff

  Logging Features:            3
    PrettyLogFormatter         → colored console output for dev
    AgentLogFormatter          → JSON structured logs for prod
    setup_agent_logger()       → dual-handler logger factory

  Context Managers:            2
    managed_api_session()      → sync resource lifecycle
    managed_agent_execution()  → async agent run lifecycle

  Retry Engine:                3 strategies
    FIXED                      → constant delay between retries
    LINEAR                     → linearly increasing delays
    EXPONENTIAL                → exponential backoff + jitter

  Resources tracked:           {resource_metrics.resources_opened} opened,
                               {resource_metrics.resources_closed} closed,
                               {resource_metrics.errors_caught} errors caught
""")
    print("=" * 64)
    print("  github.com/24A31A04N2/agentic-ai-journey/day-08")
    print("=" * 64)


if __name__ == "__main__":
    asyncio.run(run_production_agent())
