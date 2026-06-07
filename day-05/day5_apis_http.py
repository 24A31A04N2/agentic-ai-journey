"""
🤖 Day 5/150 — Agentic AI Mastery Roadmap
📌 Topic: APIs & HTTP — The Language Agents Speak

Key concepts:
- REST API concepts — GET, POST, headers, auth, status codes
- httpx for sync & async HTTP (modern alternative to requests)
- API key management with environment variables
- JSON parsing, serialization, nested access
- Rate limiting and retry logic (exponential backoff)
- Build: Production API client with retry, timeout, circuit breaker

Agents don't think in Python. They think in API calls.
Every tool call = an HTTP request under the hood.
"""

import asyncio
import time
import json
import random
from dataclasses import dataclass, field
from typing import Any
from enum import Enum
from functools import wraps


# ═══════════════════════════════════════════════════════
# 1️⃣  HTTP FUNDAMENTALS — What Agents Actually Send
#     Every agent tool call is an HTTP request
# ═══════════════════════════════════════════════════════

@dataclass
class HTTPRequest:
    """What an HTTP request looks like under the hood"""
    method: str        # GET, POST, PUT, DELETE
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    body: dict | None = None
    params: dict[str, str] = field(default_factory=dict)

    def __str__(self):
        parts = [f"{self.method} {self.url}"]
        if self.params:
            params_str = "&".join(
                f"{k}={v}" for k, v in self.params.items()
            )
            parts[0] += f"?{params_str}"
        for k, v in self.headers.items():
            parts.append(f"  {k}: {v}")
        if self.body:
            parts.append(f"  Body: {json.dumps(self.body)}")
        return "\n".join(parts)


class StatusCode(Enum):
    OK = 200
    CREATED = 201
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    RATE_LIMITED = 429
    SERVER_ERROR = 500
    SERVICE_UNAVAILABLE = 503


@dataclass
class HTTPResponse:
    """What comes back from an API"""
    status_code: int
    data: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    elapsed_ms: float = 0

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def is_retryable(self) -> bool:
        return self.status_code in (429, 500, 502, 503)

    @property
    def status_name(self) -> str:
        try:
            return StatusCode(self.status_code).name
        except ValueError:
            return "UNKNOWN"


# ═══════════════════════════════════════════════════════
# 2️⃣  RETRY ENGINE — Exponential Backoff
#     Agents MUST handle failures gracefully
#     Rate limits (429) are not errors — they're signals
# ═══════════════════════════════════════════════════════

@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    base_delay: float = 1.0    # seconds
    max_delay: float = 30.0    # cap the backoff
    backoff_factor: float = 2.0
    jitter: bool = True        # add randomness

    def get_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff + jitter"""
        delay = min(
            self.base_delay * (self.backoff_factor ** attempt),
            self.max_delay,
        )
        if self.jitter:
            delay *= random.uniform(0.5, 1.5)
        return round(delay, 2)


def with_retry(config: RetryConfig | None = None):
    """
    Decorator that adds retry logic to any async function.
    This is how production agents handle flaky APIs.
    """
    if config is None:
        config = RetryConfig()

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(config.max_retries + 1):
                try:
                    result = await func(*args, **kwargs)

                    # Check if response is retryable
                    if (isinstance(result, HTTPResponse)
                            and result.is_retryable):
                        raise RetryableError(
                            f"HTTP {result.status_code}: "
                            f"{result.status_name}"
                        )
                    return result

                except (RetryableError, ConnectionError) as e:
                    last_error = e
                    if attempt < config.max_retries:
                        delay = config.get_delay(attempt)
                        print(f"    [RETRY] Attempt "
                              f"{attempt + 1}/{config.max_retries}"
                              f" failed: {e}")
                        print(f"    [RETRY] Waiting "
                              f"{delay}s before retry...")
                        await asyncio.sleep(delay * 0.01)
                    else:
                        print(f"    [RETRY] All {config.max_retries}"
                              f" retries exhausted")

            raise last_error
        return wrapper
    return decorator


class RetryableError(Exception):
    """Error that should trigger a retry"""
    pass


# ═══════════════════════════════════════════════════════
# 3️⃣  API CLIENT — Production-Grade HTTP Client
#     With auth, headers, retry, timeout, and metrics
# ═══════════════════════════════════════════════════════

@dataclass
class APIMetrics:
    """Track API call metrics — like production observability"""
    total_calls: int = 0
    successful: int = 0
    failed: int = 0
    retried: int = 0
    total_latency_ms: float = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.total_calls == 0:
            return 0
        return round(self.total_latency_ms / self.total_calls, 1)

    @property
    def success_rate(self) -> str:
        if self.total_calls == 0:
            return "0%"
        rate = (self.successful / self.total_calls) * 100
        return f"{rate:.0f}%"

    def report(self) -> dict:
        return {
            "total": self.total_calls,
            "success_rate": self.success_rate,
            "avg_latency": f"{self.avg_latency_ms}ms",
            "retries": self.retried,
        }


class AgentAPIClient:
    """
    Production-grade API client for agents.
    Handles auth, retry, rate limits, and metrics.

    This is what sits between an agent and the outside world.
    Every tool call goes through a client like this.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout_ms: int = 5000,
        retry_config: RetryConfig | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_ms = timeout_ms
        self.retry_config = retry_config or RetryConfig()
        self.metrics = APIMetrics()
        self._default_headers = {
            "Content-Type": "application/json",
            "User-Agent": "AgentAPIClient/1.0",
        }
        if api_key:
            self._default_headers["Authorization"] = (
                f"Bearer {api_key}"
            )

    async def _simulate_request(
        self, method: str, endpoint: str,
        body: dict | None = None,
        params: dict | None = None,
    ) -> HTTPResponse:
        """
        Simulate an HTTP request (in production, use httpx).
        Randomly simulates success, rate-limits, and errors.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        start = time.perf_counter()

        # Show the request being made
        req = HTTPRequest(
            method, url, self._default_headers,
            body, params or {}
        )

        # Simulate network delay
        delay = random.uniform(0.05, 0.3)
        await asyncio.sleep(delay)

        # Simulate different responses
        roll = random.random()
        elapsed = (time.perf_counter() - start) * 1000

        if roll < 0.7:  # 70% success
            return HTTPResponse(
                200,
                {"result": f"Data from {endpoint}",
                 "items": random.randint(1, 10)},
                {"X-RateLimit-Remaining": "47"},
                elapsed,
            )
        elif roll < 0.85:  # 15% rate limit
            return HTTPResponse(
                429, {"error": "Rate limit exceeded"},
                {"Retry-After": "2"}, elapsed,
            )
        else:  # 15% server error
            return HTTPResponse(
                500, {"error": "Internal server error"},
                {}, elapsed,
            )

    @with_retry()
    async def get(
        self, endpoint: str, params: dict | None = None
    ) -> HTTPResponse:
        """GET request with automatic retry"""
        self.metrics.total_calls += 1
        response = await self._simulate_request(
            "GET", endpoint, params=params
        )
        if response.is_success:
            self.metrics.successful += 1
        else:
            self.metrics.failed += 1
        self.metrics.total_latency_ms += response.elapsed_ms
        return response

    @with_retry()
    async def post(
        self, endpoint: str, body: dict | None = None
    ) -> HTTPResponse:
        """POST request with automatic retry"""
        self.metrics.total_calls += 1
        response = await self._simulate_request(
            "POST", endpoint, body=body
        )
        if response.is_success:
            self.metrics.successful += 1
        else:
            self.metrics.failed += 1
        self.metrics.total_latency_ms += response.elapsed_ms
        return response

    async def get_many(
        self, endpoints: list[str]
    ) -> list[HTTPResponse]:
        """Fetch multiple endpoints in parallel"""
        return await asyncio.gather(
            *[self.get(ep) for ep in endpoints]
        )


# ═══════════════════════════════════════════════════════
# 4️⃣  JSON PROCESSING — Agent Data Pipelines
#     Agents receive JSON from every API call
#     Processing nested JSON is a daily skill
# ═══════════════════════════════════════════════════════

def process_api_response(raw_response: dict) -> dict:
    """
    Transform raw API response into agent-friendly format.
    Agents need clean, structured data to reason about.
    """

    # Example: OpenAI chat completion response structure
    openai_response = {
        "id": "chatcmpl-abc123",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_xyz",
                            "type": "function",
                            "function": {
                                "name": "search_web",
                                "arguments": json.dumps({
                                    "query": "LangGraph docs"
                                })
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }
        ],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 42,
            "total_tokens": 192,
        }
    }

    # Extract tool calls from nested JSON
    choices = openai_response.get("choices", [])
    if not choices:
        return {"error": "No choices in response"}

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls", [])

    extracted = []
    for tc in tool_calls:
        func = tc.get("function", {})
        extracted.append({
            "tool_id": tc["id"],
            "tool_name": func["name"],
            "arguments": json.loads(func["arguments"]),
        })

    return {
        "model": openai_response["model"],
        "tool_calls": extracted,
        "tokens_used": openai_response["usage"]["total_tokens"],
        "finish_reason": choices[0]["finish_reason"],
    }


# ═══════════════════════════════════════════════════════
# 5️⃣  ENVIRONMENT & API KEY MANAGEMENT
#     NEVER hardcode API keys — use environment variables
# ═══════════════════════════════════════════════════════

class SecureConfig:
    """
    Secure API key management.
    In production: use python-dotenv + .env files.
    """

    def __init__(self):
        # Simulating loading from .env
        self._secrets = {
            "OPENAI_API_KEY": "sk-proj-***masked***",
            "ANTHROPIC_API_KEY": "sk-ant-***masked***",
            "GOOGLE_API_KEY": "AIza-***masked***",
        }

    def get_key(self, name: str) -> str:
        key = self._secrets.get(name)
        if not key:
            raise ValueError(
                f"Missing API key: {name}. "
                f"Add it to your .env file."
            )
        return key

    def show_loaded(self):
        print("\n  Loaded API keys:")
        for name in self._secrets:
            masked = self._secrets[name][:8] + "***"
            print(f"    {name} = {masked}")


# ═══════════════════════════════════════════════════════
#  RUN ALL DEMOS
# ═══════════════════════════════════════════════════════

async def main():
    # --- Demo 1: HTTP Request Anatomy ---
    print("=" * 55)
    print("  DEMO 1: HTTP Request Anatomy")
    print("=" * 55)

    request = HTTPRequest(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": "Bearer sk-***",
            "Content-Type": "application/json",
        },
        body={
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Explain agents"}
            ],
        },
    )
    print(f"\n  Every agent tool call is this:\n")
    print(f"  {request}")

    # --- Demo 2: Retry with Exponential Backoff ---
    print("\n" + "=" * 55)
    print("  DEMO 2: Exponential Backoff (Retry Engine)")
    print("=" * 55)

    config = RetryConfig(
        max_retries=5, base_delay=1.0, backoff_factor=2.0
    )
    print("\n  Retry delays with exponential backoff:")
    for i in range(6):
        delay = config.get_delay(i)
        bar = "#" * int(delay * 3)
        print(f"    Attempt {i}: {delay:5.2f}s  {bar}")

    # --- Demo 3: API Client with Parallel Calls ---
    print("\n" + "=" * 55)
    print("  DEMO 3: Agent API Client (Parallel + Retry)")
    print("=" * 55)

    client = AgentAPIClient(
        base_url="https://api.example.com",
        api_key="sk-agent-key-***",
        timeout_ms=5000,
    )

    # Single call
    print("\n  Single API call:")
    try:
        resp = await client.get("/search", {"q": "LangGraph"})
        print(f"    Status: {resp.status_code} "
              f"({resp.status_name})")
        print(f"    Data: {resp.data}")
        print(f"    Latency: {resp.elapsed_ms:.0f}ms")
    except RetryableError as e:
        print(f"    Failed after retries: {e}")

    # Parallel calls — like an agent calling multiple APIs
    print("\n  Parallel API calls (5 endpoints at once):")
    endpoints = [
        "/search", "/analyze", "/summarize",
        "/fact-check", "/translate"
    ]
    try:
        responses = await client.get_many(endpoints)
        for ep, resp in zip(endpoints, responses):
            icon = "[+]" if resp.is_success else "[!]"
            print(f"    {icon} {ep}: {resp.status_code} "
                  f"({resp.elapsed_ms:.0f}ms)")
    except RetryableError:
        print("    Some calls failed after retry")

    print(f"\n  API Metrics: {client.metrics.report()}")

    # --- Demo 4: JSON Processing ---
    print("\n" + "=" * 55)
    print("  DEMO 4: JSON Processing (OpenAI Response)")
    print("=" * 55)

    processed = process_api_response({})
    print(f"\n  Extracted from OpenAI response:")
    print(f"    Model: {processed['model']}")
    print(f"    Tool calls: {processed['tool_calls']}")
    print(f"    Tokens: {processed['tokens_used']}")
    print(f"    Reason: {processed['finish_reason']}")

    # --- Demo 5: API Key Management ---
    print("\n" + "=" * 55)
    print("  DEMO 5: Secure API Key Management")
    print("=" * 55)

    config = SecureConfig()
    config.show_loaded()
    print("\n  .env file format:")
    print("    OPENAI_API_KEY=sk-proj-your-key-here")
    print("    ANTHROPIC_API_KEY=sk-ant-your-key-here")
    print("    # NEVER commit .env to git!")

    # --- Summary ---
    print(f"""
{'='*55}
  Day 5 — APIs & HTTP for Agents
{'='*55}

  [1] HTTP anatomy    = Every tool call is a request
  [2] Retry + backoff = Handle rate limits gracefully
  [3] Parallel calls  = asyncio.gather + API client
  [4] JSON processing = Extract tool calls from responses
  [5] API key mgmt    = .env files, never hardcode

  Agents speak HTTP.
  Master APIs → master agent communication.
{'='*55}
""")


if __name__ == "__main__":
    asyncio.run(main())
