"""
🤖 Day 09/150 — Testing: How Professionals Ship Code
======================================================
Week 2: Advanced Python + Dev Tools

Concepts:
  1. pytest — unit tests, fixtures, parametrize
  2. Mocking API calls — unittest.mock + MagicMock
  3. Testing async code — pytest-asyncio
  4. Test-Driven Development (TDD) mindset for agent tools

Why this matters for Agentic AI:
  - Agents call LLMs, external APIs, and tools — all side effects
  - Without mocking, tests cost money and are slow
  - Untested agents break silently in production
  - TDD forces you to design clean, testable tool interfaces
"""

import asyncio
import json
import time
import random
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from unittest.mock import MagicMock, AsyncMock, patch, call
from enum import Enum

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ============================================================
# SECTION 1: The Agent Tools We Will Test
# ============================================================
# These are the functions/classes we write BEFORE the tests.
# In TDD, you write tests first — but here we show both together.

class ToolError(Exception):
    """Base tool error with context."""
    def __init__(self, message: str, tool: str, context: Dict = None):
        super().__init__(message)
        self.tool = tool
        self.context = context or {}


@dataclass
class SearchResult:
    """Structured result from web search tool."""
    query: str
    results: List[Dict[str, str]]
    total_found: int
    latency_ms: float
    source: str = "web"

    def top_result(self) -> Optional[Dict]:
        return self.results[0] if self.results else None

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "total_found": self.total_found,
            "latency_ms": self.latency_ms,
            "top_title": self.top_result().get("title") if self.top_result() else None
        }


@dataclass
class LLMResponse:
    """Structured LLM API response."""
    content: str
    model: str
    tokens_used: int
    finish_reason: str = "stop"

    @property
    def is_complete(self) -> bool:
        return self.finish_reason == "stop"

    @property
    def word_count(self) -> int:
        return len(self.content.split())


class WebSearchTool:
    """Agent tool: searches the web via API."""

    def __init__(self, api_key: str, max_results: int = 5):
        if not api_key or (isinstance(api_key, str) and not api_key.strip()):
            raise ToolError("API key required", tool="web_search")
        self.api_key = api_key
        self.max_results = max_results
        self._call_count = 0

    async def search(self, query: str) -> SearchResult:
        """Performs web search. Makes real HTTP call in production."""
        if not query or len(query.strip()) < 2:
            raise ToolError(
                f"Query too short: '{query}'",
                tool="web_search",
                context={"query": query}
            )

        self._call_count += 1
        start = time.time()

        # In real code: response = await httpx.get(url, params=...)
        # In tests: this is MOCKED so we never hit the real API
        raw = await self._http_get(
            "https://api.search.example.com/v1/search",
            params={"q": query, "key": self.api_key, "num": self.max_results}
        )

        latency = (time.time() - start) * 1000
        return SearchResult(
            query=query,
            results=raw.get("items", []),
            total_found=raw.get("total", 0),
            latency_ms=latency,
            source=raw.get("source", "web")
        )

    async def _http_get(self, url: str, params: Dict) -> Dict:
        """The actual HTTP call — replaced by mock in tests."""
        raise NotImplementedError("Real HTTP not implemented in demo")

    @property
    def call_count(self) -> int:
        return self._call_count


class LLMClient:
    """Agent tool: calls LLM API."""

    def __init__(self, model: str = "gpt-4o", api_key: str = "sk-test"):
        self.model = model
        self.api_key = api_key
        self._total_tokens = 0

    async def complete(self, prompt: str,
                       temperature: float = 0.7) -> LLMResponse:
        """Calls LLM completion API."""
        if not prompt:
            raise ToolError("Prompt cannot be empty", tool="llm_client")

        # In real code: response = await openai.chat.completions.create(...)
        raw = await self._call_api({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature
        })

        response = LLMResponse(
            content=raw["choices"][0]["message"]["content"],
            model=raw["model"],
            tokens_used=raw["usage"]["total_tokens"],
            finish_reason=raw["choices"][0]["finish_reason"]
        )
        self._total_tokens += response.tokens_used
        return response

    async def _call_api(self, payload: Dict) -> Dict:
        """Real API call — mocked in tests."""
        raise NotImplementedError("Real API not implemented in demo")

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens


class ResearchAgent:
    """Agent that combines search + LLM to answer questions."""

    def __init__(self, search_tool: WebSearchTool, llm: LLMClient):
        self.search = search_tool
        self.llm = llm
        self.steps_taken: List[str] = []

    async def answer(self, question: str) -> str:
        """Full agent pipeline: search → synthesize → answer."""
        self.steps_taken = []

        # Step 1: Search for context
        self.steps_taken.append("searching")
        search_results = await self.search.search(question)

        # Step 2: Build context from results
        self.steps_taken.append("building_context")
        context = "\n".join([
            f"- {r.get('title', '')}: {r.get('snippet', '')}"
            for r in search_results.results[:3]
        ])

        # Step 3: Synthesize with LLM
        self.steps_taken.append("synthesizing")
        prompt = f"""Answer this question using the search results below.

Question: {question}

Search Results:
{context}

Provide a concise, accurate answer."""

        response = await self.llm.complete(prompt)
        self.steps_taken.append("done")
        return response.content


# ============================================================
# SECTION 2: Test Suite using pytest patterns
# ============================================================
# We simulate pytest here. In real usage: run `pytest test_day9.py -v`
# Tests follow Arrange → Act → Assert pattern (AAA)

class TestResult:
    """Simple test runner result tracker."""
    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[str] = []
        self.errors: List[str] = []

    def record(self, name: str, passed: bool, error: str = ""):
        if passed:
            self.passed.append(name)
            print(f"  ✅ PASSED  {name}")
        else:
            self.failed.append(name)
            print(f"  ❌ FAILED  {name}: {error}")

    @property
    def total(self): return len(self.passed) + len(self.failed)
    @property
    def success_rate(self): return len(self.passed) / max(self.total, 1) * 100


results = TestResult()


def test(func):
    """Simple decorator to mark and run a test function."""
    func._is_async_test = asyncio.iscoroutinefunction(func)
    def wrapper(*args, **kwargs):
        try:
            name = func.__name__.replace("_", " ").title()
            if func._is_async_test:
                asyncio.run(func(*args, **kwargs))
            else:
                func(*args, **kwargs)
            results.record(name, True)
        except AssertionError as e:
            results.record(name, False, str(e))
        except Exception as e:
            results.record(name, False, f"{type(e).__name__}: {e}")
    wrapper._is_async_test = func._is_async_test
    return wrapper


# ── GROUP 1: Unit Tests — SearchResult dataclass ──────────────

@test
def test_search_result_top_result_returns_first():
    """Arrange → Act → Assert"""
    # Arrange
    result = SearchResult(
        query="agentic AI",
        results=[
            {"title": "Best Article", "snippet": "About AI agents..."},
            {"title": "Second Article", "snippet": "More about agents..."}
        ],
        total_found=2,
        latency_ms=120.5
    )
    # Act
    top = result.top_result()
    # Assert
    assert top is not None
    assert top["title"] == "Best Article"


@test
def test_search_result_top_result_empty_returns_none():
    result = SearchResult(query="empty", results=[], total_found=0, latency_ms=50.0)
    assert result.top_result() is None


@test
def test_search_result_to_dict_structure():
    result = SearchResult(
        query="test query",
        results=[{"title": "Test", "snippet": "snippet"}],
        total_found=1,
        latency_ms=100.0
    )
    d = result.to_dict()
    assert "query" in d
    assert "total_found" in d
    assert "latency_ms" in d
    assert d["query"] == "test query"
    assert d["top_title"] == "Test"


@test
def test_llm_response_is_complete_true_for_stop():
    response = LLMResponse(content="Hello!", model="gpt-4o",
                           tokens_used=10, finish_reason="stop")
    assert response.is_complete is True


@test
def test_llm_response_is_complete_false_for_length():
    response = LLMResponse(content="Cut off...", model="gpt-4o",
                           tokens_used=500, finish_reason="length")
    assert response.is_complete is False


@test
def test_llm_response_word_count():
    response = LLMResponse(
        content="The quick brown fox jumps",
        model="gpt-4o",
        tokens_used=10
    )
    assert response.word_count == 5


# ── GROUP 2: Mocking API Calls ─────────────────────────────────

@test
async def test_web_search_calls_http_with_correct_params():
    """Mock _http_get to avoid real HTTP. Verify correct params."""
    tool = WebSearchTool(api_key="test-key-123", max_results=3)

    # MOCK: replace _http_get with a fake that returns controlled data
    mock_response = {
        "items": [{"title": "AI News", "snippet": "Latest AI..."}],
        "total": 1,
        "source": "web"
    }
    tool._http_get = AsyncMock(return_value=mock_response)

    # Act
    result = await tool.search("artificial intelligence")

    # Assert: correct data returned
    assert result.query == "artificial intelligence"
    assert result.total_found == 1
    assert result.results[0]["title"] == "AI News"

    # Assert: HTTP was called once with right params
    tool._http_get.assert_called_once()
    call_args = tool._http_get.call_args
    assert "artificial intelligence" in str(call_args)
    assert "test-key-123" in str(call_args)


@test
async def test_web_search_call_count_increments():
    """Verify the tool tracks how many searches were made."""
    tool = WebSearchTool(api_key="key-abc")
    tool._http_get = AsyncMock(return_value={
        "items": [], "total": 0, "source": "web"
    })

    assert tool.call_count == 0
    await tool.search("query one")
    await tool.search("query two")
    assert tool.call_count == 2


@test
async def test_llm_client_tracks_total_tokens():
    """Mock _call_api. Verify token tracking across multiple calls."""
    client = LLMClient(model="gpt-4o", api_key="sk-test")

    def make_response(tokens: int) -> Dict:
        return {
            "choices": [{"message": {"content": "Result"}, "finish_reason": "stop"}],
            "model": "gpt-4o",
            "usage": {"total_tokens": tokens}
        }

    client._call_api = AsyncMock(side_effect=[
        make_response(150),
        make_response(200),
    ])

    await client.complete("First prompt")
    await client.complete("Second prompt")

    assert client.total_tokens_used == 350


@test
async def test_llm_client_passes_correct_model_in_payload():
    """Verify the API payload includes the correct model name."""
    client = LLMClient(model="claude-3-5-sonnet", api_key="sk-test")
    client._call_api = AsyncMock(return_value={
        "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
        "model": "claude-3-5-sonnet",
        "usage": {"total_tokens": 50}
    })

    await client.complete("test prompt")

    called_payload = client._call_api.call_args[0][0]
    assert called_payload["model"] == "claude-3-5-sonnet"
    assert called_payload["messages"][0]["content"] == "test prompt"


# ── GROUP 3: Testing the Full Agent Pipeline ──────────────────

@test
async def test_research_agent_full_pipeline():
    """End-to-end agent test with all dependencies mocked."""
    # Arrange: create mocked tools
    mock_search = MagicMock(spec=WebSearchTool)
    mock_search.search = AsyncMock(return_value=SearchResult(
        query="what is agentic AI",
        results=[
            {"title": "Agentic AI Explained", "snippet": "AI that acts autonomously..."},
            {"title": "LLM Agents Guide", "snippet": "Agents use tools to..."}
        ],
        total_found=2,
        latency_ms=80.0
    ))

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.complete = AsyncMock(return_value=LLMResponse(
        content="Agentic AI refers to AI systems that can autonomously plan and execute tasks.",
        model="gpt-4o",
        tokens_used=45
    ))

    agent = ResearchAgent(search_tool=mock_search, llm=mock_llm)

    # Act
    answer = await agent.answer("what is agentic AI")

    # Assert: correct answer returned
    assert "Agentic AI" in answer or "agentic" in answer.lower()

    # Assert: pipeline steps were taken in order
    assert agent.steps_taken == ["searching", "building_context", "synthesizing", "done"]

    # Assert: search was called with the question
    mock_search.search.assert_called_once_with("what is agentic AI")

    # Assert: LLM was called once (not zero, not twice)
    mock_llm.complete.assert_called_once()


@test
async def test_research_agent_passes_search_context_to_llm():
    """Verify agent includes search results in the LLM prompt."""
    mock_search = MagicMock(spec=WebSearchTool)
    mock_search.search = AsyncMock(return_value=SearchResult(
        query="Python asyncio",
        results=[{"title": "Asyncio Tutorial", "snippet": "asyncio enables concurrency..."}],
        total_found=1,
        latency_ms=60.0
    ))

    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.complete = AsyncMock(return_value=LLMResponse(
        content="Asyncio allows concurrent programming.", model="gpt-4o", tokens_used=30
    ))

    agent = ResearchAgent(search_tool=mock_search, llm=mock_llm)
    await agent.answer("Python asyncio")

    # Assert: the prompt passed to LLM includes the search title
    prompt_used = mock_llm.complete.call_args[0][0]
    assert "Asyncio Tutorial" in prompt_used
    assert "asyncio enables concurrency" in prompt_used


# ── GROUP 4: Parametrize Pattern (equivalent to @pytest.mark.parametrize) ──

@test
def test_parametrize_tool_error_on_invalid_api_key():
    """Test multiple invalid API keys trigger ToolError — parametrize style."""
    invalid_keys = ["", None, "   "]
    errors_raised = 0

    for key in invalid_keys:
        try:
            WebSearchTool(api_key=key)
        except ToolError:
            errors_raised += 1
        except Exception:
            errors_raised += 1  # Any error means validation worked

    assert errors_raised == len(invalid_keys), \
        f"Expected {len(invalid_keys)} errors, got {errors_raised}"


@test
async def test_parametrize_short_queries_raise_error():
    """Test multiple short queries raise ToolError — parametrize style."""
    tool = WebSearchTool(api_key="valid-key")
    tool._http_get = AsyncMock(return_value={"items": [], "total": 0, "source": "web"})

    short_queries = ["", " ", "a"]
    errors_raised = 0

    for query in short_queries:
        try:
            await tool.search(query)
        except ToolError:
            errors_raised += 1

    assert errors_raised == len(short_queries)


# ============================================================
# SECTION 3: TDD Mindset Demonstration
# ============================================================
# TDD: Write the TEST first → see it fail → write code to pass it

def demonstrate_tdd_mindset():
    """Shows the TDD cycle: RED → GREEN → REFACTOR"""
    print("\n" + "─" * 64)
    print("  TDD MINDSET: RED → GREEN → REFACTOR")
    print("─" * 64)

    tdd_cycle = [
        {
            "phase": "🔴 RED",
            "description": "Write the test BEFORE the code",
            "code": '''
  # Step 1: Write this test first (it FAILS — no implementation yet)
  async def test_agent_caches_search_results():
      agent = ResearchAgent(search_tool, llm)
      await agent.answer("what is AI?")
      await agent.answer("what is AI?")  # same question
      # Cache means search called only ONCE, not twice
      mock_search.search.assert_called_once()  # FAILS — not built yet'''
        },
        {
            "phase": "🟢 GREEN",
            "description": "Write minimal code to make the test pass",
            "code": '''
  # Step 2: Add cache to ResearchAgent
  class ResearchAgent:
      def __init__(self, ...):
          self._cache: Dict[str, str] = {}  # Add cache

      async def answer(self, question: str) -> str:
          if question in self._cache:
              return self._cache[question]  # Return cached
          result = await self._pipeline(question)
          self._cache[question] = result
          return result  # Test PASSES now'''
        },
        {
            "phase": "🔵 REFACTOR",
            "description": "Clean up without breaking tests",
            "code": '''
  # Step 3: Extract TTL cache, add max_size, still passes test
  from functools import lru_cache

  @lru_cache(maxsize=100)
  async def answer(self, question: str) -> str:
      return await self._pipeline(question)
  # All tests still GREEN after refactor'''
        }
    ]

    for step in tdd_cycle:
        print(f"\n  {step['phase']}: {step['description']}")
        print(step['code'])


# ============================================================
# SECTION 4: Run Everything
# ============================================================

def run_all_tests():
    """Run all tests and print summary."""

    print("\n" + "=" * 64)
    print("  DAY 9: TESTING — HOW PROFESSIONALS SHIP CODE")
    print("  Week 2 — Advanced Python + Dev Tools")
    print("=" * 64)

    # All @test decorated functions — sync and async handled automatically
    test_functions = [
        test_search_result_top_result_returns_first,
        test_search_result_top_result_empty_returns_none,
        test_search_result_to_dict_structure,
        test_llm_response_is_complete_true_for_stop,
        test_llm_response_is_complete_false_for_length,
        test_llm_response_word_count,
        test_web_search_calls_http_with_correct_params,
        test_web_search_call_count_increments,
        test_llm_client_tracks_total_tokens,
        test_llm_client_passes_correct_model_in_payload,
        test_research_agent_full_pipeline,
        test_research_agent_passes_search_context_to_llm,
        test_parametrize_tool_error_on_invalid_api_key,
        test_parametrize_short_queries_raise_error,
    ]

    print(f"\n  Running {len(test_functions)} tests...\n")

    print("  GROUP 1: Unit Tests — Data Models")
    print("  " + "─" * 50)
    for fn in test_functions[:6]:
        fn()

    print("\n  GROUP 2: Mocking API Calls")
    print("  " + "─" * 50)
    for fn in test_functions[6:10]:
        fn()

    print("\n  GROUP 3: Full Agent Pipeline Tests")
    print("  " + "─" * 50)
    for fn in test_functions[10:12]:
        fn()

    print("\n  GROUP 4: Parametrize Pattern")
    print("  " + "─" * 50)
    for fn in test_functions[12:]:
        fn()

    # TDD Mindset
    demonstrate_tdd_mindset()

    # Summary
    print("\n" + "=" * 64)
    print("  TEST RESULTS SUMMARY")
    print("=" * 64)
    print(f"""
  Total Tests:      {results.total}
  Passed:           {len(results.passed)} ✅
  Failed:           {len(results.failed)} ❌
  Success Rate:     {results.success_rate:.0f}%

  Concepts Covered:
    pytest patterns      → Arrange-Act-Assert (AAA)
    unittest.mock        → MagicMock, AsyncMock, patch
    API call mocking     → No real HTTP, no real costs
    Full pipeline tests  → Agent steps verified end-to-end
    Parametrize style    → Multiple inputs, one test
    TDD cycle            → Red → Green → Refactor

  Key Insight:
    MagicMock(spec=WebSearchTool)  ← spec= ensures type safety
    AsyncMock(return_value=...)    ← for async functions
    mock.assert_called_once()      ← verify calls, not just results
    mock.call_args[0][0]           ← inspect exact arguments passed
""")
    print("=" * 64)
    print("  github.com/24A31A04N2/agentic-ai-journey/day-09")
    print("=" * 64)


if __name__ == "__main__":
    run_all_tests()

