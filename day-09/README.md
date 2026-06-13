# 🤖 Day 09/150 — Testing: How Professionals Ship Code

![Day](https://img.shields.io/badge/Day-09%2F150-blue?style=flat-square)
![Phase](https://img.shields.io/badge/Phase-2%3A%20Advanced%20Python-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11-3572a5?style=flat-square&logo=python)
![Tests](https://img.shields.io/badge/Tests-14%2F14%20Passing-3fb950?style=flat-square)

> **Key Insight:** Agents call LLMs and external APIs — all side effects.  
> Without mocking, tests cost money. Without tests, agents break silently.

---

## 📌 What I Learned Today

| Concept | What It Does | Agent Application |
|---------|-------------|-------------------|
| **pytest / AAA Pattern** | Arrange → Act → Assert | Structure every test cleanly |
| **MagicMock(spec=...)** | Mock any object with type safety | Fake tools without calling real APIs |
| **AsyncMock** | Mock async functions | Test async LLM/tool calls |
| **assert_called_once()** | Verify mock was called exactly once | Ensure agent doesn't double-call APIs |
| **call_args** | Inspect the exact arguments passed | Verify prompts sent to LLM |
| **TDD Mindset** | Write test FIRST, then code | Forces clean, testable agent design |

---

## 🔨 What I Built

### Agent Tools (Under Test)
```
WebSearchTool   → async search() + _http_get() (mockable)
LLMClient       → async complete() + _call_api() (mockable)
ResearchAgent   → orchestrates search → synthesize → answer
SearchResult    → structured dataclass with validation
LLMResponse     → structured response with is_complete, word_count
```

### Test Suite — 14 Tests, 4 Groups
```
GROUP 1: Unit Tests — Data Models (6 tests)
  ✅ SearchResult.top_result() returns first item
  ✅ SearchResult.top_result() returns None on empty
  ✅ SearchResult.to_dict() has correct structure
  ✅ LLMResponse.is_complete True for "stop"
  ✅ LLMResponse.is_complete False for "length"
  ✅ LLMResponse.word_count correct

GROUP 2: Mocking API Calls (4 tests)
  ✅ WebSearch calls HTTP with correct params
  ✅ WebSearch call_count increments per search
  ✅ LLMClient tracks total tokens across calls
  ✅ LLMClient passes correct model in payload

GROUP 3: Full Agent Pipeline (2 tests)
  ✅ ResearchAgent executes steps in correct order
  ✅ ResearchAgent includes search results in LLM prompt

GROUP 4: Parametrize Style (2 tests)
  ✅ All invalid API keys raise ToolError
  ✅ All short queries raise ToolError
```

### TDD Cycle Demonstrated
```
🔴 RED     → Write test first (it fails — no code yet)
🟢 GREEN   → Write minimal code to make it pass
🔵 REFACTOR → Clean up, all tests still green
```

---

## 📂 Files

| File | Description |
|------|-------------|
| [`day9_testing.py`](./day9_testing.py) | Full code — 14 tests, 4 groups, TDD demo |

---

## ▶️ Run It

```bash
cd agentic-ai-journey/day-09
python day9_testing.py
```

To run with real pytest (install first):
```bash
pip install pytest pytest-asyncio
pytest day9_testing.py -v
```

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| pytest docs | https://docs.pytest.org/ |
| unittest.mock | https://docs.python.org/3/library/unittest.mock.html |
| pytest-asyncio | https://pytest-asyncio.readthedocs.io/ |
| TDD in Python | https://testdriven.io/blog/modern-tdd/ |

---

## 🗓️ What's Next

**Day 10:** Git Mastery & Professional Workflow  
→ Branching strategies, conventional commits, GitHub Actions CI/CD

---

*Part of the [150-Day Agentic AI Mastery Roadmap](../README.md)*
