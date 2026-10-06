# Day 17: Model Routing & Cost Engineering Basics

![Day](https://img.shields.io/badge/Day-17%2F250-blue)
![Phase](https://img.shields.io/badge/Phase-1-orange)
![Python](https://img.shields.io/badge/Python-3.11-blue)

> "Cost engineering is what separates a weekend demo from a production AI system."

## What I Learned Today

| Concept | Description |
|---------|-------------|
| **Model Registry & Cost Tables** | Built a central repository mapping LLMs to their real-world pricing and latency characteristics. |
| **Token Counting & Calculator** | Implemented a heuristic-based tokenizer and cost calculator to track spend per request. |
| **Intelligence-Based Routing** | Designed a router that directs simple tasks to cheap models and complex tasks to high-tier models. |
| **Fallback Chains & Resilience** | Implemented the Circuit Breaker pattern to gracefully handle provider outages. |
| **Prompt Caching & Batching** | Built LRU-based prompt caching and simulated asynchronous batch processing for ~50% cost savings. |
| **Budget Management & Dashboards** | Created a financial control layer to enforce daily/monthly budgets and prevent runaway agent loops. |

## What I Built

I implemented a comprehensive Python module demonstrating the core mechanics of LLM cost engineering and orchestration resilience.

The code features 6 distinct sections:
1. `ModelRegistry` class mapping `ModelSpec`s across OpenAI, Anthropic, Google, and Meta.
2. `TokenizerApproximator` and `CostCalculator` for real-time spend tracking.
3. `ComplexityClassifier` and `ModelRouter` for intelligent dispatch.
4. `CircuitBreaker` and `FallbackChain` to simulate and handle network partitions.
5. `PromptCache` and `BatchProcessor` to optimize repeated queries.
6. `BudgetManager` and `CostDashboard` for executive-level oversight.

## Code Highlights

### Circuit Breaker Pattern for Provider Resilience
```python
class CircuitBreaker:
    def is_open(self, provider: str) -> bool:
        if self.failures.get(provider, 0) >= self.failure_threshold:
            time_since_failure = time.time() - self.last_failure_time.get(provider, 0)
            if time_since_failure > self.reset_timeout_sec:
                self.failures[provider] = 0
                return False
            return True
        return False
```

### Intelligence-Based Routing
```python
def route(self, prompt: str) -> ModelSpec:
    complexity = ComplexityClassifier.classify(prompt)
    min_tier = self.routing_rules[complexity]
    eligible_models = [m for m in self.registry.models.values() if m.intelligence_tier >= min_tier]
    eligible_models.sort(key=lambda m: (m.cost_per_1k_input * 3) + m.cost_per_1k_output)
    return eligible_models[0]
```

## Run It

To execute the simulations and view the console dashboards:

```bash
python day17_model_routing_cost.py
```

## Why This Matters for Agents

In Agentic AI workflows, an agent might autonomously issue dozens or hundreds of LLM calls to complete a single user objective. Without routing, calling a flagship model like GPT-4o for trivial formatting tasks burns budget rapidly. Additionally, provider APIs occasionally fail or enforce strict rate limits; a resilient agent needs built-in fallback mechanisms to maintain autonomy. Cost engineering is fundamentally an architecture problem.

## Resources

| Resource | Link |
|----------|------|
| LiteLLM Documentation | [https://docs.litellm.ai](https://docs.litellm.ai) |
| OpenAI Pricing | [https://openai.com/pricing](https://openai.com/pricing) |
| Circuit Breaker Pattern | [Martin Fowler - Circuit Breaker](https://martinfowler.com/bliki/CircuitBreaker.html) |
