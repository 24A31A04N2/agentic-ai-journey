import sys
import time
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum, auto

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

"""
🤖 Day 17/250 — Model Routing & Cost Engineering Basics
=====================================================
Phase 1: Software Engineering + AI Foundations

Concepts:
  1. Model Registry & Cost Tables — Real-world LLM pricing across providers
  2. Token Counting & Cost Calculator — Per-request cost tracking
  3. Intelligence-Based Routing — Simple tasks → cheap models, complex → expensive
  4. Fallback Chains & Resilience — Provider outages, circuit breakers
  5. Prompt Caching & Batch APIs — Cost reduction strategies
  6. Budget Management & Cost Dashboard — Spend tracking with alerts

Why this matters for Agentic AI:
  - A single GPT-4o request costs 100x more than GPT-4o-mini
  - Without routing, agents burn budget on trivial tasks
  - Provider outages are inevitable — fallback chains keep agents running
  - Caching repeated context saves 30-70% on real workloads
  - Budget management prevents runaway costs in autonomous agents
  - Cost engineering is what separates a demo from a production system
"""

# ==========================================
# Section 1: Model Registry & Cost Tables
# ==========================================

@dataclass
class ModelSpec:
    """Defines a language model and its characteristics."""
    name: str
    provider: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    max_tokens: int
    latency_ms_avg: int
    intelligence_tier: int  # 1 (low) to 4 (flagship)

class ModelRegistry:
    """Central repository for model pricing and specs."""
    
    def __init__(self):
        self.models: Dict[str, ModelSpec] = {}
        self._initialize_defaults()
        
    def _initialize_defaults(self):
        # OpenAI Models
        self.register(ModelSpec(
            name="gpt-4o", provider="openai", cost_per_1k_input=0.005, 
            cost_per_1k_output=0.015, max_tokens=128000, latency_ms_avg=800, intelligence_tier=4
        ))
        self.register(ModelSpec(
            name="gpt-4o-mini", provider="openai", cost_per_1k_input=0.00015, 
            cost_per_1k_output=0.0006, max_tokens=128000, latency_ms_avg=400, intelligence_tier=2
        ))
        
        # Anthropic Models
        self.register(ModelSpec(
            name="claude-3-5-sonnet", provider="anthropic", cost_per_1k_input=0.003, 
            cost_per_1k_output=0.015, max_tokens=200000, latency_ms_avg=900, intelligence_tier=4
        ))
        self.register(ModelSpec(
            name="claude-3-haiku", provider="anthropic", cost_per_1k_input=0.00025, 
            cost_per_1k_output=0.00125, max_tokens=200000, latency_ms_avg=300, intelligence_tier=2
        ))
        
        # Google Models
        self.register(ModelSpec(
            name="gemini-1.5-pro", provider="google", cost_per_1k_input=0.0035, 
            cost_per_1k_output=0.0105, max_tokens=2000000, latency_ms_avg=1000, intelligence_tier=4
        ))
        self.register(ModelSpec(
            name="gemini-1.5-flash", provider="google", cost_per_1k_input=0.000075, 
            cost_per_1k_output=0.0003, max_tokens=1000000, latency_ms_avg=350, intelligence_tier=1
        ))
        
        # Meta/Open Source Models
        self.register(ModelSpec(
            name="llama-3.1-70b", provider="together", cost_per_1k_input=0.0008, 
            cost_per_1k_output=0.0008, max_tokens=128000, latency_ms_avg=600, intelligence_tier=3
        ))
        self.register(ModelSpec(
            name="llama-3.1-8b", provider="together", cost_per_1k_input=0.00018, 
            cost_per_1k_output=0.00018, max_tokens=128000, latency_ms_avg=200, intelligence_tier=1
        ))
        
    def register(self, spec: ModelSpec):
        self.models[spec.name] = spec
        
    def get_model(self, name: str) -> Optional[ModelSpec]:
        return self.models.get(name)

    def print_registry(self):
        print("\n--- Model Registry & Cost Tables ---")
        print(f"{'Model Name':<20} | {'Provider':<10} | {'In/1K':<8} | {'Out/1K':<8} | {'Tier'}")
        print("-" * 65)
        for name, spec in sorted(self.models.items(), key=lambda x: x[1].intelligence_tier, reverse=True):
            print(f"{name:<20} | {spec.provider:<10} | ${spec.cost_per_1k_input:<7.5f} | ${spec.cost_per_1k_output:<7.5f} | {spec.intelligence_tier}")

def demonstrate_model_registry():
    registry = ModelRegistry()
    registry.print_registry()
    
    print("\nInsight: The cost differential is massive.")
    gpt4o = registry.get_model("gpt-4o")
    flash = registry.get_model("gemini-1.5-flash")
    if gpt4o and flash:
        ratio = gpt4o.cost_per_1k_input / flash.cost_per_1k_input
        print(f"GPT-4o input is {ratio:.1f}x more expensive than Gemini 1.5 Flash.")

# ==========================================
# Section 2: Token Counting & Cost Calculator
# ==========================================

class TokenizerApproximator:
    """Approximates token counts using a simple word-based heuristic."""
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Rough rule of thumb for English: 1 token ~= 4 characters or 0.75 words.
        """
        if not text:
            return 0
        words = len(text.split())
        chars = len(text)
        # Average of char-based and word-based estimation
        return int(((chars / 4.0) + (words / 0.75)) / 2)

@dataclass
class CostRecord:
    timestamp: float
    model_name: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float

class CostCalculator:
    """Calculates and tracks costs across multiple requests."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.history: List[CostRecord] = []
        
    def calculate_cost(self, model_name: str, input_text: str, output_text: str) -> CostRecord:
        model = self.registry.get_model(model_name)
        if not model:
            raise ValueError(f"Unknown model: {model_name}")
            
        input_tokens = TokenizerApproximator.count_tokens(input_text)
        output_tokens = TokenizerApproximator.count_tokens(output_text)
        
        input_cost = (input_tokens / 1000.0) * model.cost_per_1k_input
        output_cost = (output_tokens / 1000.0) * model.cost_per_1k_output
        total_cost = input_cost + output_cost
        
        record = CostRecord(
            timestamp=time.time(),
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost
        )
        self.history.append(record)
        return record
        
    def get_total_spend(self) -> float:
        return sum(record.total_cost for record in self.history)
        
    def print_history(self):
        print("\n--- Cost Calculator History ---")
        for idx, rec in enumerate(self.history):
            print(f"Req {idx+1}: [{rec.model_name}] In: {rec.input_tokens}t Out: {rec.output_tokens}t -> Total: ${rec.total_cost:.6f}")
        print(f"Total Spend: ${self.get_total_spend():.6f}")

def demonstrate_cost_calculator():
    registry = ModelRegistry()
    calculator = CostCalculator(registry)
    
    prompt = "Summarize the history of the Roman Empire in detail." * 10
    response = "The Roman Empire was a post-Republican period of ancient Rome. " * 50
    
    calculator.calculate_cost("gpt-4o", prompt, response)
    calculator.calculate_cost("gpt-4o-mini", prompt, response)
    calculator.calculate_cost("claude-3-haiku", prompt, response)
    
    calculator.print_history()

# ==========================================
# Section 3: Model Router — Intelligence-Based Routing
# ==========================================

class TaskComplexity(Enum):
    SIMPLE = auto()      # Formatting, basic extraction, translation
    MODERATE = auto()    # Summarization, simple logic, API mapping
    COMPLEX = auto()     # Planning, complex reasoning, coding
    CRITICAL = auto()    # High-stakes decision making, final review

class ComplexityClassifier:
    """Analyzes a prompt to determine the required task complexity."""
    
    COMPLEX_KEYWORDS = ['plan', 'architect', 'analyze', 'synthesize', 'debug', 'reason', 'evaluate']
    CRITICAL_KEYWORDS = ['final decision', 'financial', 'medical', 'security', 'approve']
    
    @classmethod
    def classify(cls, prompt: str) -> TaskComplexity:
        prompt_lower = prompt.lower()
        token_count = TokenizerApproximator.count_tokens(prompt)
        
        # Check critical first
        if any(kw in prompt_lower for kw in cls.CRITICAL_KEYWORDS):
            return TaskComplexity.CRITICAL
            
        # Check complex
        if token_count > 5000 or any(kw in prompt_lower for kw in cls.COMPLEX_KEYWORDS):
            return TaskComplexity.COMPLEX
            
        # Moderate
        if token_count > 1000 or 'summarize' in prompt_lower:
            return TaskComplexity.MODERATE
            
        return TaskComplexity.SIMPLE

class ModelRouter:
    """Routes tasks to the most cost-effective model that can handle them."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        # Mapping complexity to minimum required intelligence tier
        self.routing_rules = {
            TaskComplexity.SIMPLE: 1,
            TaskComplexity.MODERATE: 2,
            TaskComplexity.COMPLEX: 3,
            TaskComplexity.CRITICAL: 4
        }
        
    def route(self, prompt: str) -> ModelSpec:
        complexity = ComplexityClassifier.classify(prompt)
        min_tier = self.routing_rules[complexity]
        
        # Find all models meeting the minimum tier
        eligible_models = [m for m in self.registry.models.values() if m.intelligence_tier >= min_tier]
        
        # Sort by total blended cost (assuming roughly 3:1 input:output ratio)
        eligible_models.sort(key=lambda m: (m.cost_per_1k_input * 3) + m.cost_per_1k_output)
        
        # Select the cheapest eligible model
        selected = eligible_models[0]
        
        print(f"Routing Prompt ('{prompt[:30]}...'):")
        print(f"  Complexity: {complexity.name}")
        print(f"  Selected Model: {selected.name} (Tier {selected.intelligence_tier}, Provider: {selected.provider})")
        return selected

def demonstrate_model_routing():
    registry = ModelRegistry()
    router = ModelRouter(registry)
    
    print("\n--- Intelligence-Based Routing ---")
    prompts = [
        "Fix the formatting of this list: apples, bananas, oranges.",
        "Summarize this 10-page article about climate change.",
        "Architect a multi-region distributed database system for high availability.",
        "Review this financial report and make a final decision on the investment."
    ]
    
    for p in prompts:
        router.route(p)

# ==========================================
# Section 4: Fallback Chains & Provider Resilience
# ==========================================

class CircuitBreaker:
    """Prevents continuous requests to a failing provider."""
    
    def __init__(self, failure_threshold: int = 3, reset_timeout_sec: int = 10):
        self.failure_threshold = failure_threshold
        self.reset_timeout_sec = reset_timeout_sec
        self.failures: Dict[str, int] = {}
        self.last_failure_time: Dict[str, float] = {}
        
    def record_failure(self, provider: str):
        self.failures[provider] = self.failures.get(provider, 0) + 1
        self.last_failure_time[provider] = time.time()
        
    def record_success(self, provider: str):
        self.failures[provider] = 0
        
    def is_open(self, provider: str) -> bool:
        """Returns True if the circuit is open (provider is blocked)."""
        if self.failures.get(provider, 0) >= self.failure_threshold:
            time_since_failure = time.time() - self.last_failure_time.get(provider, 0)
            if time_since_failure > self.reset_timeout_sec:
                print(f"[CircuitBreaker] Resetting circuit for {provider}")
                self.failures[provider] = 0
                return False
            return True
        return False

class FallbackChain:
    """Executes a chain of models, falling back on failure."""
    
    def __init__(self, models: List[str], registry: ModelRegistry, circuit_breaker: CircuitBreaker):
        self.models = models
        self.registry = registry
        self.circuit_breaker = circuit_breaker
        # Simulate network state
        self.simulated_outages = set()
        
    def set_outage(self, provider: str):
        self.simulated_outages.add(provider)
        
    def resolve_outage(self, provider: str):
        if provider in self.simulated_outages:
            self.simulated_outages.remove(provider)
            
    def execute(self, prompt: str) -> str:
        for model_name in self.models:
            model = self.registry.get_model(model_name)
            if not model:
                continue
                
            provider = model.provider
            
            if self.circuit_breaker.is_open(provider):
                print(f"  [Fallback] Skipping {model_name} - Provider '{provider}' circuit is OPEN.")
                continue
                
            print(f"  [Fallback] Attempting {model_name} (Provider: {provider})...", end="")
            
            # Simulate request
            time.sleep(0.1) 
            
            if provider in self.simulated_outages:
                print(" FAILED (Simulated Outage)")
                self.circuit_breaker.record_failure(provider)
                continue
                
            # Success
            print(" SUCCESS")
            self.circuit_breaker.record_success(provider)
            return f"Response from {model_name}"
            
        raise Exception("All fallback models failed!")

def demonstrate_fallback_chains():
    print("\n--- Fallback Chains & Provider Resilience ---")
    registry = ModelRegistry()
    breaker = CircuitBreaker(failure_threshold=2, reset_timeout_sec=2)
    chain = FallbackChain(["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"], registry, breaker)
    
    print("Normal execution:")
    chain.execute("Hello")
    
    print("\nSimulating OpenAI outage...")
    chain.set_outage("openai")
    chain.execute("Hello")
    
    print("\nSimulating OpenAI and Anthropic outages...")
    chain.set_outage("anthropic")
    chain.execute("Hello")
    
    print("\nTriggering circuit breaker on OpenAI (2nd failure)...")
    chain.execute("Hello")
    
    print("\nAttempting while circuit is open...")
    chain.execute("Hello")
    
    print("\nWaiting for circuit reset...")
    time.sleep(2.1)
    chain.resolve_outage("openai")
    chain.execute("Hello")

# ==========================================
# Section 5: Prompt Caching & Batch API Simulation
# ==========================================

@dataclass
class CacheEntry:
    response: str
    expiry_time: float
    access_count: int = 0

class PromptCache:
    """LRU-style cache with TTL to save costs on repeated prompts."""
    
    def __init__(self, max_size: int = 100, ttl_sec: int = 3600):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.ttl_sec = ttl_sec
        self.hits = 0
        self.misses = 0
        
    def _generate_key(self, model: str, prompt: str) -> str:
        # In reality, hash the prompt
        return f"{model}:{hash(prompt)}"
        
    def get(self, model: str, prompt: str) -> Optional[str]:
        key = self._generate_key(model, prompt)
        entry = self.cache.get(key)
        
        if entry:
            if time.time() > entry.expiry_time:
                del self.cache[key]
                self.misses += 1
                return None
            
            entry.access_count += 1
            self.hits += 1
            return entry.response
            
        self.misses += 1
        return None
        
    def set(self, model: str, prompt: str, response: str):
        if len(self.cache) >= self.max_size:
            # Evict least accessed
            lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
            del self.cache[lru_key]
            
        key = self._generate_key(model, prompt)
        self.cache[key] = CacheEntry(
            response=response,
            expiry_time=time.time() + self.ttl_sec
        )
        
    def get_hit_ratio(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0

class BatchProcessor:
    """Simulates offline batch processing which typically offers 50% discount."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.queue = []
        self.discount_rate = 0.5 # 50% off for batch
        
    def submit_job(self, model: str, prompt: str):
        self.queue.append({"model": model, "prompt": prompt})
        
    def process_all(self, calculator: CostCalculator):
        print(f"\nProcessing {len(self.queue)} batch jobs...")
        saved_cost = 0.0
        
        for job in self.queue:
            model_spec = self.registry.get_model(job["model"])
            if not model_spec:
                continue
                
            # Temporarily adjust registry prices for this calculation
            orig_in = model_spec.cost_per_1k_input
            orig_out = model_spec.cost_per_1k_output
            
            model_spec.cost_per_1k_input *= self.discount_rate
            model_spec.cost_per_1k_output *= self.discount_rate
            
            # Calculate cost
            response = "Batch response simulation"
            record = calculator.calculate_cost(job["model"], job["prompt"], response)
            
            # Restore prices
            model_spec.cost_per_1k_input = orig_in
            model_spec.cost_per_1k_output = orig_out
            
            normal_cost = record.total_cost / self.discount_rate
            saved_cost += (normal_cost - record.total_cost)
            
        self.queue.clear()
        print(f"Batch processing complete. Saved ${saved_cost:.6f}")

def demonstrate_caching_and_batching():
    print("\n--- Prompt Caching & Batch API Simulation ---")
    cache = PromptCache()
    
    prompt = "What is the capital of France?"
    
    # First request
    print("Request 1:")
    res = cache.get("gpt-4o-mini", prompt)
    if not res:
        print("  Cache MISS. Generating...")
        res = "Paris"
        cache.set("gpt-4o-mini", prompt, res)
        
    # Second request
    print("Request 2:")
    res2 = cache.get("gpt-4o-mini", prompt)
    if res2:
        print(f"  Cache HIT. Response: {res2}")
        
    print(f"Cache Hit Ratio: {cache.get_hit_ratio()*100:.1f}%")
    
    # Batch processing
    registry = ModelRegistry()
    calc = CostCalculator(registry)
    batch = BatchProcessor(registry)
    
    for i in range(5):
        batch.submit_job("gpt-4o", f"Analyze dataset chunk {i}")
        
    batch.process_all(calc)

# ==========================================
# Section 6: Cost Dashboard & Budget Management
# ==========================================

class BudgetAlert(Exception):
    pass

class BudgetManager:
    """Tracks spending against limits and triggers alerts."""
    
    def __init__(self, daily_limit: float, monthly_limit: float):
        self.daily_limit = daily_limit
        self.monthly_limit = monthly_limit
        self.current_daily_spend = 0.0
        self.current_monthly_spend = 0.0
        
    def record_spend(self, amount: float):
        self.current_daily_spend += amount
        self.current_monthly_spend += amount
        self._check_limits()
        
    def _check_limits(self):
        if self.current_daily_spend > self.daily_limit:
            raise BudgetAlert(f"DAILY BUDGET EXCEEDED! Spend: ${self.current_daily_spend:.2f} > Limit: ${self.daily_limit:.2f}")
        if self.current_monthly_spend > self.monthly_limit:
            raise BudgetAlert(f"MONTHLY BUDGET EXCEEDED! Spend: ${self.current_monthly_spend:.2f} > Limit: ${self.monthly_limit:.2f}")
            
    def get_status(self) -> str:
        d_pct = (self.current_daily_spend / self.daily_limit) * 100
        m_pct = (self.current_monthly_spend / self.monthly_limit) * 100
        return f"Daily: ${self.current_daily_spend:.2f}/${self.daily_limit:.2f} ({d_pct:.1f}%) | Monthly: ${self.current_monthly_spend:.2f}/${self.monthly_limit:.2f} ({m_pct:.1f}%)"

class CostDashboard:
    """Generates visual reports of spending."""
    
    def __init__(self, calculator: CostCalculator):
        self.calculator = calculator
        
    def generate_report(self):
        print("\n=== EXECUTIVE COST DASHBOARD ===")
        spend_by_model: Dict[str, float] = {}
        tokens_by_model: Dict[str, int] = {}
        
        for record in self.calculator.history:
            spend_by_model[record.model_name] = spend_by_model.get(record.model_name, 0.0) + record.total_cost
            tokens = record.input_tokens + record.output_tokens
            tokens_by_model[record.model_name] = tokens_by_model.get(record.model_name, 0) + tokens
            
        print(f"Total Spend: ${self.calculator.get_total_spend():.6f}")
        print("\nSpend by Model:")
        for model, spend in sorted(spend_by_model.items(), key=lambda x: x[1], reverse=True):
            tokens = tokens_by_model[model]
            print(f"  - {model:<20}: ${spend:.6f} ({tokens} tokens)")
        print("================================")

def demonstrate_cost_dashboard():
    print("\n--- Budget Management & Dashboard ---")
    registry = ModelRegistry()
    calc = CostCalculator(registry)
    budget = BudgetManager(daily_limit=0.05, monthly_limit=1.0)
    dash = CostDashboard(calc)
    
    # Simulate workloads
    try:
        print("Simulating workloads...")
        for i in range(3):
            record = calc.calculate_cost("gpt-4o", "Analyze this large document" * 100, "Here is the comprehensive analysis" * 50)
            budget.record_spend(record.total_cost)
            print(f"  Job {i+1} OK. Status: {budget.get_status()}")
            
        # This one should trip the daily budget
        print("Attempting large job...")
        record = calc.calculate_cost("gpt-4o", "Analyze EVERYTHING" * 1000, "Huge output" * 500)
        budget.record_spend(record.total_cost)
    except BudgetAlert as e:
        print(f"\n🚨 ALERT: {e}")
        
    dash.generate_report()

# ==========================================
# Main Execution
# ==========================================

def main():
    print("Starting Day 17 Demonstrations...")
    demonstrate_model_registry()
    demonstrate_cost_calculator()
    demonstrate_model_routing()
    demonstrate_fallback_chains()
    demonstrate_caching_and_batching()
    demonstrate_cost_dashboard()
    print("\nDay 17 Demonstrations Complete!")

if __name__ == "__main__":
    main()
