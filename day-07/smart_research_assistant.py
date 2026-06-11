"""
🤖 Day 07/150 — Smart Research Assistant CLI
=============================================
Week 1 Integration Project: Combines ALL concepts from Days 1-6

Day 1: Python fundamentals, type hints, modern syntax
Day 2: Data structures (graphs, priority queues, hash maps)
Day 3: OOP — Base classes, Strategy pattern, Observer pattern
Day 4: Async Python — asyncio.gather, streaming, parallel execution
Day 5: APIs & HTTP — retry logic, exponential backoff, JSON parsing
Day 6: FastAPI — Pydantic validation, structured I/O

This CLI tool:
  1. Takes a research topic as input
  2. Async fetches data from 3 different API sources simultaneously
  3. Processes & structures results using Pydantic models
  4. Outputs a beautifully formatted research summary
"""

import asyncio
import json
import time
import sys
import random
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Protocol
from enum import Enum
from datetime import datetime

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ============================================================
# SECTION 1: Pydantic-Style Validation Models (Day 6 — Pydantic)
# ============================================================

class ValidationError(Exception):
    """Custom validation error for input models."""
    pass


class ResearchQuery:
    """Validates and structures the user's research input.
    Inspired by Pydantic BaseModel — manual validation with type hints."""

    def __init__(self, topic: str, max_sources: int = 3, depth: str = "standard"):
        # Validation rules (like Pydantic Field constraints)
        if not topic or len(topic.strip()) < 3:
            raise ValidationError(f"Topic must be at least 3 characters. Got: '{topic}'")
        if max_sources < 1 or max_sources > 10:
            raise ValidationError(f"max_sources must be 1-10. Got: {max_sources}")
        if depth not in ("quick", "standard", "deep"):
            raise ValidationError(f"depth must be 'quick', 'standard', or 'deep'. Got: '{depth}'")

        self.topic: str = topic.strip()
        self.max_sources: int = max_sources
        self.depth: str = depth
        self.timestamp: str = datetime.now().isoformat()
        self.query_id: str = hashlib.md5(f"{topic}{self.timestamp}".encode()).hexdigest()[:8]

    def __repr__(self) -> str:
        return f"ResearchQuery(topic='{self.topic}', sources={self.max_sources}, depth='{self.depth}')"


class SourceResult:
    """Structured result from a single API source."""

    def __init__(self, source_name: str, status: str, data: Dict[str, Any],
                 latency_ms: float, relevance_score: float):
        self.source_name = source_name
        self.status = status  # "success" | "error" | "timeout"
        self.data = data
        self.latency_ms = round(latency_ms, 2)
        self.relevance_score = min(max(relevance_score, 0.0), 1.0)  # Clamp 0-1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_name,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "relevance_score": self.relevance_score,
            "data": self.data
        }


class ResearchReport:
    """Final structured output combining all source results."""

    def __init__(self, query: ResearchQuery, results: List[SourceResult],
                 total_time_ms: float):
        self.query = query
        self.results = results
        self.total_time_ms = round(total_time_ms, 2)
        self.sources_succeeded = sum(1 for r in results if r.status == "success")
        self.sources_failed = sum(1 for r in results if r.status != "success")
        self.avg_relevance = (
            sum(r.relevance_score for r in results if r.status == "success")
            / max(self.sources_succeeded, 1)
        )


# ============================================================
# SECTION 2: Observer Pattern for Event Logging (Day 3 — OOP)
# ============================================================

class ResearchEvent(Enum):
    QUERY_RECEIVED = "query_received"
    FETCH_STARTED = "fetch_started"
    FETCH_COMPLETED = "fetch_completed"
    FETCH_FAILED = "fetch_failed"
    PROCESSING = "processing"
    REPORT_READY = "report_ready"


class EventObserver(ABC):
    """Observer interface — Day 3 Observer Pattern."""
    @abstractmethod
    def on_event(self, event: ResearchEvent, data: Dict[str, Any]) -> None:
        ...


class ConsoleLogger(EventObserver):
    """Logs research events to console with timestamps and colors."""

    ICONS = {
        ResearchEvent.QUERY_RECEIVED: "📥",
        ResearchEvent.FETCH_STARTED: "🔄",
        ResearchEvent.FETCH_COMPLETED: "✅",
        ResearchEvent.FETCH_FAILED: "❌",
        ResearchEvent.PROCESSING: "⚙️",
        ResearchEvent.REPORT_READY: "📊",
    }

    def on_event(self, event: ResearchEvent, data: Dict[str, Any]) -> None:
        icon = self.ICONS.get(event, "📌")
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        msg = data.get("message", "")
        print(f"  [{ts}] {icon} {event.value}: {msg}")


class MetricsCollector(EventObserver):
    """Collects performance metrics silently — Day 3 Observer Pattern."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def on_event(self, event: ResearchEvent, data: Dict[str, Any]) -> None:
        self.events.append({
            "event": event.value,
            "timestamp": time.time(),
            "data": data
        })

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_events": len(self.events),
            "event_types": list(set(e["event"] for e in self.events))
        }


class EventBus:
    """Central event dispatcher — supports multiple observers."""

    def __init__(self):
        self._observers: List[EventObserver] = []

    def subscribe(self, observer: EventObserver) -> None:
        self._observers.append(observer)

    def emit(self, event: ResearchEvent, data: Dict[str, Any]) -> None:
        for observer in self._observers:
            observer.on_event(event, data)


# ============================================================
# SECTION 3: Strategy Pattern for Data Sources (Day 3 — OOP)
# ============================================================

class DataSourceStrategy(ABC):
    """Abstract strategy for fetching data from different API sources.
    Day 3: Strategy Pattern — swap data sources without changing core logic."""

    @abstractmethod
    async def fetch(self, topic: str) -> SourceResult:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class WikipediaSource(DataSourceStrategy):
    """Simulates fetching from Wikipedia API."""

    @property
    def name(self) -> str:
        return "Wikipedia"

    async def fetch(self, topic: str) -> SourceResult:
        start = time.time()
        # Simulate API latency (Day 5 — HTTP concepts)
        await asyncio.sleep(random.uniform(0.3, 0.8))

        # Simulate occasional failures (Day 5 — error handling)
        if random.random() < 0.1:  # 10% failure rate
            latency = (time.time() - start) * 1000
            return SourceResult(
                source_name=self.name,
                status="error",
                data={"error": "API rate limit exceeded (429)"},
                latency_ms=latency,
                relevance_score=0.0
            )

        latency = (time.time() - start) * 1000
        return SourceResult(
            source_name=self.name,
            status="success",
            data={
                "title": f"{topic} — Wikipedia",
                "summary": f"Comprehensive encyclopedia article covering the fundamentals of {topic}, "
                           f"including history, key concepts, recent developments, and notable contributors.",
                "sections": ["Overview", "History", "Key Concepts", "Applications", "See Also"],
                "word_count": random.randint(2000, 8000),
                "last_edited": "2026-06-10"
            },
            latency_ms=latency,
            relevance_score=round(random.uniform(0.7, 0.95), 2)
        )


class ArxivSource(DataSourceStrategy):
    """Simulates fetching from arXiv academic papers API."""

    @property
    def name(self) -> str:
        return "arXiv"

    async def fetch(self, topic: str) -> SourceResult:
        start = time.time()
        await asyncio.sleep(random.uniform(0.4, 1.0))

        if random.random() < 0.05:  # 5% failure rate
            latency = (time.time() - start) * 1000
            return SourceResult(
                source_name=self.name,
                status="timeout",
                data={"error": "Connection timeout after 5000ms"},
                latency_ms=latency,
                relevance_score=0.0
            )

        latency = (time.time() - start) * 1000
        papers = [
            {
                "title": f"A Survey of {topic}: Methods and Applications",
                "authors": ["Chen et al.", "2026"],
                "abstract": f"We present a comprehensive survey of {topic} techniques...",
                "citations": random.randint(10, 500)
            },
            {
                "title": f"Advances in {topic} using Transformer Architectures",
                "authors": ["Kumar, Singh", "2026"],
                "abstract": f"This paper explores novel approaches to {topic}...",
                "citations": random.randint(5, 200)
            },
            {
                "title": f"{topic}: Benchmarks and Evaluation Frameworks",
                "authors": ["Wang et al.", "2025"],
                "abstract": f"We introduce new benchmarks for evaluating {topic}...",
                "citations": random.randint(20, 350)
            }
        ]

        return SourceResult(
            source_name=self.name,
            status="success",
            data={
                "total_results": len(papers),
                "papers": papers,
                "query_category": "cs.AI"
            },
            latency_ms=latency,
            relevance_score=round(random.uniform(0.8, 0.98), 2)
        )


class NewsAPISource(DataSourceStrategy):
    """Simulates fetching from a News aggregation API."""

    @property
    def name(self) -> str:
        return "NewsAPI"

    async def fetch(self, topic: str) -> SourceResult:
        start = time.time()
        await asyncio.sleep(random.uniform(0.2, 0.6))

        if random.random() < 0.08:  # 8% failure rate
            latency = (time.time() - start) * 1000
            return SourceResult(
                source_name=self.name,
                status="error",
                data={"error": "Invalid API key (401 Unauthorized)"},
                latency_ms=latency,
                relevance_score=0.0
            )

        latency = (time.time() - start) * 1000
        articles = [
            {
                "headline": f"Breaking: Major breakthrough in {topic} announced",
                "source": "TechCrunch",
                "published": "2026-06-10",
                "sentiment": "positive"
            },
            {
                "headline": f"Industry leaders discuss the future of {topic}",
                "source": "The Verge",
                "published": "2026-06-09",
                "sentiment": "neutral"
            }
        ]

        return SourceResult(
            source_name=self.name,
            status="success",
            data={
                "total_articles": len(articles),
                "articles": articles,
                "trending_score": round(random.uniform(60, 99), 1)
            },
            latency_ms=latency,
            relevance_score=round(random.uniform(0.6, 0.9), 2)
        )


# ============================================================
# SECTION 4: Retry Logic with Exponential Backoff (Day 5)
# ============================================================

@dataclass
class RetryConfig:
    """Configuration for exponential backoff — Day 5 concept."""
    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    backoff_factor: float = 2.0


async def fetch_with_retry(
    source: DataSourceStrategy,
    topic: str,
    config: RetryConfig,
    event_bus: EventBus
) -> SourceResult:
    """Wraps a data source fetch with exponential backoff retry logic.
    Day 5: Exponential backoff + jitter pattern."""

    for attempt in range(config.max_retries + 1):
        event_bus.emit(ResearchEvent.FETCH_STARTED, {
            "message": f"{source.name} — attempt {attempt + 1}/{config.max_retries + 1}"
        })

        result = await source.fetch(topic)

        if result.status == "success":
            event_bus.emit(ResearchEvent.FETCH_COMPLETED, {
                "message": f"{source.name} — {result.latency_ms}ms — relevance: {result.relevance_score}"
            })
            return result

        if attempt < config.max_retries:
            delay = min(
                config.base_delay * (config.backoff_factor ** attempt) + random.uniform(0, 0.5),
                config.max_delay
            )
            event_bus.emit(ResearchEvent.FETCH_FAILED, {
                "message": f"{source.name} — {result.status}: retrying in {delay:.1f}s..."
            })
            await asyncio.sleep(delay)

    event_bus.emit(ResearchEvent.FETCH_FAILED, {
        "message": f"{source.name} — ALL retries exhausted. Marking as failed."
    })
    return result


# ============================================================
# SECTION 5: Research Engine (Day 4 — Async Parallel Execution)
# ============================================================

class SmartResearchAssistant:
    """The core engine — orchestrates parallel async data fetching,
    processes results, and generates structured reports.

    Combines:
      - Strategy Pattern (Day 3) for swappable data sources
      - Observer Pattern (Day 3) for event logging
      - Async parallelism (Day 4) via asyncio.gather
      - Retry logic (Day 5) for resilient API calls
      - Pydantic-style validation (Day 6) for structured I/O
    """

    def __init__(self):
        self.sources: List[DataSourceStrategy] = [
            WikipediaSource(),
            ArxivSource(),
            NewsAPISource()
        ]
        self.retry_config = RetryConfig()
        self.event_bus = EventBus()

        # Attach observers (Day 3 — Observer Pattern)
        self.logger = ConsoleLogger()
        self.metrics = MetricsCollector()
        self.event_bus.subscribe(self.logger)
        self.event_bus.subscribe(self.metrics)

    async def research(self, query: ResearchQuery) -> ResearchReport:
        """Execute the full research pipeline."""

        self.event_bus.emit(ResearchEvent.QUERY_RECEIVED, {
            "message": f"Topic: '{query.topic}' | Depth: {query.depth} | ID: {query.query_id}"
        })

        start_time = time.time()

        # Day 4: Parallel async execution with asyncio.gather
        self.event_bus.emit(ResearchEvent.PROCESSING, {
            "message": f"Launching {len(self.sources)} API fetches in parallel..."
        })

        tasks = [
            fetch_with_retry(source, query.topic, self.retry_config, self.event_bus)
            for source in self.sources[:query.max_sources]
        ]

        results: List[SourceResult] = await asyncio.gather(*tasks)

        total_time = (time.time() - start_time) * 1000

        # Build structured report (Day 6 — Pydantic-style output)
        report = ResearchReport(
            query=query,
            results=results,
            total_time_ms=total_time
        )

        self.event_bus.emit(ResearchEvent.REPORT_READY, {
            "message": f"Done! {report.sources_succeeded}/{len(results)} sources succeeded in {total_time:.0f}ms"
        })

        return report


# ============================================================
# SECTION 6: Report Formatter (Beautiful CLI Output)
# ============================================================

def format_report(report: ResearchReport) -> str:
    """Formats the research report into a beautiful CLI output."""

    lines = []
    w = 64  # width

    lines.append("")
    lines.append("=" * w)
    lines.append("  SMART RESEARCH ASSISTANT — REPORT")
    lines.append("=" * w)
    lines.append(f"  Topic:     {report.query.topic}")
    lines.append(f"  Query ID:  {report.query.query_id}")
    lines.append(f"  Depth:     {report.query.depth}")
    lines.append(f"  Time:      {report.query.timestamp}")
    lines.append("-" * w)

    # Performance metrics
    lines.append(f"  Total Time:        {report.total_time_ms:.0f}ms")
    lines.append(f"  Sources OK:        {report.sources_succeeded}/{len(report.results)}")
    lines.append(f"  Sources Failed:    {report.sources_failed}/{len(report.results)}")
    lines.append(f"  Avg Relevance:     {report.avg_relevance:.0%}")
    lines.append("-" * w)

    # Latency comparison bar chart (Day 2 — Data visualization)
    lines.append("  LATENCY COMPARISON:")
    max_latency = max((r.latency_ms for r in report.results), default=1)
    for r in report.results:
        bar_len = int((r.latency_ms / max_latency) * 30) if max_latency > 0 else 0
        bar = "#" * bar_len
        status_icon = "[OK]" if r.status == "success" else "[FAIL]"
        lines.append(f"    {r.source_name:<12} {r.latency_ms:>7.0f}ms  {bar} {status_icon}")
    lines.append("-" * w)

    # Detailed results per source
    for i, result in enumerate(report.results, 1):
        lines.append(f"  SOURCE {i}: {result.source_name}")
        lines.append(f"    Status:    {result.status}")
        lines.append(f"    Latency:   {result.latency_ms}ms")
        lines.append(f"    Relevance: {result.relevance_score:.0%}")

        if result.status == "success":
            data = result.data
            if result.source_name == "Wikipedia":
                lines.append(f"    Title:     {data.get('title', 'N/A')}")
                lines.append(f"    Summary:   {data.get('summary', 'N/A')[:80]}...")
                lines.append(f"    Sections:  {', '.join(data.get('sections', []))}")
                lines.append(f"    Words:     {data.get('word_count', 'N/A')}")
            elif result.source_name == "arXiv":
                papers = data.get("papers", [])
                lines.append(f"    Papers Found: {data.get('total_results', 0)}")
                for p in papers:
                    lines.append(f"      - {p['title']} ({p['citations']} citations)")
            elif result.source_name == "NewsAPI":
                articles = data.get("articles", [])
                lines.append(f"    Trending Score: {data.get('trending_score', 'N/A')}")
                for a in articles:
                    lines.append(f"      - [{a['sentiment'].upper()}] {a['headline']}")
        else:
            lines.append(f"    Error:     {result.data.get('error', 'Unknown error')}")

        lines.append("")

    lines.append("=" * w)
    lines.append("  Week 1 Integration: Days 1-6 Combined")
    lines.append("  github.com/24A31A04N2/agentic-ai-journey")
    lines.append("=" * w)

    return "\n".join(lines)


# ============================================================
# SECTION 7: CLI Entry Point
# ============================================================

async def main():
    print("\n" + "=" * 64)
    print("  SMART RESEARCH ASSISTANT")
    print("  Week 1 Integration Project — Day 7/150")
    print("=" * 64)

    # Default topic for demo (can be overridden via CLI args)
    topic = "Agentic AI"
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])

    print(f"\n  Research topic: {topic}")
    print(f"  Fetching from 3 sources in parallel...\n")

    # Step 1: Validate input (Day 6 — Pydantic validation)
    try:
        query = ResearchQuery(topic=topic, max_sources=3, depth="standard")
    except ValidationError as e:
        print(f"  Validation Error: {e}")
        sys.exit(1)

    # Step 2: Run the async research engine
    assistant = SmartResearchAssistant()
    report = await assistant.research(query)

    # Step 3: Format and display the report
    formatted = format_report(report)
    print(formatted)

    # Step 4: Show metrics summary (Day 3 — Observer Pattern)
    metrics = assistant.metrics.get_summary()
    print(f"\n  Metrics: {metrics['total_events']} events tracked across {len(metrics['event_types'])} types")
    print(f"  Event types: {', '.join(metrics['event_types'])}")


if __name__ == "__main__":
    asyncio.run(main())
