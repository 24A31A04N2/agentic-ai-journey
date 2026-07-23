# Day 16: Pandas & Data Wrangling — Agents That Clean Their Own Data 🐼

## 📝 Overview

Welcome to Day 16! Building on NumPy, today we conquer **Pandas**, the gold standard for tabular data manipulation in Python. 

Why is this critical for Agentic AI? AI agents operate in the real world. They ingest CSVs, read from databases, analyze server logs, and parse messy API responses. If an agent cannot programmatically clean, filter, aggregate, and transform structured data, it will hallucinate or fail when passing massive, noisy contexts to an LLM. 

Today, we learn how to give agents the capability to **wrangle and summarize their own data**.

## 🧠 Core Concepts Mastered

| Concept | What It Is | Why Agents Need It |
|---------|------------|---------------------|
| **DataFrame Fundamentals** | 2D labeled data structures (like Excel in Python). | To hold tabular state (e.g., interaction logs, DB queries). |
| **Selection & Filtering** | Slicing data with `.loc`, `.iloc`, and boolean masks. | To extract only the relevant subsets before passing to an LLM context. |
| **Data Cleaning** | Handling NaNs, deduping, string cleaning, type coercion. | Real-world data is dirty. Agents must sanitize inputs before acting. |
| **Feature Engineering** | `.apply()`, `.map()`, binning, and datetime parsing. | To derive actionable metrics (e.g., sentiment scores, time-since-failure). |
| **GroupBy & Aggregations** | Summarizing data across categories. | To compress thousands of rows into a clean summary table for LLM reasoning. |

## 🚀 Practical Application: Agent Log Analyzer

In the capstone section of today's script, we build an **Agent Log Analyzer**.
Imagine an Orchestrator Agent receiving a raw, messy CSV of its sub-agents' activities. The script demonstrates how the agent can:
1. Load the messy CSV into a DataFrame.
2. Clean invalid inputs, coerce token counts to integers, and fill NaNs.
3. Compute derived features like token cost and success rates.
4. Group by sub-agent to create a statistical summary.
5. Identify anomalies (e.g., API calls taking > 10,000ms).
6. Auto-generate a Markdown report of its findings.

## 💻 Code Highlights

### Intelligent Filtering with Boolean Masks
```python
# Agent finds slow but successful requests
slow_success = df[(df["status_code"] == 200) & (df["response_ms"] > 300)]
```

### Applying Agentic Heuristics
```python
# Applying a custom heuristic to classify sentiment
df['sentiment_category'] = df['sentiment_score'].apply(
    lambda score: "Positive" if score > 0.5 else ("Negative" if score < 0.0 else "Neutral")
)
```

### Summarizing for the LLM Context Window
```python
# Summarizing 10,000 rows into 3 rows to fit context window
agent_metrics = df_logs.groupby('sub_agent').agg(
    tasks_succeeded=('is_success', 'sum'),
    total_cost=('cost_usd', 'sum'),
    avg_exec_time_ms=('exec_time_ms', 'mean')
)
```

## ▶️ How to Run

1. Ensure you have the required packages installed:
   ```bash
   pip install pandas numpy
   ```
2. Run the script:
   ```bash
   python day16_pandas_wrangling.py
   ```
3. Follow the console output as it walks through the 6 sections, concluding with the generated Agent Log Report.
