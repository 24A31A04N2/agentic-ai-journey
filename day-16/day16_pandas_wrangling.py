# -*- coding: utf-8 -*-
import sys
import codecs

# Fix for Windows console UTF-8 encoding issues
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

"""
=============================================================================
Day 16: Pandas & Data Wrangling — Agents That Clean Their Own Data
=============================================================================

Concepts:
- DataFrame Fundamentals (creation, inspection, stats)
- Selection & Filtering (.loc, .iloc, booleans)
- Data Cleaning (NaNs, deduplication, types, string cleaning)
- Transformations & Feature Engineering (apply, map, bins, datetimes)
- GroupBy & Aggregations (summarization, pivots)
- Practical Agent Application (Agent Log Analyzer)

Why it matters for AI Agents:
Agents constantly ingest messy tabular data from logs, databases, user uploads,
and CSVs. Without the ability to programmatically clean, filter, and transform
structured data, agents cannot effectively pass context to LLMs, generate insights,
or make data-driven decisions. Pandas is the standard for tabular data in Python.

Section Map:
1. DataFrame Fundamentals
2. Selection & Filtering
3. Data Cleaning
4. Transformations & Feature Engineering
5. GroupBy & Aggregations
6. Practical Agent Application (Agent Log Analyzer)
=============================================================================
"""

import pandas as pd
import numpy as np
import io
import time
from datetime import datetime, timedelta

# =====================================================================
# Helper Function for Printing
# =====================================================================
def print_section(title: str):
    """Prints a styled section header."""
    print("\n" + "=" * 80)
    print(f"🚀 SECTION: {title}")
    print("=" * 80)


def print_step(step: str):
    """Prints a styled step sub-header."""
    print(f"\n🔹 {step}")
    print("-" * 40)


# =====================================================================
# SECTION 1: DataFrame Fundamentals
# =====================================================================
def section_1_dataframe_fundamentals():
    print_section("1. DataFrame Fundamentals")
    
    # In this section, we cover the very basics of how agents might 
    # instantiate data structures holding tabular data.
    
    print_step("Creating DataFrames from Dictionaries")
    # A common scenario: an agent gets a JSON response from an API
    # and converts it to a DataFrame.
    mock_api_response = {
        "agent_id": ["agent_alpha", "agent_beta", "agent_gamma", "agent_delta"],
        "model_version": ["gpt-4", "gpt-3.5", "claude-3", "gpt-4"],
        "tasks_completed": [152, 340, 89, 210],
        "uptime_hours": [120.5, 400.1, 48.0, 150.2],
        "is_active": [True, True, False, True]
    }
    
    df_agents = pd.DataFrame(mock_api_response)
    print("DataFrame from Dictionary:")
    print(df_agents)
    
    print_step("Creating DataFrames from List of Dictionaries")
    # Another common scenario: reading rows from a database query.
    db_rows = [
        {"task_id": 1, "status": "success", "latency": 1.2},
        {"task_id": 2, "status": "failed", "latency": 5.4},
        {"task_id": 3, "status": "success", "latency": 0.8},
    ]
    df_tasks = pd.DataFrame(db_rows)
    print("DataFrame from List of Dicts:")
    print(df_tasks)
    
    print_step("Simulating CSV Ingestion with io.StringIO")
    # Agents often read from files. Here we simulate reading a CSV.
    csv_data = """id,name,score,department
1,Alice,85,Sales
2,Bob,92,Engineering
3,Charlie,78,Sales
4,Diana,95,Engineering
5,Eve,88,HR"""
    
    df_employees = pd.read_csv(io.StringIO(csv_data))
    print("DataFrame read from CSV string:")
    print(df_employees)
    
    print_step("Inspecting the DataFrame")
    # When agents get data, they need to know its shape and types before
    # acting on it.
    print(f"Shape of df_agents: {df_agents.shape} (rows, columns)")
    print(f"Columns: {df_agents.columns.tolist()}")
    print("\nData Types:")
    print(df_agents.dtypes)
    
    print_step("Summary Information (.info and .describe)")
    print("info() gives a concise summary of the DataFrame:")
    # info() prints directly to stdout
    df_agents.info()
    
    print("\ndescribe() provides summary statistics of numerical columns:")
    print(df_agents.describe())
    
    print_step("Viewing Data (.head and .tail)")
    # For large datasets, agents might sample the top or bottom rows.
    print("Top 2 rows:")
    print(df_agents.head(2))
    print("\nBottom 2 rows:")
    print(df_agents.tail(2))


# =====================================================================
# SECTION 2: Selection & Filtering
# =====================================================================
def section_2_selection_filtering():
    print_section("2. Selection & Filtering")
    
    # We need a robust dataset to demonstrate filtering.
    # We will generate a mock dataset of API requests.
    np.random.seed(42)
    num_requests = 20
    
    data = {
        "request_id": range(1001, 1001 + num_requests),
        "endpoint": np.random.choice(["/chat", "/embeddings", "/images", "/audio"], num_requests),
        "status_code": np.random.choice([200, 400, 429, 500], num_requests, p=[0.7, 0.1, 0.1, 0.1]),
        "response_ms": np.random.normal(loc=250, scale=100, size=num_requests).round(2),
        "user_tier": np.random.choice(["free", "pro", "enterprise"], num_requests)
    }
    
    df = pd.DataFrame(data)
    # Ensure no negative response times by applying absolute value
    df['response_ms'] = df['response_ms'].abs()
    
    print_step("The Mock API Request Data")
    print(df.head(8))
    
    print_step("Selecting Columns")
    # Agents often need only specific columns for their prompt context.
    endpoints_only = df["endpoint"]
    print("Single column (Series):")
    print(endpoints_only.head(3))
    
    subset_df = df[["request_id", "endpoint", "status_code"]]
    print("\nMultiple columns (DataFrame):")
    print(subset_df.head(3))
    
    print_step("Row Selection by Position (.iloc)")
    # .iloc is integer-location based indexing.
    print("Selecting the 5th row (index 4):")
    print(df.iloc[4])
    print("\nSelecting rows 2 to 4:")
    print(df.iloc[2:5])
    
    print_step("Row Selection by Label (.loc)")
    # Let's set request_id as the index to demonstrate label-based indexing.
    df_indexed = df.set_index("request_id")
    print("Selecting request_id 1005 using .loc:")
    print(df_indexed.loc[1005])
    
    print_step("Boolean Indexing (Filtering)")
    # This is crucial for agents deciding what needs attention.
    # E.g., finding all failed requests (status != 200)
    failed_requests = df[df["status_code"] != 200]
    print("Failed requests:")
    print(failed_requests)
    
    # Combining conditions: Slow AND successful requests
    slow_success = df[(df["status_code"] == 200) & (df["response_ms"] > 300)]
    print("\nSlow but successful requests (>300ms):")
    print(slow_success)
    
    print_step("Filtering with .isin()")
    # E.g., agent wants to see only specific endpoints.
    target_endpoints = ["/chat", "/embeddings"]
    chat_and_embed = df[df["endpoint"].isin(target_endpoints)]
    print(f"Requests to {target_endpoints}:")
    print(chat_and_embed.head())
    
    print_step("Filtering with .query()")
    # .query() allows filtering using a string expression, which is 
    # very useful when LLMs generate the filtering logic dynamically.
    query_string = "user_tier == 'pro' and status_code == 200"
    pro_success = df.query(query_string)
    print(f"Query Result for '{query_string}':")
    print(pro_success)


# =====================================================================
# SECTION 3: Data Cleaning
# =====================================================================
def section_3_data_cleaning():
    print_section("3. Data Cleaning")
    
    # Dirty data is the bane of agentic systems. We need to handle NaNs,
    # weird strings, incorrect types, and duplicates.
    
    dirty_data = """id,  name  ,age,email,join_date,score
1,  Alice Smith  ,28,alice@example.com,2023-01-15,85.5
2,Bob Jones,,bob@example.com,,90.0
3,  CHARLIE  ,35,invalid_email,2022-11-01,
4,Diana Prince,42,diana@example.com,2023-03-10,95.5
1,  Alice Smith  ,28,alice@example.com,2023-01-15,85.5
5,Eve,not_an_age,eve@example.com,2023-05-20,70.0"""

    df = pd.read_csv(io.StringIO(dirty_data))
    print_step("Raw Dirty DataFrame")
    print(df)
    
    print_step("Handling Duplicates")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
    df = df.drop_duplicates()
    print("DataFrame after drop_duplicates():")
    print(df)
    
    print_step("String Cleaning")
    # Column names might have leading/trailing spaces.
    df.columns = df.columns.str.strip()
    
    # Clean the 'name' column: strip spaces and convert to title case.
    df['name'] = df['name'].astype(str).str.strip().str.title()
    print("DataFrame after string cleaning on 'name':")
    print(df[['id', 'name']])
    
    print_step("Handling Incorrect Data Types")
    # 'age' has 'not_an_age' which makes it an object column. We need it as numeric.
    # pd.to_numeric with errors='coerce' will turn invalid parsing into NaN.
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    print("DataFrame after coercing 'age' to numeric:")
    print(df[['id', 'name', 'age']])
    print(df.dtypes)
    
    print_step("Handling Missing Values (NaNs)")
    # Finding missing values
    print("Missing values per column:")
    print(df.isnull().sum())
    
    # Strategy 1: Fill missing age with the median age.
    median_age = df['age'].median()
    df['age'] = df['age'].fillna(median_age)
    print(f"\nFilled missing age with median ({median_age}).")
    
    # Strategy 2: Fill missing score with 0.
    df['score'] = df['score'].fillna(0)
    
    # Strategy 3: Drop rows where 'join_date' is missing.
    df = df.dropna(subset=['join_date'])
    
    print("\nDataFrame after handling missing values:")
    print(df)
    
    print_step("Advanced String Validation (Regex)")
    # Agents might need to validate emails. Let's find rows with valid emails.
    # A simple regex for demo purposes.
    valid_email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    is_valid_email = df['email'].astype(str).str.match(valid_email_pattern)
    print("\nValid email mask:")
    print(is_valid_email)
    
    # Keep only rows with valid emails (or flag them)
    df['is_valid_email'] = is_valid_email
    print("\nFinal Cleaned Data:")
    print(df)


# =====================================================================
# SECTION 4: Transformations & Feature Engineering
# =====================================================================
def section_4_transformations():
    print_section("4. Transformations & Feature Engineering")
    
    # Agents often need to derive new features or metrics from raw data
    # to feed better signals into LLMs or decision engines.
    
    data = {
        "text": ["Hello world", "Agentic AI is cool", "Pandas helps a lot", "Short"],
        "sentiment_score": [0.1, 0.9, 0.8, -0.2],
        "created_at": ["2023-10-01 10:15:00", "2023-10-01 12:30:00", 
                       "2023-10-02 08:45:00", "2023-10-03 18:00:00"]
    }
    df = pd.DataFrame(data)
    print_step("Initial DataFrame")
    print(df)
    
    print_step("Using .apply() for Row/Column Transformations")
    # Let's count words in the 'text' column.
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    
    # Complex transformation function simulating an agent's heuristic
    def categorize_sentiment(score):
        if score > 0.5:
            return "Positive"
        elif score < 0.0:
            return "Negative"
        else:
            return "Neutral"
            
    df['sentiment_category'] = df['sentiment_score'].apply(categorize_sentiment)
    
    print("DataFrame with derived features:")
    print(df[['text', 'word_count', 'sentiment_category']])
    
    print_step("Using .map() for Value Substitution")
    # Mapping categorical values to numerical IDs or other representations.
    category_mapping = {"Positive": 1, "Neutral": 0, "Negative": -1}
    df['sentiment_id'] = df['sentiment_category'].map(category_mapping)
    print("\nMapped categories to IDs:")
    print(df[['sentiment_category', 'sentiment_id']])
    
    print_step("Binning with pd.cut()")
    # Creating discrete bins from continuous variables.
    # Suppose we have user ages and want to bin them into demographics.
    ages = pd.DataFrame({"age": [15, 22, 35, 45, 60, 72, 8]})
    bins = [0, 18, 35, 50, 100]
    labels = ["Child", "Young Adult", "Adult", "Senior"]
    ages['demographic'] = pd.cut(ages['age'], bins=bins, labels=labels)
    print("\nAge Binning:")
    print(ages)
    
    print_step("Datetime Parsing and Extraction")
    # Time is critical for agents (e.g., "what happened in the last hour?").
    # Convert string to datetime objects.
    df['created_at'] = pd.to_datetime(df['created_at'])
    print(f"\nData type of 'created_at' is now: {df['created_at'].dtype}")
    
    # Extracting components
    df['hour'] = df['created_at'].dt.hour
    df['day_of_week'] = df['created_at'].dt.day_name()
    
    print("DataFrame with extracted datetime features:")
    print(df[['created_at', 'hour', 'day_of_week']])
    
    # Calculating time differences (e.g., time since first event)
    first_event = df['created_at'].min()
    df['hours_since_start'] = (df['created_at'] - first_event).dt.total_seconds() / 3600
    print("\nHours since first event:")
    print(df[['created_at', 'hours_since_start']])


# =====================================================================
# SECTION 5: GroupBy & Aggregations
# =====================================================================
def section_5_groupby_aggregations():
    print_section("5. GroupBy & Aggregations")
    
    # Agents summarize information. Instead of passing 10,000 logs to an LLM
    # (which would exhaust context), an agent uses groupby to pass a summary table.
    
    # Mock data: sales transactions processed by different agents
    data = {
        "date": pd.date_range(start="2023-01-01", periods=10, freq="D").tolist() * 2,
        "agent": ["Agent_A"] * 10 + ["Agent_B"] * 10,
        "region": ["North", "South", "North", "North", "South", "East", "East", "North", "South", "East"] * 2,
        "revenue": np.random.uniform(100, 1000, 20).round(2),
        "tickets_resolved": np.random.randint(1, 10, 20)
    }
    df = pd.DataFrame(data)
    print_step("Initial Transaction Data (Sample)")
    print(df.head())
    
    print_step("Basic GroupBy")
    # Total revenue by agent
    revenue_by_agent = df.groupby("agent")["revenue"].sum().reset_index()
    print("Total Revenue by Agent:")
    print(revenue_by_agent)
    
    print_step("Multiple Grouping Columns")
    # Revenue by agent AND region
    revenue_by_agent_region = df.groupby(["agent", "region"])["revenue"].sum().reset_index()
    print("\nTotal Revenue by Agent and Region:")
    print(revenue_by_agent_region)
    
    print_step("Aggregating Multiple Metrics (.agg)")
    # We want sum of revenue, but mean of tickets resolved, and count of transactions.
    summary_stats = df.groupby("agent").agg(
        total_revenue=("revenue", "sum"),
        avg_revenue=("revenue", "mean"),
        total_tickets=("tickets_resolved", "sum"),
        transaction_count=("date", "count")
    ).reset_index()
    
    # Format the revenue columns
    summary_stats['total_revenue'] = summary_stats['total_revenue'].round(2)
    summary_stats['avg_revenue'] = summary_stats['avg_revenue'].round(2)
    
    print("\nComplex Aggregations per Agent:")
    print(summary_stats)
    
    print_step("Pivot Tables")
    # Pivot tables are excellent for reshaping data into a matrix, very readable for LLMs.
    pivot = pd.pivot_table(
        df, 
        values='revenue', 
        index='agent', 
        columns='region', 
        aggfunc='sum',
        fill_value=0
    )
    print("\nPivot Table (Revenue by Agent vs Region):")
    print(pivot)
    
    print_step("Cross-Tabulations")
    # Count frequency of occurrences
    crosstab = pd.crosstab(df['agent'], df['region'])
    print("\nCross-Tabulation (Count of transactions by Agent vs Region):")
    print(crosstab)


# =====================================================================
# SECTION 6: Practical Agent Application (Agent Log Analyzer)
# =====================================================================
def section_6_agent_log_analyzer():
    print_section("6. Practical Agent Application: Agent Log Analyzer")
    
    """
    Scenario:
    You are the 'Orchestrator Agent'. You oversee 3 sub-agents working on a complex
    research task. You periodically receive a raw, slightly messy CSV log of their
    interactions with various tools and LLMs.
    
    Your task is to ingest this log, clean it, calculate performance metrics,
    identify any anomalies, and generate a markdown summary report to send to the user.
    """
    
    print_step("Generating Mock Log Data")
    
    # Simulating a messy CSV log file
    raw_csv = """timestamp,sub_agent,tool_used,prompt_tokens,completion_tokens,exec_time_ms,status,error_msg
2023-11-01 08:00:10,Researcher,search_web,150,500,2500,SUCCESS,
2023-11-01 08:02:15,Researcher,read_page,200,800,3200,SUCCESS,
2023-11-01 08:05:00,Coder,write_script,500,1200,5500,SUCCESS,
2023-11-01 08:06:30,Coder,run_script,0,0,1200,FAILED,SyntaxError on line 42
2023-11-01 08:08:00,Coder,write_script,600,1300,6000,SUCCESS,
2023-11-01 08:10:00,Reviewer,analyze_code,1900,400,4100,SUCCESS,
2023-11-01 08:15:00,Researcher,search_web,, ,90000,FAILED,TimeoutError
2023-11-01 08:16:00,Researcher,search_web,160,450,2300,SUCCESS,
2023-11-01 08:20:00,Coder,run_script,0,0,1500,SUCCESS,
2023-11-01 08:25:00,Reviewer,analyze_code,2100,550,4800,SUCCESS,
2023-11-01 08:30:00,UNKNOWN,,,,,FAILED,Unknown Agent Error
2023-11-01 08:31:00,Researcher,read_page,210,850,3300,SUCCESS,"""

    print("Raw CSV Data loaded into Pandas.")
    df_logs = pd.read_csv(io.StringIO(raw_csv))
    
    print_step("Phase 1: Data Cleaning")
    
    # 1. Drop completely invalid rows (e.g., UNKNOWN agent)
    df_logs = df_logs[df_logs['sub_agent'] != 'UNKNOWN']
    
    # 2. Handle missing tokens. If a tool failed, tokens might be NaN. Fill with 0.
    # Also coerce empty spaces to NaN first using pd.to_numeric
    df_logs['prompt_tokens'] = pd.to_numeric(df_logs['prompt_tokens'], errors='coerce').fillna(0)
    df_logs['completion_tokens'] = pd.to_numeric(df_logs['completion_tokens'], errors='coerce').fillna(0)
    
    # 3. Ensure exec_time_ms is numeric
    df_logs['exec_time_ms'] = pd.to_numeric(df_logs['exec_time_ms'], errors='coerce')
    
    # 4. Parse timestamp
    df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
    
    # 5. Fill NaN error messages with empty string
    df_logs['error_msg'] = df_logs['error_msg'].fillna("")
    
    print("Data after cleaning:")
    print(df_logs.info())
    
    print_step("Phase 2: Feature Engineering")
    
    # Calculate Total Tokens
    df_logs['total_tokens'] = df_logs['prompt_tokens'] + df_logs['completion_tokens']
    
    # Calculate Cost (Assuming $0.01 per 1K prompt tokens, $0.03 per 1K completion tokens)
    df_logs['cost_usd'] = (df_logs['prompt_tokens'] / 1000 * 0.01) + (df_logs['completion_tokens'] / 1000 * 0.03)
    
    # Create a boolean flag for success
    df_logs['is_success'] = df_logs['status'] == 'SUCCESS'
    
    print("\nEngineered Features Sample:")
    print(df_logs[['sub_agent', 'total_tokens', 'cost_usd', 'is_success']].head())
    
    print_step("Phase 3: Aggregations & Metrics")
    
    # Group by Sub-Agent to get performance metrics
    agent_metrics = df_logs.groupby('sub_agent').agg(
        tasks_attempted=('status', 'count'),
        tasks_succeeded=('is_success', 'sum'),
        total_cost=('cost_usd', 'sum'),
        avg_exec_time_ms=('exec_time_ms', 'mean'),
        total_tokens=('total_tokens', 'sum')
    ).reset_index()
    
    # Calculate Success Rate
    agent_metrics['success_rate_%'] = (agent_metrics['tasks_succeeded'] / agent_metrics['tasks_attempted'] * 100).round(1)
    
    # Format Cost
    agent_metrics['total_cost'] = agent_metrics['total_cost'].round(4)
    agent_metrics['avg_exec_time_ms'] = agent_metrics['avg_exec_time_ms'].round(0)
    
    print("\nAggregated Agent Metrics:")
    print(agent_metrics)
    
    print_step("Phase 4: Anomaly Detection")
    
    # Rule 1: Find tasks that took unusually long (> 10000 ms)
    slow_tasks = df_logs[df_logs['exec_time_ms'] > 10000]
    
    # Rule 2: Find frequent errors
    failed_tasks = df_logs[df_logs['status'] == 'FAILED']
    
    print(f"\nFound {len(slow_tasks)} exceptionally slow tasks.")
    if not slow_tasks.empty:
        print(slow_tasks[['sub_agent', 'tool_used', 'exec_time_ms']])
        
    print(f"Found {len(failed_tasks)} failed tasks.")
    
    print_step("Phase 5: Report Generation")
    
    # The agent builds a markdown report from the pandas structures
    
    report = f"""# 📊 Agent Execution Summary Report
**Generated at:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total Tasks Processed:** {len(df_logs)}
**Overall Success Rate:** {(df_logs['is_success'].mean() * 100):.1f}%
**Total Run Cost:** ${df_logs['cost_usd'].sum():.4f}

## 🤖 Sub-Agent Performance
{agent_metrics.to_string(index=False)}

## ⚠️ Anomalies & Errors Detected
"""
    
    if len(failed_tasks) > 0:
        report += "\n### Failed Tasks\n"
        for _, row in failed_tasks.iterrows():
            report += f"- **{row['sub_agent']}** failed using `{row['tool_used']}` at {row['timestamp'].strftime('%H:%M:%S')}. Error: *{row['error_msg']}*\n"
            
    if len(slow_tasks) > 0:
        report += "\n### Performance Warnings\n"
        for _, row in slow_tasks.iterrows():
            report += f"- **{row['sub_agent']}** experienced a slow response ({row['exec_time_ms']}ms) while using `{row['tool_used']}`.\n"
            
    print("\n" + "=" * 60)
    print("GENERATED MARKDOWN REPORT:")
    print("=" * 60)
    print(report)


# =====================================================================
# Main Execution
# =====================================================================
def demonstrate_all():
    """Runs all sections to demonstrate Pandas capabilities for Agents."""
    print("\nStarting Day 16 Pandas Demonstrations...\n")
    time.sleep(1)
    
    section_1_dataframe_fundamentals()
    time.sleep(1)
    
    section_2_selection_filtering()
    time.sleep(1)
    
    section_3_data_cleaning()
    time.sleep(1)
    
    section_4_transformations()
    time.sleep(1)
    
    section_5_groupby_aggregations()
    time.sleep(1)
    
    section_6_agent_log_analyzer()
    
    print("\n" + "=" * 80)
    print("✅ Day 16 Pandas & Data Wrangling Demonstrations Completed Successfully.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    demonstrate_all()
