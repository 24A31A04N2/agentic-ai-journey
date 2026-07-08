# -*- coding: utf-8 -*-
"""
🤖 Day 14/150 — Week 2 Capstone: Self-Correcting Agentic Data Pipeline
====================================================================
Phase 1: Python & Engineering Foundations

Concepts:
  1. YAML Configuration Parsing — Loading hierarchy from YAML, env, and CLI
  2. Concurrency & Parallelism — ThreadPoolExecutor for ingestion of multiple files
  3. Persistent SQLite DB — Thread-safe inserts, logs, and run execution metrics
  4. Self-Correction & Auto-healing — Robust clean-ups of dirty strings, dates, and types
  5. Schema Enforcement — Strict validations, data quarantine, and diagnostic audit
  6. Structured Logging — Outputting structured messages for trace logs and metrics
  7. argparse CLI Subcommands — A polished utility to run, inspect, generate, and test

Why this matters for Agentic AI:
  - Agent pipelines ingest external tool outputs, user inputs, and environment telemetry.
  - Raw data is frequently noisy, partially formatted, or missing optional values.
  - Rather than crash or ignore inputs, a self-correcting engine repairs recoverable data
    automatically (reducing downtime) and quarantines unrecoverable data safely.
  - Structured logs and run summaries allow agents to monitor their own health and trace anomalies.

Section Map:
  - Section 1: Config Management (Custom YAML Loader, Env overrides, CLI parsing)
  - Section 2: Schema & Self-Correction Engine
  - Section 3: SQLite Database Operations
  - Section 4: Concurrent Ingestion Pipeline
  - Section 5: Built-in Data Generator (for demonstration & testing)
  - Section 6: CLI & Commands
  - Section 7: Automated Integration Tests
"""

import os
import sys
import csv
import json
import time
import uuid
import logging
import sqlite3
import datetime
import argparse
import shutil
import threading
import gc
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Setup basic logging to stdout (will configure file/JSON logs dynamically in Section 4)
logger = logging.getLogger("DataPipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s'))
    logger.addHandler(handler)


# ============================================================================
# SECTION 1: Config Management (Custom YAML & Configuration Hierarchy)
# ============================================================================

def parse_simple_yaml(yaml_content: str) -> Dict[str, Any]:
    """
    Parses a basic YAML configuration string to avoid external dependencies like PyYAML.
    Supports basic nesting (indentation based), booleans, ints, floats, and strings.
    """
    result: Dict[str, Any] = {}
    indent_stack: List[Tuple[int, Dict[str, Any]]] = [(-1, result)]
    
    for line in yaml_content.splitlines():
        # Remove comments and whitespace
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
            
        indent = len(line) - len(line.lstrip())
        
        if ':' not in stripped:
            continue
            
        key, val_str = stripped.split(':', 1)
        key = key.strip()
        val_str = val_str.strip()
        
        # Determine value and type conversion
        if val_str == '':
            value: Any = {}
        elif (val_str.startswith('"') and val_str.endswith('"')) or (val_str.startswith("'") and val_str.endswith("'")):
            value = val_str[1:-1]
        elif val_str.lower() == 'true':
            value = True
        elif val_str.lower() == 'false':
            value = False
        else:
            try:
                if '.' in val_str:
                    value = float(val_str)
                else:
                    value = int(val_str)
            except ValueError:
                value = val_str
                
        # Pop indentation stack to find parent
        while indent_stack and indent_stack[-1][0] >= indent:
            indent_stack.pop()
            
        parent = indent_stack[-1][1]
        parent[key] = value
        
        if isinstance(value, dict):
            indent_stack.append((indent, value))
            
    return result


@dataclass
class PipelineConfig:
    db_path: str = "data/pipeline.db"
    input_dir: str = "data/input"
    quarantine_dir: str = "data/quarantine"
    archive_dir: str = "data/archive"
    log_level: str = "INFO"
    max_workers: int = 4
    batch_size: int = 50
    retry_attempts: int = 3
    retry_backoff_factor: float = 2.0
    auto_correct_enabled: bool = True
    default_agent_id: str = "agent-helper-v1"
    default_status: str = "success"
    default_tokens_used: int = 0
    default_cost: float = 0.0

    @classmethod
    def load(cls, yaml_path: Optional[str] = None) -> 'PipelineConfig':
        """
        Load configuration with a defined hierarchy:
        1. Class defaults
        2. YAML configuration file (if present)
        3. Environment variables (prefix: PIPELINE_)
        """
        config = cls()

        # 1. Load from YAML file if it exists
        if yaml_path and os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    yaml_data = parse_simple_yaml(f.read())
                
                # Extract values from hierarchical blocks
                pipeline_block = yaml_data.get('pipeline', {})
                ingestion_block = yaml_data.get('ingestion', {})
                defaults_block = yaml_data.get('defaults', {})
                
                # Populate config fields
                config.db_path = pipeline_block.get('db_path', config.db_path)
                config.input_dir = pipeline_block.get('input_dir', config.input_dir)
                config.quarantine_dir = pipeline_block.get('quarantine_dir', config.quarantine_dir)
                config.archive_dir = pipeline_block.get('archive_dir', config.archive_dir)
                config.log_level = pipeline_block.get('log_level', config.log_level)
                config.max_workers = int(pipeline_block.get('max_workers', config.max_workers))
                config.batch_size = int(pipeline_block.get('batch_size', config.batch_size))
                
                config.retry_attempts = int(ingestion_block.get('retry_attempts', config.retry_attempts))
                config.retry_backoff_factor = float(ingestion_block.get('retry_backoff_factor', config.retry_backoff_factor))
                config.auto_correct_enabled = bool(ingestion_block.get('auto_correct_enabled', config.auto_correct_enabled))
                
                config.default_agent_id = defaults_block.get('agent_id', config.default_agent_id)
                config.default_status = defaults_block.get('status', config.default_status)
                config.default_tokens_used = int(defaults_block.get('tokens_used', config.default_tokens_used))
                config.default_cost = float(defaults_block.get('cost', config.default_cost))
            except Exception as e:
                logger.warning(f"Failed to parse configuration file {yaml_path}: {e}. Using defaults.")

        # 2. Override from Environment Variables
        config.db_path = os.getenv("PIPELINE_DB_PATH", config.db_path)
        config.input_dir = os.getenv("PIPELINE_INPUT_DIR", config.input_dir)
        config.quarantine_dir = os.getenv("PIPELINE_QUARANTINE_DIR", config.quarantine_dir)
        config.archive_dir = os.getenv("PIPELINE_ARCHIVE_DIR", config.archive_dir)
        config.log_level = os.getenv("PIPELINE_LOG_LEVEL", config.log_level)
        
        if os.getenv("PIPELINE_MAX_WORKERS"):
            config.max_workers = int(os.environ["PIPELINE_MAX_WORKERS"])
        if os.getenv("PIPELINE_BATCH_SIZE"):
            config.batch_size = int(os.environ["PIPELINE_BATCH_SIZE"])
        if os.getenv("PIPELINE_RETRY_ATTEMPTS"):
            config.retry_attempts = int(os.environ["PIPELINE_RETRY_ATTEMPTS"])
        if os.getenv("PIPELINE_RETRY_BACKOFF_FACTOR"):
            config.retry_backoff_factor = float(os.environ["PIPELINE_RETRY_BACKOFF_FACTOR"])
        if os.getenv("PIPELINE_AUTO_CORRECT_ENABLED"):
            config.auto_correct_enabled = os.environ["PIPELINE_AUTO_CORRECT_ENABLED"].lower() in ('true', '1')

        # Override log level on parent logger
        numeric_level = getattr(logging, config.log_level.upper(), None)
        if isinstance(numeric_level, int):
            logger.setLevel(numeric_level)
            
        return config


# ============================================================================
# SECTION 2: Schema & Self-Correction Engine
# ============================================================================

class SelfCorrectionEngine:
    """
    Validates ingested records against the schema and performs self-correction on recoverable issues.
    Tracks all changes made during the repair operations.
    """
    def __init__(self, config: PipelineConfig):
        self.config = config

    def correct_timestamp(self, raw_ts: Any) -> Tuple[str, Optional[str]]:
        """
        Attempts to parse and standardize various timestamp formats into timezone-aware ISO 8601 UTC.
        """
        if not raw_ts:
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            return now, f"Timestamp was empty. Assigned current UTC timestamp: {now}"
            
        # Try numeric unix timestamps
        if isinstance(raw_ts, (int, float)):
            try:
                dt = datetime.datetime.fromtimestamp(raw_ts, datetime.timezone.utc)
                return dt.isoformat(), f"Coerced numeric timestamp '{raw_ts}' to ISO 8601 UTC"
            except Exception:
                pass
                
        s_ts = str(raw_ts).strip()
        # Try numeric unix timestamp representation in string
        try:
            val = float(s_ts)
            dt = datetime.datetime.fromtimestamp(val, datetime.timezone.utc)
            return dt.isoformat(), f"Coerced string unix timestamp '{s_ts}' to ISO 8601 UTC"
        except ValueError:
            pass

        # Match common date formats
        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y",
        ]
        
        for fmt in formats:
            try:
                dt = datetime.datetime.strptime(s_ts, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                return dt.isoformat(), f"Parsed date '{s_ts}' using format '{fmt}' and set to UTC"
            except ValueError:
                continue
                
        # If unable to parse, raise ValueError so it will be quarantined
        raise ValueError(f"Unparseable timestamp format: '{raw_ts}'")

    def correct_tokens(self, raw_tokens: Any) -> Tuple[int, Optional[str]]:
        """
        Ensures tokens is a non-negative integer. Coerces floats, strings, and fills nulls.
        """
        if raw_tokens is None or str(raw_tokens).strip() == "":
            return self.config.default_tokens_used, f"Missing tokens_used. Used default: {self.config.default_tokens_used}"
            
        try:
            val_str = str(raw_tokens).strip()
            # Clean floating decimal strings if they represent integer-like values
            val_float = float(val_str)
            val_int = int(val_float)
            if val_int < 0:
                return 0, f"Negative tokens_used '{val_int}' coerced to 0"
            if abs(val_float - val_int) > 1e-9:
                return val_int, f"Float tokens_used '{val_float}' coerced to int '{val_int}'"
            if val_str != str(val_int):
                return val_int, f"Coerced tokens_used string '{val_str}' to int '{val_int}'"
            return val_int, None
        except (ValueError, TypeError):
            return self.config.default_tokens_used, f"Invalid tokens_used '{raw_tokens}'. Coerced to default: {self.config.default_tokens_used}"

    def correct_cost(self, raw_cost: Any) -> Tuple[float, Optional[str]]:
        """
        Ensures cost is a non-negative float. Cleans currency symbols, whitespace, and handles nulls.
        """
        if raw_cost is None or str(raw_cost).strip() == "":
            return self.config.default_cost, f"Missing cost. Used default: {self.config.default_cost}"
            
        try:
            # Strip currency sign if present
            cleaned = str(raw_cost).strip().replace('$', '').replace('€', '').replace('£', '')
            val_float = float(cleaned)
            if val_float < 0.0:
                return 0.0, f"Negative cost '{val_float}' coerced to 0.0"
            if str(raw_cost).strip() != str(val_float):
                return val_float, f"Coerced cost value '{raw_cost}' to float '{val_float}'"
            return val_float, None
        except (ValueError, TypeError):
            return self.config.default_cost, f"Invalid cost '{raw_cost}'. Coerced to default: {self.config.default_cost}"

    def process_record(self, raw_record: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], List[str], Optional[str]]:
        """
        Processes and validates a single record.
        Returns:
            - success: bool (True if the record is valid or successfully corrected)
            - record: Dict (The clean/corrected record)
            - corrections: List[str] (List of correction warnings applied)
            - error_reason: str (Reason for failure, if success is False)
        """
        corrections: List[str] = []
        cleaned: Dict[str, Any] = {}
        
        # 1. ID check (Required, cannot be auto-generated without knowing identity, but we can generate a UUID if missing)
        raw_id = raw_record.get('id')
        if not raw_id:
            raw_id = raw_record.get('interaction_id')  # Alias fallback
        
        if not raw_id:
            # Let's generate a unique ID, but track it as a correction
            generated_id = str(uuid.uuid4())
            cleaned['id'] = generated_id
            corrections.append(f"Missing required identifier 'id'. Generated UUID: {generated_id}")
        else:
            cleaned['id'] = str(raw_id).strip()

        # 2. Timestamp Validation & Correction
        try:
            ts, correction_msg = self.correct_timestamp(raw_record.get('timestamp'))
            cleaned['timestamp'] = ts
            if correction_msg and self.config.auto_correct_enabled:
                corrections.append(correction_msg)
        except Exception as e:
            return False, raw_record, [], f"Timestamp validation failed: {e}"

        # 3. User ID Check
        raw_user = raw_record.get('user_id')
        if raw_user is None or str(raw_user).strip() == "":
            return False, raw_record, [], "Missing required field 'user_id'"
        cleaned['user_id'] = str(raw_user).strip()

        # 4. Agent ID Fallback
        raw_agent = raw_record.get('agent_id')
        if not raw_agent or str(raw_agent).strip() == "":
            cleaned['agent_id'] = self.config.default_agent_id
            if self.config.auto_correct_enabled:
                corrections.append(f"Missing agent_id. Assigned default: {self.config.default_agent_id}")
        else:
            cleaned['agent_id'] = str(raw_agent).strip()

        # 5. Prompt Validation
        raw_prompt = raw_record.get('prompt')
        if raw_prompt is None or str(raw_prompt).strip() == "":
            return False, raw_record, [], "Missing required field 'prompt'"
        cleaned['prompt'] = str(raw_prompt)

        # 6. Response Validation
        raw_response = raw_record.get('response')
        if raw_response is None or str(raw_response).strip() == "":
            return False, raw_record, [], "Missing required field 'response'"
        cleaned['response'] = str(raw_response)

        # 7. Tokens Used Validation & Correction
        tokens, correction_msg = self.correct_tokens(raw_record.get('tokens_used'))
        cleaned['tokens_used'] = tokens
        if correction_msg and self.config.auto_correct_enabled:
            corrections.append(correction_msg)

        # 8. Cost Validation & Correction
        cost, correction_msg = self.correct_cost(raw_record.get('cost'))
        cleaned['cost'] = cost
        if correction_msg and self.config.auto_correct_enabled:
            corrections.append(correction_msg)

        # 9. Status Validation & Correction
        raw_status = raw_record.get('status')
        valid_statuses = {"success", "warning", "error"}
        if not raw_status or str(raw_status).strip().lower() not in valid_statuses:
            cleaned['status'] = self.config.default_status
            if self.config.auto_correct_enabled:
                corrections.append(f"Invalid/missing status '{raw_status}'. Coerced to default: {self.config.default_status}")
        else:
            cleaned['status'] = str(raw_status).strip().lower()

        # If auto-correction is disabled, but corrections were needed, reject the record
        if not self.config.auto_correct_enabled and len(corrections) > 0:
            # Clean up the corrections array to explain why it failed validation
            reasons = "; ".join(corrections)
            return False, raw_record, [], f"Validation constraints violated (auto-correct disabled): {reasons}"

        return True, cleaned, corrections, None


# ============================================================================
# SECTION 3: SQLite Database Operations
# ============================================================================

class PipelineDatabase:
    """
    Manages connections and CRUD operations for SQLite in a thread-safe manner.
    Uses locking to coordinate concurrent inserts.
    """
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.db_path = config.db_path
        self._lock = threading.Lock()
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """
        Creates a new SQLite database connection.
        """
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """
        Initializes schema tables if they do not exist.
        """
        with self._lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Processed Records Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS processed_records (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        agent_id TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        response TEXT NOT NULL,
                        tokens_used INTEGER NOT NULL,
                        cost REAL NOT NULL,
                        status TEXT NOT NULL,
                        corrections TEXT, -- JSON string list
                        ingested_at TEXT NOT NULL
                    )
                """)
                
                # Quarantined Records Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS quarantined_records (
                        id TEXT,
                        timestamp TEXT,
                        file_name TEXT NOT NULL,
                        original_payload TEXT NOT NULL, -- JSON formatted dump
                        quarantine_reason TEXT NOT NULL,
                        quarantined_at TEXT NOT NULL
                    )
                """)
                
                # Pipeline Run Metrics Table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pipeline_metrics (
                        run_id TEXT PRIMARY KEY,
                        run_timestamp TEXT NOT NULL,
                        duration_seconds REAL NOT NULL,
                        files_processed INTEGER NOT NULL,
                        total_records INTEGER NOT NULL,
                        processed_count INTEGER NOT NULL,
                        corrected_count INTEGER NOT NULL,
                        quarantined_count INTEGER NOT NULL,
                        status TEXT NOT NULL
                    )
                """)
                conn.commit()

    def execute_with_retry(self, operation_func, *args, **kwargs) -> Any:
        """
        Executes a database operation with exponential backoff to handle temporary lock conflicts.
        """
        attempts = self.config.retry_attempts
        backoff = self.config.retry_backoff_factor
        current_sleep = 0.1
        
        for attempt in range(1, attempts + 1):
            try:
                with self._lock:
                    return operation_func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < attempts:
                    logger.warning(f"Database locked. Retrying in {current_sleep:.2f}s (Attempt {attempt}/{attempts})")
                    time.sleep(current_sleep)
                    current_sleep *= backoff
                else:
                    raise e

    def insert_records(self, processed: List[Dict[str, Any]], quarantined: List[Dict[str, Any]]):
        """
        Inserts batches of processed and quarantined records inside a single transaction.
        """
        def _batch_insert():
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Insert processed
                for r in processed:
                    cursor.execute("""
                        INSERT OR REPLACE INTO processed_records (
                            id, timestamp, user_id, agent_id, prompt, response, 
                            tokens_used, cost, status, corrections, ingested_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        r['id'], r['timestamp'], r['user_id'], r['agent_id'], r['prompt'], r['response'],
                        r['tokens_used'], r['cost'], r['status'], json.dumps(r.get('corrections', [])),
                        datetime.datetime.now(datetime.timezone.utc).isoformat()
                    ))
                
                # Insert quarantined
                for q in quarantined:
                    cursor.execute("""
                        INSERT INTO quarantined_records (
                            id, timestamp, file_name, original_payload, quarantine_reason, quarantined_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        q.get('id'), q.get('timestamp'), q['file_name'], q['original_payload'],
                        q['quarantine_reason'], datetime.datetime.now(datetime.timezone.utc).isoformat()
                    ))
                conn.commit()

        self.execute_with_retry(_batch_insert)

    def save_run_metrics(self, metrics: Dict[str, Any]):
        """
        Saves a run summary entry into the pipeline_metrics table.
        """
        def _insert_metrics():
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO pipeline_metrics (
                        run_id, run_timestamp, duration_seconds, files_processed, 
                        total_records, processed_count, corrected_count, quarantined_count, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics['run_id'], metrics['run_timestamp'], metrics['duration_seconds'],
                    metrics['files_processed'], metrics['total_records'], metrics['processed_count'],
                    metrics['corrected_count'], metrics['quarantined_count'], metrics['status']
                ))
                conn.commit()

        self.execute_with_retry(_insert_metrics)


# ============================================================================
# SECTION 4: Concurrent Ingestion Pipeline
# ============================================================================

class IngestionPipeline:
    """
    Coordinates loading data files, processing them concurrently using workers,
    persisting output, and logging operational runs.
    """
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.db = PipelineDatabase(config)
        self.engine = SelfCorrectionEngine(config)

        # Setup directories
        os.makedirs(self.config.input_dir, exist_ok=True)
        os.makedirs(self.config.quarantine_dir, exist_ok=True)
        os.makedirs(self.config.archive_dir, exist_ok=True)

    def parse_file(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Parses a file (JSON or CSV) and returns a list of raw dictionaries.
        Supports basic syntactical error handling (e.g. malformed JSON objects).
        """
        ext = os.path.splitext(filepath)[1].lower()
        records: List[Dict[str, Any]] = []

        if ext == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    return []
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        records = data
                    elif isinstance(data, dict):
                        records = [data]
                except json.JSONDecodeError as jde:
                    # Attempt self-healing of minor JSON format issues (e.g. missing trailing brackets or simple quotes)
                    logger.warning(f"File {os.path.basename(filepath)} is not valid JSON ({jde}). Attempting basic structural heal.")
                    # Try to heal trailing commas or missing closing array bracket
                    healed = False
                    if content.startswith('[') and not content.endswith(']'):
                        try:
                            records = json.loads(content + ']')
                            healed = True
                            logger.info("Successfully healed missing closing bracket ']' in JSON payload.")
                        except json.JSONDecodeError:
                            pass
                    if not healed:
                        raise ValueError(f"Malformed JSON syntax: {jde.msg} on line {jde.lineno}")

        elif ext == '.csv':
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Filter empty or fully-null rows
                    if any(row.values()):
                        records.append(dict(row))
        else:
            raise ValueError(f"Unsupported file extension '{ext}'")

        return records

    def process_single_file(self, filepath: str) -> Dict[str, Any]:
        """
        Worker task to read, validate, self-correct, and save data for one file.
        Returns worker metrics.
        """
        file_name = os.path.basename(filepath)
        metrics = {
            "file_name": file_name,
            "total_records": 0,
            "processed": 0,
            "corrected": 0,
            "quarantined": 0,
            "status": "success",
            "error_msg": None
        }

        processed_batch: List[Dict[str, Any]] = []
        quarantined_batch: List[Dict[str, Any]] = []

        try:
            raw_records = self.parse_file(filepath)
            metrics["total_records"] = len(raw_records)

            for raw in raw_records:
                success, clean, corrections, error_reason = self.engine.process_record(raw)
                
                if success:
                    clean['corrections'] = corrections
                    processed_batch.append(clean)
                    metrics["processed"] += 1
                    if len(corrections) > 0:
                        metrics["corrected"] += 1
                        logger.debug(f"[{file_name}] Corrected record ID {clean.get('id')}: {corrections}")
                else:
                    quarantined_batch.append({
                        "id": raw.get("id") or raw.get("interaction_id"),
                        "timestamp": raw.get("timestamp"),
                        "file_name": file_name,
                        "original_payload": json.dumps(raw),
                        "quarantine_reason": error_reason or "Unknown schema violation"
                    })
                    metrics["quarantined"] += 1
                    logger.warning(f"[{file_name}] Quarantining record. Reason: {error_reason}")

            # Write batches to SQLite
            if processed_batch or quarantined_batch:
                self.db.insert_records(processed_batch, quarantined_batch)

            # Move file to archive directory
            archive_path = os.path.join(self.config.archive_dir, file_name)
            if os.path.exists(archive_path):
                # Ensure unique filename to prevent overwrite
                base, ext = os.path.splitext(file_name)
                archive_path = os.path.join(self.config.archive_dir, f"{base}_{int(time.time())}{ext}")
            os.rename(filepath, archive_path)

        except Exception as e:
            metrics["status"] = "failed"
            metrics["error_msg"] = str(e)
            logger.error(f"Failed to process file {file_name}: {e}")
            
            # Quarantine the entire file
            quarantine_path = os.path.join(self.config.quarantine_dir, file_name)
            try:
                if os.path.exists(filepath):
                    os.rename(filepath, quarantine_path)
            except Exception as move_err:
                logger.critical(f"Could not move bad file {file_name} to quarantine: {move_err}")

        return metrics

    def run(self) -> Dict[str, Any]:
        """
        Executes pipeline over all available files in input_dir concurrently.
        """
        start_time = time.time()
        run_id = str(uuid.uuid4())
        run_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        logger.info(f"Starting pipeline execution run_id={run_id}")
        
        # Scan files
        all_files = [
            os.path.join(self.config.input_dir, f)
            for f in os.listdir(self.config.input_dir)
            if os.path.isfile(os.path.join(self.config.input_dir, f)) and f.lower().endswith(('.json', '.csv'))
        ]

        if not all_files:
            logger.info("No files found to ingest.")
            summary = {
                "run_id": run_id,
                "run_timestamp": run_ts,
                "duration_seconds": 0.0,
                "files_processed": 0,
                "total_records": 0,
                "processed_count": 0,
                "corrected_count": 0,
                "quarantined_count": 0,
                "status": "success"
            }
            self.db.save_run_metrics(summary)
            return summary

        logger.info(f"Found {len(all_files)} files to ingest. Concurrency level: {self.config.max_workers}")

        total_records = 0
        processed_count = 0
        corrected_count = 0
        quarantined_count = 0
        files_success = 0

        # Execute parallel ingestion
        with ThreadPoolExecutor(max_workers=self.config.max_workers, thread_name_prefix="IngestWorker") as executor:
            futures = {executor.submit(self.process_single_file, f): f for f in all_files}
            
            for future in as_completed(futures):
                res = future.result()
                if res["status"] == "success":
                    files_success += 1
                total_records += res["total_records"]
                processed_count += res["processed"]
                corrected_count += res["corrected"]
                quarantined_count += res["quarantined"]

        duration = round(time.time() - start_time, 4)
        run_status = "success" if files_success == len(all_files) else "partial_failure"
        if files_success == 0:
            run_status = "failed"

        summary = {
            "run_id": run_id,
            "run_timestamp": run_ts,
            "duration_seconds": duration,
            "files_processed": len(all_files),
            "total_records": total_records,
            "processed_count": processed_count,
            "corrected_count": corrected_count,
            "quarantined_count": quarantined_count,
            "status": run_status
        }

        # Save metrics to DB
        self.db.save_run_metrics(summary)
        logger.info(
            f"Run completed in {duration}s. Status={run_status}. "
            f"Processed: {processed_count} (Corrected: {corrected_count}), Quarantined: {quarantined_count}"
        )
        return summary


# ============================================================================
# SECTION 5: Built-in Data Generator (for testing and staging)
# ============================================================================

class MockDataGenerator:
    """
    Generates mock files in the input folder containing clean, repairable, and invalid records
    to test the self-correcting logic.
    """
    @staticmethod
    def generate_files(input_dir: str):
        os.makedirs(input_dir, exist_ok=True)
        
        # 1. Clean JSON File
        clean_data = [
            {
                "id": "interaction-001",
                "timestamp": "2026-07-08T06:00:00Z",
                "user_id": "usr_99",
                "agent_id": "translator-agent",
                "prompt": "Translate 'hello' to Spanish",
                "response": "Hola",
                "tokens_used": 12,
                "cost": 0.00024,
                "status": "success"
            },
            {
                "id": "interaction-002",
                "timestamp": "2026-07-08T06:05:00Z",
                "user_id": "usr_102",
                "agent_id": "code-assistant",
                "prompt": "Write a python print statement",
                "response": "print('hello')",
                "tokens_used": 20,
                "cost": 0.0004,
                "status": "success"
            }
        ]
        with open(os.path.join(input_dir, "clean_records.json"), "w", encoding="utf-8") as f:
            json.dump(clean_data, f, indent=2)

        # 2. Repairable JSON File (containing type anomalies, invalid format, missing values)
        repairable_data = [
            {
                "id": "interaction-003",
                "timestamp": "2026-07-08 06:10:00",  # Space-delimited instead of T
                "user_id": "usr_77",
                # missing agent_id
                "prompt": "Summarize this article",
                "response": "Summarized content...",
                "tokens_used": "150",                 # String instead of int
                "cost": "$0.0035",                   # String with currency symbol
                "status": "SUCCESS"                  # Uppercase status
            },
            {
                # Missing id -> will auto-generate one
                "timestamp": 1782201600.0,            # Unix float timestamp
                "user_id": 444,                       # Int user_id
                "agent_id": "  qa-bot  ",            # Whitespace
                "prompt": "Is this working?",
                "response": "Yes",
                "tokens_used": -20,                   # Negative token count (should coerce to 0)
                "cost": -1.0,                         # Negative cost (should coerce to 0.0)
                "status": "invalid_status_string"     # Invalid option -> defaults
            }
        ]
        with open(os.path.join(input_dir, "repairable_records.json"), "w", encoding="utf-8") as f:
            json.dump(repairable_data, f, indent=2)

        # 3. Invalid (Quarantine-bound) CSV File
        # Columns: id,timestamp,user_id,agent_id,prompt,response,tokens_used,cost,status
        csv_path = os.path.join(input_dir, "bad_records.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "timestamp", "user_id", "agent_id", "prompt", "response", "tokens_used", "cost", "status"])
            
            # Row A: Missing user_id (Unrecoverable)
            writer.writerow(["interaction-005", "2026-07-08T06:20:00Z", "", "agent-1", "Hello", "Hi there", "5", "0.0", "success"])
            
            # Row B: Missing prompt (Unrecoverable)
            writer.writerow(["interaction-006", "2026-07-08T06:21:00Z", "usr_1", "agent-1", "", "Response without prompt", "10", "0.0", "success"])
            
            # Row C: Unparseable Date (Unrecoverable)
            writer.writerow(["interaction-007", "not-a-date", "usr_2", "agent-1", "Test date", "Ok", "10", "0.0", "success"])

        # 4. Syntactically Malformed JSON (Missing trailing bracket, should be recovered if parser can heal it)
        malformed_json_content = """[
            {
                "id": "interaction-008",
                "timestamp": "2026-07-08T06:30:00Z",
                "user_id": "usr_999",
                "agent_id": "healer-agent",
                "prompt": "Test unclosed JSON bracket",
                "response": "Healed",
                "tokens_used": 15,
                "cost": 0.0003,
                "status": "success"
            }
        """ # Missing closing ']'
        with open(os.path.join(input_dir, "healed_syntax.json"), "w", encoding="utf-8") as f:
            f.write(malformed_json_content)

        logger.info(f"Mock data files created in '{input_dir}'")


# ============================================================================
# SECTION 6: CLI & Commands
# ============================================================================

def cmd_run(args, config: PipelineConfig):
    """Executes the pipeline ingest action."""
    pipeline = IngestionPipeline(config)
    summary = pipeline.run()
    
    print("\n================ Pipeline Execution Summary ================")
    print(f"Run ID:            {summary['run_id']}")
    print(f"Timestamp:         {summary['run_timestamp']}")
    print(f"Files Processed:   {summary['files_processed']}")
    print(f"Total Records:     {summary['total_records']}")
    print(f"Processed (Clean): {summary['processed_count']}")
    print(f"Corrected:         {summary['corrected_count']}")
    print(f"Quarantined:       {summary['quarantined_count']}")
    print(f"Duration:          {summary['duration_seconds']}s")
    print(f"Execution Status:  {summary['status'].upper()}")
    print("============================================================")


def cmd_generate(args, config: PipelineConfig):
    """Generates mock test files."""
    MockDataGenerator.generate_files(config.input_dir)
    print(f"Mock ingestion files generated successfully in: {os.path.abspath(config.input_dir)}")


def cmd_inspect(args, config: PipelineConfig):
    """Inspects the tables inside the SQLite Database."""
    db = PipelineDatabase(config)
    conn = db.get_connection()
    cursor = conn.cursor()
    
    print(f"Database Path: {os.path.abspath(config.db_path)}")
    
    # 1. Total processed records
    cursor.execute("SELECT COUNT(*) FROM processed_records")
    processed_count = cursor.fetchone()[0]
    
    # 2. Total quarantined records
    cursor.execute("SELECT COUNT(*) FROM quarantined_records")
    quarantined_count = cursor.fetchone()[0]
    
    # 3. Total runs
    cursor.execute("SELECT COUNT(*) FROM pipeline_metrics")
    runs_count = cursor.fetchone()[0]
    
    print(f"Total Runs Count: {runs_count}")
    print(f"Processed Records: {processed_count}")
    print(f"Quarantined Records: {quarantined_count}")
    
    if args.table == 'processed' and processed_count > 0:
        print("\n--- Processed Records (showing last 5) ---")
        cursor.execute("SELECT id, timestamp, user_id, agent_id, tokens_used, cost, status, corrections FROM processed_records ORDER BY ingested_at DESC LIMIT 5")
        for row in cursor.fetchall():
            corrections = json.loads(row['corrections']) if row['corrections'] else []
            corr_flag = f"(corrected: {len(corrections)} edits)" if corrections else "(clean)"
            print(f"ID: {row['id']} | Time: {row['timestamp']} | Agent: {row['agent_id']} | Status: {row['status']} | Cost: ${row['cost']} {corr_flag}")
            for c in corrections:
                print(f"  -> {c}")
                
    elif args.table == 'quarantine' and quarantined_count > 0:
        print("\n--- Quarantined Records (showing last 5) ---")
        cursor.execute("SELECT id, file_name, quarantine_reason, quarantined_at FROM quarantined_records ORDER BY quarantined_at DESC LIMIT 5")
        for row in cursor.fetchall():
            print(f"ID: {row['id']} | File: {row['file_name']} | Reason: {row['quarantine_reason']} | Quarantined At: {row['quarantined_at']}")

    elif args.table == 'metrics' and runs_count > 0:
        print("\n--- Pipeline Runs Summary (showing last 5) ---")
        cursor.execute("SELECT run_id, run_timestamp, duration_seconds, total_records, processed_count, quarantined_count, status FROM pipeline_metrics ORDER BY run_timestamp DESC LIMIT 5")
        for row in cursor.fetchall():
            print(f"Run: {row['run_id'][:8]}... | Time: {row['run_timestamp']} | Dur: {row['duration_seconds']}s | Recs: {row['total_records']} (P: {row['processed_count']}, Q: {row['quarantined_count']}) | Status: {row['status'].upper()}")

    conn.close()


def cmd_test(args, config: PipelineConfig):
    """Executes the automated integration tests."""
    run_integration_tests(config)


# ============================================================================
# SECTION 7: Automated Integration Tests
# ============================================================================

def run_integration_tests(config: PipelineConfig):
    """
    Performs a full end-to-end integration test of the self-correcting pipeline.
    Creates temporary paths, mock datasets, executes runs, and asserts correct operations.
    """
    logger.info("Initializing Integration Test Run...")
    
    # 1. Setup temporary testing environment with unique paths to avoid stale file conflicts
    test_suffix = str(int(time.time() * 1000))[-6:]
    test_base = os.path.join("data", f"test_{test_suffix}")
    test_db = os.path.join(test_base, "pipeline.db")
    test_input = os.path.join(test_base, "input")
    test_quarantine = os.path.join(test_base, "quarantine")
    test_archive = os.path.join(test_base, "archive")

    # Clean up any leftover test directories/files (robust on Windows)
    gc.collect()  # Release any lingering SQLite connections
    for p in [test_db, test_input, test_quarantine, test_archive]:
        try:
            if os.path.exists(p):
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.remove(p)
        except Exception:
            pass  # Best-effort cleanup of previous test artifacts

    # Instantiate custom test config
    test_config = PipelineConfig(
        db_path=test_db,
        input_dir=test_input,
        quarantine_dir=test_quarantine,
        archive_dir=test_archive,
        log_level="DEBUG",
        max_workers=2,
        batch_size=10,
        retry_attempts=2,
        retry_backoff_factor=1.0,
        auto_correct_enabled=True
    )

    try:
        # 2. Generate Mock Data
        MockDataGenerator.generate_files(test_input)

        # Confirm files were created
        files_created = os.listdir(test_input)
        assert len(files_created) == 4, f"Expected 4 mock files, found {len(files_created)}"
        logger.info("[PASS] Mock files generated successfully.")

        # 3. Instantiate pipeline & DB
        pipeline = IngestionPipeline(test_config)
        db = pipeline.db
        
        # Verify empty state
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM processed_records")
        assert cursor.fetchone()[0] == 0
        cursor.execute("SELECT COUNT(*) FROM quarantined_records")
        assert cursor.fetchone()[0] == 0
        conn.close()
        logger.info("[PASS] Database tables initialized and empty.")

        # 4. Execute the pipeline
        summary = pipeline.run()

        # 5. Perform Assertions on Metrics
        assert summary["files_processed"] == 4, f"Expected 4 files processed, got {summary['files_processed']}"
        # All files process successfully (quarantined records are per-record, not per-file failures)
        assert summary["status"] == "success", f"Expected 'success' (all files processed), got {summary['status']}"
        
        conn = db.get_connection()
        cursor = conn.cursor()

        # Verify processed records count
        # clean_records.json = 2 records (clean)
        # repairable_records.json = 2 records (repairable & corrected)
        # bad_records.csv = 3 records (all quarantined)
        # healed_syntax.json = 1 record (syntax healed and valid)
        # Total processed expected: 5
        cursor.execute("SELECT COUNT(*) FROM processed_records")
        processed_db_count = cursor.fetchone()[0]
        assert processed_db_count == 5, f"Expected 5 processed records in DB, found {processed_db_count}"
        logger.info("[PASS] Processed record count matches expected (5).")

        # Verify corrections were logged
        # Interaction-003 has: missing agent_id, cost with "$" symbol, uppercase STATUS
        cursor.execute("SELECT id, corrections FROM processed_records WHERE id='interaction-003'")
        row_3 = cursor.fetchone()
        assert row_3 is not None, "Record interaction-003 not found in DB"
        corrections_3 = json.loads(row_3['corrections'])
        assert len(corrections_3) > 0, "Expected corrections for interaction-003"
        assert any("cost" in c.lower() for c in corrections_3), "Expected cost correction notice"
        assert any("agent_id" in c.lower() for c in corrections_3), "Expected agent_id correction notice"
        logger.info("[PASS] Self-correction logic repaired cost, agent_id, and status values successfully.")

        # Verify the auto-generated ID record was saved and has corrections
        cursor.execute("SELECT id, corrections FROM processed_records WHERE user_id='444'")
        row_gen = cursor.fetchone()
        assert row_gen is not None, "Coerced record for user 444 not found"
        corrections_gen = json.loads(row_gen['corrections'])
        assert any("Missing required identifier 'id'" in c for c in corrections_gen), "Expected generated UUID correction note"
        logger.info("[PASS] Auto-assigned UUID to records lacking an identifier.")

        # Verify quarantine records
        # bad_records.csv had 3 records, all should be quarantined.
        cursor.execute("SELECT COUNT(*) FROM quarantined_records")
        quarantine_db_count = cursor.fetchone()[0]
        assert quarantine_db_count == 3, f"Expected 3 quarantined records in DB, found {quarantine_db_count}"
        logger.info("[PASS] Quarantined record count matches expected (3).")

        # Verify syntax healer corrected healed_syntax.json
        cursor.execute("SELECT id FROM processed_records WHERE id='interaction-008'")
        row_healed = cursor.fetchone()
        assert row_healed is not None, "Record interaction-008 was not healed and imported"
        logger.info("[PASS] Syntax repair successfully loaded unclosed JSON array file.")

        # Verify that archived folder contains the processed files
        archive_files = os.listdir(test_archive)
        assert len(archive_files) == 4, f"Expected 4 files in archive, found {len(archive_files)}"
        logger.info("[PASS] Ingested files relocated to the archive folder.")

        # Verify pipeline metrics were persisted
        cursor.execute("SELECT COUNT(*) FROM pipeline_metrics")
        metrics_count = cursor.fetchone()[0]
        assert metrics_count >= 1, f"Expected at least 1 pipeline run metric, found {metrics_count}"
        cursor.execute("SELECT run_id, status, processed_count, quarantined_count FROM pipeline_metrics ORDER BY run_timestamp DESC LIMIT 1")
        last_run = cursor.fetchone()
        assert last_run['processed_count'] == 5, f"Expected 5 processed in metrics, got {last_run['processed_count']}"
        assert last_run['quarantined_count'] == 3, f"Expected 3 quarantined in metrics, got {last_run['quarantined_count']}"
        logger.info("[PASS] Pipeline run metrics persisted correctly to database.")

        conn.close()

        print("\n🎉 ALL INTEGRATION TESTS PASSED SUCCESSFULLY! 🎉\n")

    except AssertionError as ae:
        logger.error(f"❌ INTEGRATION TEST FAILED: {ae}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ INTEGRATION TEST RUNTIME ERROR: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Close all connections and force garbage collection to release SQLite file locks
        try:
            conn.close()
        except Exception:
            pass
        conn = None
        db = None
        pipeline = None
        gc.collect()
        time.sleep(0.1)  # Brief pause to let Windows release file handles
        try:
            shutil.rmtree(test_base, ignore_errors=True)
        except Exception:
            pass


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Self-Correcting Agentic Data Pipeline - Week 2 Capstone Integration Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  run            Execute pipeline ingestion on all input files.
  generate-data  Create mock files (clean, repairable, invalid) in the input directory.
  inspect        Query and check internal SQLite database and pipeline run metrics.
  test           Run built-in automated integration test suite.
"""
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to YAML configuration file (default: config.yaml)"
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Pipeline action command")

    # Run command
    subparsers.add_parser("run", help="Run the ingestion pipeline on incoming files")

    # Generate command
    subparsers.add_parser("generate-data", help="Generate mock files for testing ingestion")

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect internal pipeline tables")
    inspect_parser.add_argument(
        "--table", 
        choices=["processed", "quarantine", "metrics"], 
        default="metrics", 
        help="Choose the database table to inspect (default: metrics)"
    )

    # Test command
    subparsers.add_parser("test", help="Run automated integration tests")

    args = parser.parse_args()

    # Load Config from YAML and environment variables
    # Check current directory and day-14 sub-directory
    config_path = args.config
    if not os.path.exists(config_path):
        # Fallback to check relative path inside day-14 directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_config = os.path.join(script_dir, "config.yaml")
        if os.path.exists(alt_config):
            config_path = alt_config

    config = PipelineConfig.load(config_path)

    if args.command == "run":
        cmd_run(args, config)
    elif args.command == "generate-data":
        cmd_generate(args, config)
    elif args.command == "inspect":
        cmd_inspect(args, config)
    elif args.command == "test":
        cmd_test(args, config)


if __name__ == "__main__":
    main()
