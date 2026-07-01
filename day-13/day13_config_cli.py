"""
🤖 Day 13/150 — Configuration Management & CLI — Agents That Adapt to Any Environment
=====================================================================================
Phase 1: Python & Engineering Foundations

Concepts:
  1. Environment Variables — os.environ, .env files, dotenv patterns for secrets
  2. Configuration Dataclasses — Typed, validated, immutable config objects
  3. YAML/JSON Config Files — Loading agent config from external files
  4. Configuration Hierarchy — defaults → config file → env vars → CLI args
  5. argparse CLI — Professional CLIs with subcommands and argument groups
  6. AgentConfigManager — Complete multi-source config manager with validation

Why this matters for Agentic AI:
  - Agents need API keys, model names, temperature, max_tokens — all configurable.
  - Hardcoding config means rewriting code for every deployment environment.
  - Production agents run in dev/staging/prod with different settings each time.
  - CLI interfaces let operators control agents without editing source code.
  - A proper config hierarchy (defaults → file → env → CLI) lets teams override
    only what they need at each layer, following the 12-Factor App methodology.
  - Secrets (API keys) must NEVER be in source code — env vars or vaults only.
"""

import os
import sys
import json
import copy
import argparse
import tempfile
import textwrap
from dataclasses import dataclass, field, asdict, fields
from typing import (
    List, Dict, Optional, Any, Tuple, Union, Type, get_type_hints,
)
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ============================================================================
# SECTION 1: Environment Variables — Secrets & Runtime Config
# ============================================================================
# Why: Agents interact with external APIs (OpenAI, Anthropic, search engines).
# API keys are secrets that must NEVER be committed to source control. The
# standard practice is to load them from environment variables, optionally
# populated from a .env file during local development. This section shows
# how to safely read, validate, and organize environment-based configuration.

def get_env(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """
    Safely read an environment variable with validation.

    Why this wrapper exists:
      - os.environ[key] raises KeyError with no context.
      - os.environ.get(key) silently returns None — bugs hide for hours.
      - This function gives clear error messages and enforces requirements.

    Args:
        key: Environment variable name (e.g., "OPENAI_API_KEY").
        default: Fallback value if the variable is not set.
        required: If True, raises ValueError when the variable is missing.

    Returns:
        The environment variable value, or the default.
    """
    value = os.environ.get(key)
    if value is not None:
        return value
    if required and default is None:
        raise ValueError(
            f"Required environment variable '{key}' is not set. "
            f"Set it with: export {key}=<value>  (Linux/Mac) or "
            f"$env:{key}='<value>'  (PowerShell)"
        )
    return default or ""


def get_env_int(key: str, default: int = 0) -> int:
    """Read an environment variable as an integer with type safety."""
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable '{key}' must be an integer, got '{raw}'"
        )


def get_env_float(key: str, default: float = 0.0) -> float:
    """Read an environment variable as a float with type safety."""
    raw = os.environ.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable '{key}' must be a float, got '{raw}'"
        )


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Read an environment variable as a boolean.

    Recognizes: "true", "1", "yes", "on" as True (case-insensitive).
    Everything else (including empty string) is False.
    """
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("true", "1", "yes", "on")


def parse_dotenv_line(line: str) -> Optional[Tuple[str, str]]:
    """
    Parse a single line from a .env file.

    Format: KEY=value or KEY="quoted value"
    Ignores comments (#) and blank lines.

    Why we implement this instead of using python-dotenv:
      - Zero dependencies — important for learning and minimal deployments.
      - python-dotenv is great in production but understanding the format
        matters more at this stage.
    """
    line = line.strip()
    # Skip empty lines and comments
    if not line or line.startswith("#"):
        return None
    # Split on first '='
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    # Remove surrounding quotes if present
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        value = value[1:-1]
    return key, value


def load_dotenv(filepath: str, override: bool = False) -> Dict[str, str]:
    """
    Load environment variables from a .env file.

    Args:
        filepath: Path to the .env file.
        override: If True, overwrite existing env vars. Default: False
                  (existing env vars take priority — safer for production).

    Returns:
        Dict of key-value pairs that were loaded.

    Why override=False by default:
      - In production, env vars are set by the deployment system (Docker,
        Kubernetes, CI/CD). The .env file is for local dev convenience only.
      - override=False ensures deployment env vars always win.
    """
    loaded = {}
    path = Path(filepath)
    if not path.exists():
        return loaded

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_dotenv_line(line)
            if parsed:
                key, value = parsed
                if override or key not in os.environ:
                    os.environ[key] = value
                loaded[key] = value
    return loaded


def demonstrate_env_vars() -> Dict[str, Any]:
    """
    Demonstrates environment variable handling for agent configuration.

    Sets up sample env vars, reads them with type safety, and shows
    the .env file parsing pattern.
    """
    # Set some demo env vars (in real usage, these come from the OS or .env)
    demo_vars = {
        "AGENT_MODEL": "gpt-4o",
        "AGENT_TEMPERATURE": "0.7",
        "AGENT_MAX_TOKENS": "2048",
        "AGENT_DEBUG": "true",
        "AGENT_API_KEY": "sk-demo-key-not-real-12345",
    }
    for k, v in demo_vars.items():
        os.environ[k] = v

    # Read with type-safe helpers
    config = {
        "model": get_env("AGENT_MODEL", default="gpt-3.5-turbo"),
        "temperature": get_env_float("AGENT_TEMPERATURE", default=0.5),
        "max_tokens": get_env_int("AGENT_MAX_TOKENS", default=1024),
        "debug": get_env_bool("AGENT_DEBUG", default=False),
        "api_key": get_env("AGENT_API_KEY", default=""),
        "missing_var": get_env("NONEXISTENT_VAR", default="<not set>"),
    }

    # Create and parse a temporary .env file
    dotenv_content = textwrap.dedent("""\
        # Agent Configuration (.env file)
        AGENT_NAME="research-assistant"
        AGENT_VERSION=1.0.0
        AGENT_LOG_LEVEL=INFO

        # API Keys (never commit real keys!)
        OPENAI_API_KEY="sk-placeholder-key"
    """)

    # Write to a temp file and parse it
    dotenv_path = Path(tempfile.gettempdir()) / ".env.demo"
    dotenv_path.write_text(dotenv_content, encoding="utf-8")
    dotenv_vars = load_dotenv(str(dotenv_path), override=False)

    # Clean up demo env vars (don't pollute the real environment)
    for k in demo_vars:
        os.environ.pop(k, None)
    for k in dotenv_vars:
        os.environ.pop(k, None)

    return {
        "typed_config": config,
        "dotenv_vars": dotenv_vars,
        "dotenv_content": dotenv_content.strip(),
    }


# ============================================================================
# SECTION 2: Configuration Dataclasses — Typed, Validated Config Objects
# ============================================================================
# Why: Dicts are flexible but offer no type safety, no autocompletion, and no
# validation. Dataclasses give us typed fields with defaults, __post_init__
# for validation, and frozen=True for immutability. This prevents config from
# being accidentally mutated after initialization — critical when config is
# shared across threads or agent components.

class LogLevel(Enum):
    """Log level enum for type-safe configuration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelProvider(Enum):
    """Supported LLM providers for agent configuration."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"


@dataclass(frozen=True)
class ModelConfig:
    """
    Immutable configuration for an LLM model.

    Why frozen=True:
      - Once an agent starts a conversation with specific model settings,
        changing temperature or max_tokens mid-conversation causes
        inconsistent behavior.
      - Frozen dataclasses are hashable — can be used as dict keys or in sets.
      - Thread-safe by nature: no mutation means no race conditions.

    Validation in __post_init__:
      - Catches invalid config early (at startup) instead of mid-conversation.
      - temperature must be 0.0–2.0 (OpenAI range).
      - max_tokens must be positive.
    """
    model_name: str = "gpt-4o"
    provider: ModelProvider = ModelProvider.OPENAI
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    def __post_init__(self):
        # frozen=True means we can't do self.x = y, but __post_init__ runs
        # before freeze takes effect for validation purposes.
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be 0.0–2.0, got {self.temperature}"
            )
        if self.max_tokens < 1:
            raise ValueError(
                f"max_tokens must be >= 1, got {self.max_tokens}"
            )
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(
                f"top_p must be 0.0–1.0, got {self.top_p}"
            )


@dataclass(frozen=True)
class ToolConfig:
    """Configuration for a single agent tool."""
    name: str
    enabled: bool = True
    timeout_seconds: float = 30.0
    max_retries: int = 3
    description: str = ""

    def __post_init__(self):
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be > 0, got {self.timeout_seconds}"
            )


@dataclass(frozen=True)
class AgentConfig:
    """
    Complete agent configuration — the single source of truth.

    Why one top-level config object:
      - Every component receives the same config instance.
      - No ambiguity about which setting applies where.
      - Easy to serialize (asdict), log, and compare between environments.

    Design pattern: Nested frozen dataclasses.
      - AgentConfig contains ModelConfig, List[ToolConfig], etc.
      - Each sub-config validates independently.
      - The whole tree is immutable once constructed.
    """
    agent_name: str = "default-agent"
    agent_version: str = "1.0.0"
    environment: str = "development"
    log_level: LogLevel = LogLevel.INFO
    model: ModelConfig = field(default_factory=ModelConfig)
    tools: Tuple[ToolConfig, ...] = ()  # Tuple for frozen compatibility
    system_prompt: str = "You are a helpful AI assistant."
    max_conversation_turns: int = 50
    debug: bool = False

    def __post_init__(self):
        valid_envs = {"development", "staging", "production"}
        if self.environment not in valid_envs:
            raise ValueError(
                f"environment must be one of {valid_envs}, got '{self.environment}'"
            )
        if self.max_conversation_turns < 1:
            raise ValueError(
                f"max_conversation_turns must be >= 1, got {self.max_conversation_turns}"
            )


def demonstrate_config_dataclasses() -> Dict[str, Any]:
    """Demonstrates typed configuration dataclasses with validation."""
    # Create a valid config
    model = ModelConfig(
        model_name="gpt-4o",
        provider=ModelProvider.OPENAI,
        temperature=0.7,
        max_tokens=4096,
    )

    tools = (
        ToolConfig(name="web_search", timeout_seconds=10.0, description="Search the web"),
        ToolConfig(name="calculator", timeout_seconds=5.0, max_retries=1),
        ToolConfig(name="code_executor", timeout_seconds=30.0, enabled=False),
    )

    config = AgentConfig(
        agent_name="research-assistant",
        agent_version="2.1.0",
        environment="development",
        log_level=LogLevel.DEBUG,
        model=model,
        tools=tools,
        system_prompt="You are a research assistant specializing in AI papers.",
        debug=True,
    )

    # Show serialization
    config_dict = asdict(config)

    # Demonstrate validation errors
    validation_errors = []
    try:
        ModelConfig(temperature=3.0)  # Out of range
    except ValueError as e:
        validation_errors.append(str(e))

    try:
        ModelConfig(max_tokens=-1)
    except ValueError as e:
        validation_errors.append(str(e))

    try:
        AgentConfig(environment="invalid")
    except ValueError as e:
        validation_errors.append(str(e))

    # Demonstrate immutability
    immutability_error = None
    try:
        config.debug = False  # type: ignore  # Should raise FrozenInstanceError
    except Exception as e:
        immutability_error = f"{type(e).__name__}: {e}"

    return {
        "config": config,
        "config_dict": config_dict,
        "validation_errors": validation_errors,
        "immutability_error": immutability_error,
    }


# ============================================================================
# SECTION 3: YAML/JSON Config Files — External Agent Configuration
# ============================================================================
# Why: Config files let non-developers modify agent behavior without touching
# Python code. YAML is human-friendly (comments, clean syntax). JSON is
# universal (APIs, tooling). Agents should support both. This section shows
# how to load, parse, and validate configuration from external files.

def load_json_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.

    Why JSON:
      - Universal format — every language and tool supports it.
      - Strict syntax means fewer ambiguity bugs.
      - Great for machine-generated config (API responses, CI/CD).

    Args:
        filepath: Path to the JSON config file.

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filepath}: {e}")


def load_yaml_config(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Why YAML over JSON for config:
      - Supports comments (# this is a comment) — essential for documenting
        why a setting has a particular value.
      - Cleaner syntax: no braces, no quotes for simple strings.
      - More readable for humans editing config files.

    Implementation note:
      - We use a simple YAML parser that handles the subset of YAML commonly
        used in config files (key: value, lists, nested dicts).
      - For production, use PyYAML or ruamel.yaml for full YAML spec support.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    return parse_simple_yaml(content)


def parse_simple_yaml(content: str) -> Dict[str, Any]:
    """
    Parse a subset of YAML sufficient for agent configuration files.

    Supports:
      - key: value pairs (strings, numbers, booleans)
      - Nested sections via indentation
      - Lists with "- item" syntax
      - Comments with #
      - Quoted strings

    Why a custom parser instead of importing PyYAML:
      - Zero external dependencies for this learning module.
      - Understanding the format is more valuable than importing a library.
      - Handles the 90% of YAML features used in config files.
    """
    result: Dict[str, Any] = {}
    lines = content.split("\n")
    stack: List[Tuple[int, Dict]] = [(0, result)]  # (indent_level, current_dict)
    current_list_key: Optional[str] = None
    current_list: Optional[List] = None
    list_indent: int = 0

    for line in lines:
        stripped = line.strip()
        # Skip blank lines and comments
        if not stripped or stripped.startswith("#"):
            continue

        # Calculate indentation
        indent = len(line) - len(line.lstrip())

        # Handle list items
        if stripped.startswith("- "):
            item_value = _parse_yaml_value(stripped[2:].strip())
            if current_list is not None:
                current_list.append(item_value)
            continue

        # End current list if we're back to key: value
        if current_list is not None and not stripped.startswith("- "):
            current_list_key = None
            current_list = None

        # Parse key: value
        if ":" in stripped:
            colon_idx = stripped.index(":")
            key = stripped[:colon_idx].strip()
            value_str = stripped[colon_idx + 1:].strip()

            # Adjust stack based on indentation
            while len(stack) > 1 and indent <= stack[-1][0]:
                stack.pop()

            current_dict = stack[-1][1]

            if value_str == "" or value_str.startswith("#"):
                # Nested section — create sub-dict
                new_dict: Dict[str, Any] = {}
                current_dict[key] = new_dict
                stack.append((indent + 2, new_dict))
            elif value_str.startswith("[") and value_str.endswith("]"):
                # Inline list: [item1, item2, ...]
                items_str = value_str[1:-1]
                items = [_parse_yaml_value(i.strip()) for i in items_str.split(",") if i.strip()]
                current_dict[key] = items
            else:
                # Check if next lines are list items
                current_dict[key] = _parse_yaml_value(value_str)
                # Peek ahead for list
                line_idx = lines.index(line) if line in lines else -1
                if line_idx >= 0 and line_idx + 1 < len(lines):
                    next_stripped = lines[line_idx + 1].strip()
                    if next_stripped.startswith("- "):
                        current_list = []
                        current_dict[key] = current_list
                        current_list_key = key
                        list_indent = indent

    return result


def _parse_yaml_value(value: str) -> Any:
    """Parse a YAML scalar value into the appropriate Python type."""
    if not value or value.startswith("#"):
        return ""

    # Remove inline comments
    if " #" in value:
        value = value[:value.index(" #")].strip()

    # Remove surrounding quotes
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]

    # Boolean
    lower = value.lower()
    if lower in ("true", "yes", "on"):
        return True
    if lower in ("false", "no", "off"):
        return False
    if lower in ("null", "none", "~"):
        return None

    # Number
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    return value


def dict_to_agent_config(data: Dict[str, Any]) -> AgentConfig:
    """
    Convert a parsed config dictionary to a typed AgentConfig.

    Why this conversion step:
      - Config files are untyped (YAML/JSON → dict).
      - AgentConfig is typed (dataclass with validation).
      - This function bridges the gap, applying defaults and validation.
    """
    # Parse model config
    model_data = data.get("model", {})
    provider_str = model_data.get("provider", "openai")
    try:
        provider = ModelProvider(provider_str.lower())
    except ValueError:
        provider = ModelProvider.OPENAI

    model = ModelConfig(
        model_name=model_data.get("model_name", "gpt-4o"),
        provider=provider,
        temperature=float(model_data.get("temperature", 0.7)),
        max_tokens=int(model_data.get("max_tokens", 2048)),
        top_p=float(model_data.get("top_p", 1.0)),
        frequency_penalty=float(model_data.get("frequency_penalty", 0.0)),
        presence_penalty=float(model_data.get("presence_penalty", 0.0)),
    )

    # Parse tool configs
    tools_data = data.get("tools", [])
    tools = []
    if isinstance(tools_data, list):
        for tool in tools_data:
            if isinstance(tool, dict):
                tools.append(ToolConfig(
                    name=tool.get("name", "unknown"),
                    enabled=bool(tool.get("enabled", True)),
                    timeout_seconds=float(tool.get("timeout_seconds", 30.0)),
                    max_retries=int(tool.get("max_retries", 3)),
                    description=tool.get("description", ""),
                ))

    # Parse log level
    log_level_str = data.get("log_level", "INFO")
    try:
        log_level = LogLevel(str(log_level_str).upper())
    except ValueError:
        log_level = LogLevel.INFO

    return AgentConfig(
        agent_name=data.get("agent_name", "default-agent"),
        agent_version=str(data.get("agent_version", "1.0.0")),
        environment=data.get("environment", "development"),
        log_level=log_level,
        model=model,
        tools=tuple(tools),
        system_prompt=data.get("system_prompt", "You are a helpful AI assistant."),
        max_conversation_turns=int(data.get("max_conversation_turns", 50)),
        debug=bool(data.get("debug", False)),
    )


def demonstrate_config_files() -> Dict[str, Any]:
    """Demonstrates loading configuration from YAML and JSON files."""
    # Create a sample JSON config
    json_config = {
        "agent_name": "json-agent",
        "agent_version": "1.0.0",
        "environment": "staging",
        "log_level": "WARNING",
        "model": {
            "model_name": "claude-3-sonnet",
            "provider": "anthropic",
            "temperature": 0.5,
            "max_tokens": 4096,
        },
        "tools": [
            {"name": "web_search", "timeout_seconds": 15.0, "enabled": True},
            {"name": "code_runner", "timeout_seconds": 60.0, "max_retries": 2},
        ],
        "debug": False,
    }

    # Write JSON to temp file
    json_path = Path(tempfile.gettempdir()) / "agent_config.json"
    json_path.write_text(json.dumps(json_config, indent=2), encoding="utf-8")

    # Load it back
    loaded_json = load_json_config(str(json_path))
    json_agent_config = dict_to_agent_config(loaded_json)

    # Create a sample YAML config string for parsing
    yaml_content = textwrap.dedent("""\
        # Agent Configuration
        agent_name: yaml-research-bot
        agent_version: 2.0.0
        environment: development
        log_level: DEBUG
        debug: true

        # Model Settings
        model:
          model_name: gpt-4o
          provider: openai
          temperature: 0.8
          max_tokens: 8192
          top_p: 0.95

        # System Prompt
        system_prompt: "You are a research assistant that finds and summarizes papers."

        max_conversation_turns: 100
    """)

    parsed_yaml = parse_simple_yaml(yaml_content)
    yaml_agent_config = dict_to_agent_config(parsed_yaml)

    return {
        "json_config": json_agent_config,
        "yaml_config": yaml_agent_config,
        "parsed_yaml_dict": parsed_yaml,
        "json_path": str(json_path),
    }


# ============================================================================
# SECTION 4: Configuration Hierarchy — Multi-Layer Config Resolution
# ============================================================================
# Why: Production systems need layered configuration:
#   1. DEFAULTS (in code) — sensible baseline for all environments.
#   2. CONFIG FILE — environment-specific overrides (dev.yaml, prod.yaml).
#   3. ENV VARS — deployment-level overrides (Docker, Kubernetes secrets).
#   4. CLI ARGS — operator overrides for one-off runs or debugging.
# Each layer overrides the previous. This is the 12-Factor App approach.

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge two dictionaries. Override values take priority.

    Why deep merge instead of dict.update():
      - dict.update() replaces entire nested dicts, losing non-overridden keys.
      - Deep merge preserves base values for keys not present in override.

    Example:
      base = {"model": {"name": "gpt-4o", "temp": 0.7}}
      override = {"model": {"temp": 0.9}}
      result = {"model": {"name": "gpt-4o", "temp": 0.9}}
      # dict.update would give: {"model": {"temp": 0.9}} — "name" is lost!
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# Default configuration — the foundation of the hierarchy
DEFAULT_CONFIG: Dict[str, Any] = {
    "agent_name": "default-agent",
    "agent_version": "1.0.0",
    "environment": "development",
    "log_level": "INFO",
    "debug": False,
    "model": {
        "model_name": "gpt-4o",
        "provider": "openai",
        "temperature": 0.7,
        "max_tokens": 2048,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    },
    "tools": [],
    "system_prompt": "You are a helpful AI assistant.",
    "max_conversation_turns": 50,
}


def env_vars_to_config(prefix: str = "AGENT_") -> Dict[str, Any]:
    """
    Extract agent configuration from environment variables.

    Convention: AGENT_MODEL_NAME → {"model": {"model_name": value}}
    The prefix is stripped, and underscores map to nested dict keys.

    Why a prefix:
      - Avoids collisions with other apps' env vars.
      - Makes it clear which vars belong to the agent.
      - Common pattern: MYAPP_DB_HOST, MYAPP_LOG_LEVEL.
    """
    config: Dict[str, Any] = {}
    prefix_upper = prefix.upper()

    # Mapping from env var suffix to config path
    env_mappings = {
        "NAME": ("agent_name",),
        "VERSION": ("agent_version",),
        "ENVIRONMENT": ("environment",),
        "LOG_LEVEL": ("log_level",),
        "DEBUG": ("debug",),
        "MODEL_NAME": ("model", "model_name"),
        "MODEL_PROVIDER": ("model", "provider"),
        "TEMPERATURE": ("model", "temperature"),
        "MAX_TOKENS": ("model", "max_tokens"),
        "TOP_P": ("model", "top_p"),
        "SYSTEM_PROMPT": ("system_prompt",),
        "MAX_TURNS": ("max_conversation_turns",),
    }

    for suffix, path in env_mappings.items():
        env_key = f"{prefix_upper}{suffix}"
        value = os.environ.get(env_key)
        if value is not None:
            # Type coercion
            parsed_value: Any = value
            if suffix == "DEBUG":
                parsed_value = value.lower() in ("true", "1", "yes")
            elif suffix in ("TEMPERATURE", "TOP_P"):
                try:
                    parsed_value = float(value)
                except ValueError:
                    continue
            elif suffix in ("MAX_TOKENS", "MAX_TURNS"):
                try:
                    parsed_value = int(value)
                except ValueError:
                    continue

            # Set nested value
            current = config
            for key in path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[path[-1]] = parsed_value

    return config


def resolve_config(
    config_file: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
    env_prefix: str = "AGENT_",
) -> Dict[str, Any]:
    """
    Resolve the final configuration from all sources.

    Priority (highest wins):
      4. CLI arguments  (operator wants this specific run to behave differently)
      3. Environment variables  (deployment sets API keys, model names)
      2. Config file  (team agrees on settings for this environment)
      1. Defaults  (sensible baseline coded into the application)

    Why this order:
      - Defaults keep the app running even with zero configuration.
      - Config files capture team/environment decisions.
      - Env vars handle secrets and deployment-specific overrides.
      - CLI args are the highest priority — the operator knows best right now.
    """
    # Layer 1: Start with defaults
    config = copy.deepcopy(DEFAULT_CONFIG)

    # Layer 2: Merge config file (if provided)
    if config_file:
        try:
            if config_file.endswith(".json"):
                file_config = load_json_config(config_file)
            elif config_file.endswith((".yaml", ".yml")):
                file_config = load_yaml_config(config_file)
            else:
                file_config = {}
            config = deep_merge(config, file_config)
        except (FileNotFoundError, ValueError) as e:
            print(f"  Warning: Could not load config file: {e}")

    # Layer 3: Merge environment variables
    env_config = env_vars_to_config(prefix=env_prefix)
    if env_config:
        config = deep_merge(config, env_config)

    # Layer 4: Merge CLI overrides (highest priority)
    if cli_overrides:
        config = deep_merge(config, cli_overrides)

    return config


def demonstrate_config_hierarchy() -> Dict[str, Any]:
    """Demonstrates the 4-layer configuration hierarchy."""
    # Layer 1: Defaults
    defaults_only = copy.deepcopy(DEFAULT_CONFIG)

    # Layer 2: Simulate a config file override
    file_overrides = {
        "agent_name": "file-configured-agent",
        "environment": "staging",
        "model": {
            "model_name": "claude-3-opus",
            "provider": "anthropic",
            "temperature": 0.5,
        },
    }

    # Layer 3: Simulate env var overrides
    os.environ["AGENT_TEMPERATURE"] = "0.9"
    os.environ["AGENT_DEBUG"] = "true"

    # Layer 4: Simulate CLI overrides
    cli_overrides = {
        "model": {
            "max_tokens": 8192,
        },
        "debug": True,
    }

    # Build the resolution chain step by step
    after_defaults = copy.deepcopy(DEFAULT_CONFIG)
    after_file = deep_merge(after_defaults, file_overrides)

    env_config = env_vars_to_config()
    after_env = deep_merge(after_file, env_config)

    after_cli = deep_merge(after_env, cli_overrides)

    # Clean up env vars
    os.environ.pop("AGENT_TEMPERATURE", None)
    os.environ.pop("AGENT_DEBUG", None)

    return {
        "layer_1_defaults": defaults_only,
        "layer_2_after_file": after_file,
        "layer_3_after_env": after_env,
        "layer_4_final": after_cli,
        "resolution_chain": [
            ("Defaults", defaults_only.get("model", {}).get("temperature")),
            ("+ Config File", after_file.get("model", {}).get("temperature")),
            ("+ Env Vars", after_env.get("model", {}).get("temperature")),
            ("+ CLI Args", after_cli.get("model", {}).get("temperature")),
        ],
    }


# ============================================================================
# SECTION 5: argparse CLI — Professional Command-Line Interfaces
# ============================================================================
# Why: Agents run as CLI programs: python agent.py --model gpt-4o --temperature 0.9
# argparse is Python's built-in module for building professional CLIs with:
#   - Typed arguments with defaults and help text
#   - Subcommands (agent run, agent config show, agent config validate)
#   - Argument groups for organized help output
#   - Automatic --help generation

def build_agent_cli() -> argparse.ArgumentParser:
    """
    Build a professional CLI for an AI agent application.

    Design decisions:
      - Subcommands: 'run', 'config', 'tools' — mirrors real agent CLIs.
      - Argument groups: model settings, runtime settings, output settings.
      - Type annotations: --temperature is float, --max-tokens is int.
      - Defaults align with DEFAULT_CONFIG for consistency.

    Why argparse over click/typer:
      - Zero dependencies — built into Python stdlib.
      - Sufficient for 90% of CLI needs.
      - Understanding argparse makes click/typer trivial to learn later.
    """
    # Top-level parser
    parser = argparse.ArgumentParser(
        prog="agent",
        description="🤖 AI Agent CLI — Configure and run AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              agent run --model gpt-4o --temperature 0.8
              agent run --config prod.yaml --debug
              agent config show --config my_config.yaml
              agent config validate --config my_config.yaml
              agent tools list
        """),
    )

    # Global arguments (apply to all subcommands)
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration file (YAML or JSON)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode with verbose output",
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available agent commands",
    )

    # --- 'run' subcommand ---
    run_parser = subparsers.add_parser(
        "run",
        help="Run the AI agent",
        description="Start the AI agent with the specified configuration",
    )

    # Model settings group
    model_group = run_parser.add_argument_group("Model Settings")
    model_group.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model name (e.g., gpt-4o, claude-3-sonnet)",
    )
    model_group.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "google", "local"],
        default=None,
        help="LLM provider",
    )
    model_group.add_argument(
        "--temperature", "-t",
        type=float,
        default=None,
        help="Sampling temperature (0.0–2.0)",
    )
    model_group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens in response",
    )

    # Runtime settings group
    runtime_group = run_parser.add_argument_group("Runtime Settings")
    runtime_group.add_argument(
        "--environment", "-e",
        type=str,
        choices=["development", "staging", "production"],
        default=None,
        help="Deployment environment",
    )
    runtime_group.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum conversation turns",
    )
    runtime_group.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt for the agent",
    )

    # --- 'config' subcommand ---
    config_parser = subparsers.add_parser(
        "config",
        help="Manage agent configuration",
        description="View, validate, or export agent configuration",
    )
    config_subparsers = config_parser.add_subparsers(
        dest="config_action",
        title="config actions",
    )

    # config show
    show_parser = config_subparsers.add_parser(
        "show",
        help="Display the resolved configuration",
    )
    show_parser.add_argument(
        "--format", "-f",
        choices=["json", "yaml", "table"],
        default="json",
        help="Output format",
    )

    # config validate
    validate_parser = config_subparsers.add_parser(
        "validate",
        help="Validate a configuration file",
    )

    # --- 'tools' subcommand ---
    tools_parser = subparsers.add_parser(
        "tools",
        help="Manage agent tools",
        description="List and configure agent tools",
    )
    tools_subparsers = tools_parser.add_subparsers(
        dest="tools_action",
        title="tool actions",
    )

    list_parser = tools_subparsers.add_parser(
        "list",
        help="List all configured tools",
    )
    list_parser.add_argument(
        "--enabled-only",
        action="store_true",
        help="Show only enabled tools",
    )

    return parser


def cli_args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Convert parsed CLI arguments to a config override dictionary.

    Only includes arguments that were explicitly set (not None).
    This ensures CLI args only override what the user specified.
    """
    overrides: Dict[str, Any] = {}

    # Model overrides
    model_overrides: Dict[str, Any] = {}
    if getattr(args, "model", None) is not None:
        model_overrides["model_name"] = args.model
    if getattr(args, "provider", None) is not None:
        model_overrides["provider"] = args.provider
    if getattr(args, "temperature", None) is not None:
        model_overrides["temperature"] = args.temperature
    if getattr(args, "max_tokens", None) is not None:
        model_overrides["max_tokens"] = args.max_tokens
    if model_overrides:
        overrides["model"] = model_overrides

    # Runtime overrides
    if getattr(args, "environment", None) is not None:
        overrides["environment"] = args.environment
    if getattr(args, "max_turns", None) is not None:
        overrides["max_conversation_turns"] = args.max_turns
    if getattr(args, "system_prompt", None) is not None:
        overrides["system_prompt"] = args.system_prompt
    if getattr(args, "debug", False):
        overrides["debug"] = True

    return overrides


def demonstrate_argparse_cli() -> Dict[str, Any]:
    """Demonstrates building and parsing a professional CLI."""
    parser = build_agent_cli()

    # Simulate different CLI invocations
    test_commands = [
        # Simple run with model override
        ["run", "--model", "gpt-4o", "--temperature", "0.9"],
        # Run with config file and debug
        ["run", "--config", "prod.yaml", "--debug", "--environment", "production"],
        # Config show
        ["config", "show", "--format", "json"],
        # Tools list
        ["tools", "list", "--enabled-only"],
    ]

    results = []
    for cmd in test_commands:
        try:
            args = parser.parse_args(cmd)
            cli_overrides = cli_args_to_config(args)
            results.append({
                "command": " ".join(cmd),
                "parsed": vars(args),
                "config_overrides": cli_overrides,
            })
        except SystemExit:
            results.append({
                "command": " ".join(cmd),
                "error": "Parse error (expected for demo)",
            })

    # Show help text
    help_text = parser.format_help()

    return {
        "test_results": results,
        "help_text": help_text,
        "num_commands_parsed": len(results),
    }


# ============================================================================
# SECTION 6: AgentConfigManager — Complete Multi-Source Config Manager
# ============================================================================
# Why: This class ties everything together into a single, production-ready
# configuration manager. It loads from all sources (defaults, files, env vars,
# CLI args), validates the result, provides typed access, and supports config
# comparison between environments. This is what a real agent framework uses.

class ConfigSource(Enum):
    """Tracks where each config value came from — essential for debugging."""
    DEFAULT = "default"
    FILE = "config_file"
    ENV_VAR = "env_var"
    CLI_ARG = "cli_arg"
    PROGRAMMATIC = "programmatic"


@dataclass
class ConfigEntry:
    """A single configuration entry with source tracking."""
    key: str
    value: Any
    source: ConfigSource
    description: str = ""


class AgentConfigManager:
    """
    Complete configuration manager for AI agents.

    Responsibilities:
      1. Load configuration from all sources (defaults, file, env, CLI).
      2. Resolve the final config using the priority hierarchy.
      3. Validate the resolved config and report errors.
      4. Provide typed access (get_str, get_int, get_float, get_bool).
      5. Track which source provided each value (for debugging).
      6. Support config comparison between environments.

    Usage:
        manager = AgentConfigManager()
        manager.load_from_file("config.yaml")
        manager.load_from_env(prefix="AGENT_")
        manager.load_from_cli(args)
        config = manager.resolve()
        agent_config = manager.to_agent_config()

    Why a manager class instead of just functions:
      - Encapsulates the multi-step loading process.
      - Tracks source information for each value.
      - Provides a clean API for the rest of the application.
      - Makes testing easy — mock the manager, not os.environ.
    """

    def __init__(self):
        self._defaults: Dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)
        self._file_config: Dict[str, Any] = {}
        self._env_config: Dict[str, Any] = {}
        self._cli_config: Dict[str, Any] = {}
        self._resolved: Optional[Dict[str, Any]] = None
        self._sources: Dict[str, ConfigSource] = {}
        self._validation_errors: List[str] = []
        self._config_file_path: Optional[str] = None

    def load_from_file(self, filepath: str) -> "AgentConfigManager":
        """
        Load configuration from a YAML or JSON file.

        Returns self for method chaining:
            manager.load_from_file("config.yaml").load_from_env().resolve()
        """
        self._config_file_path = filepath
        try:
            if filepath.endswith(".json"):
                self._file_config = load_json_config(filepath)
            elif filepath.endswith((".yaml", ".yml")):
                self._file_config = load_yaml_config(filepath)
            else:
                self._validation_errors.append(
                    f"Unsupported config file format: {filepath}"
                )
        except (FileNotFoundError, ValueError) as e:
            self._validation_errors.append(f"Config file error: {e}")

        # Track sources
        self._track_sources(self._file_config, ConfigSource.FILE)
        self._resolved = None  # Invalidate cache
        return self

    def load_from_env(self, prefix: str = "AGENT_") -> "AgentConfigManager":
        """Load configuration from environment variables."""
        self._env_config = env_vars_to_config(prefix=prefix)
        self._track_sources(self._env_config, ConfigSource.ENV_VAR)
        self._resolved = None
        return self

    def load_from_cli(self, cli_overrides: Dict[str, Any]) -> "AgentConfigManager":
        """Load configuration from CLI argument overrides."""
        self._cli_config = cli_overrides
        self._track_sources(self._cli_config, ConfigSource.CLI_ARG)
        self._resolved = None
        return self

    def load_from_dict(self, data: Dict[str, Any]) -> "AgentConfigManager":
        """Load configuration from a programmatic dictionary."""
        # Merge into CLI config (same priority as programmatic overrides)
        self._cli_config = deep_merge(self._cli_config, data)
        self._track_sources(data, ConfigSource.PROGRAMMATIC)
        self._resolved = None
        return self

    def _track_sources(self, config: Dict[str, Any], source: ConfigSource,
                       prefix: str = "") -> None:
        """Recursively track which source provided each config key."""
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self._track_sources(value, source, full_key)
            else:
                self._sources[full_key] = source

    def resolve(self) -> Dict[str, Any]:
        """
        Resolve the final configuration by merging all sources.

        Priority: defaults → file → env → CLI (highest).
        """
        if self._resolved is not None:
            return self._resolved

        config = copy.deepcopy(self._defaults)
        config = deep_merge(config, self._file_config)
        config = deep_merge(config, self._env_config)
        config = deep_merge(config, self._cli_config)

        # Track default sources for keys not overridden
        self._track_sources(self._defaults, ConfigSource.DEFAULT)
        # Re-apply override sources (they take priority in tracking)
        self._track_sources(self._file_config, ConfigSource.FILE)
        self._track_sources(self._env_config, ConfigSource.ENV_VAR)
        self._track_sources(self._cli_config, ConfigSource.CLI_ARG)

        self._resolved = config
        return config

    def validate(self) -> List[str]:
        """
        Validate the resolved configuration.

        Returns a list of validation error messages (empty = valid).
        """
        config = self.resolve()
        errors = list(self._validation_errors)  # Start with load errors

        # Validate environment
        valid_envs = {"development", "staging", "production"}
        env = config.get("environment", "")
        if env not in valid_envs:
            errors.append(f"Invalid environment '{env}'. Must be one of {valid_envs}")

        # Validate model config
        model = config.get("model", {})
        temp = model.get("temperature", 0.7)
        if not isinstance(temp, (int, float)) or not 0.0 <= float(temp) <= 2.0:
            errors.append(f"temperature must be 0.0–2.0, got {temp}")

        max_tokens = model.get("max_tokens", 2048)
        if not isinstance(max_tokens, int) or max_tokens < 1:
            errors.append(f"max_tokens must be >= 1, got {max_tokens}")

        top_p = model.get("top_p", 1.0)
        if not isinstance(top_p, (int, float)) or not 0.0 <= float(top_p) <= 1.0:
            errors.append(f"top_p must be 0.0–1.0, got {top_p}")

        # Validate log level
        log_level = config.get("log_level", "INFO")
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if str(log_level).upper() not in valid_levels:
            errors.append(f"Invalid log_level '{log_level}'. Must be one of {valid_levels}")

        return errors

    def to_agent_config(self) -> AgentConfig:
        """Convert the resolved configuration to a typed AgentConfig."""
        return dict_to_agent_config(self.resolve())

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a config value by dot-notation key.

        Example: manager.get("model.temperature") → 0.7
        """
        config = self.resolve()
        keys = key.split(".")
        current: Any = config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    def get_str(self, key: str, default: str = "") -> str:
        """Get a config value as a string."""
        return str(self.get(key, default))

    def get_int(self, key: str, default: int = 0) -> int:
        """Get a config value as an integer."""
        return int(self.get(key, default))

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get a config value as a float."""
        return float(self.get(key, default))

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get a config value as a boolean."""
        return bool(self.get(key, default))

    def get_source(self, key: str) -> ConfigSource:
        """Get the source that provided a config value."""
        return self._sources.get(key, ConfigSource.DEFAULT)

    def get_source_report(self) -> Dict[str, str]:
        """Get a report of which source provided each config value."""
        return {key: source.value for key, source in sorted(self._sources.items())}

    def compare(self, other: "AgentConfigManager") -> Dict[str, Any]:
        """
        Compare two configurations and return differences.

        Useful for:
          - Comparing dev vs prod configs.
          - Auditing what changed between deployments.
          - Debugging why an agent behaves differently in staging.
        """
        self_config = self.resolve()
        other_config = other.resolve()

        differences: Dict[str, Any] = {}

        def compare_dicts(d1: Dict, d2: Dict, prefix: str = ""):
            all_keys = set(list(d1.keys()) + list(d2.keys()))
            for key in sorted(all_keys):
                full_key = f"{prefix}.{key}" if prefix else key
                v1 = d1.get(key)
                v2 = d2.get(key)
                if isinstance(v1, dict) and isinstance(v2, dict):
                    compare_dicts(v1, v2, full_key)
                elif v1 != v2:
                    differences[full_key] = {"self": v1, "other": v2}

        compare_dicts(self_config, other_config)
        return differences

    def to_json(self, indent: int = 2) -> str:
        """Export the resolved config as a JSON string."""
        return json.dumps(self.resolve(), indent=indent, default=str)

    def summary(self) -> str:
        """Generate a human-readable configuration summary."""
        config = self.resolve()
        model = config.get("model", {})
        lines = [
            f"Agent: {config.get('agent_name', 'unknown')} v{config.get('agent_version', '?')}",
            f"Environment: {config.get('environment', 'unknown')}",
            f"Model: {model.get('model_name', '?')} ({model.get('provider', '?')})",
            f"Temperature: {model.get('temperature', '?')}",
            f"Max Tokens: {model.get('max_tokens', '?')}",
            f"Log Level: {config.get('log_level', '?')}",
            f"Debug: {config.get('debug', False)}",
        ]
        return "\n".join(lines)


def demonstrate_config_manager() -> Dict[str, Any]:
    """Demonstrates the complete AgentConfigManager."""
    # --- Manager 1: Dev environment ---
    dev_manager = AgentConfigManager()

    # Simulate a config file by loading from dict (same effect)
    dev_manager.load_from_dict({
        "agent_name": "research-bot-dev",
        "environment": "development",
        "log_level": "DEBUG",
        "debug": True,
        "model": {
            "model_name": "gpt-4o-mini",
            "temperature": 0.9,
            "max_tokens": 1024,
        },
    })

    dev_config = dev_manager.resolve()
    dev_validation = dev_manager.validate()
    dev_typed = dev_manager.to_agent_config()

    # --- Manager 2: Prod environment ---
    prod_manager = AgentConfigManager()
    prod_manager.load_from_dict({
        "agent_name": "research-bot-prod",
        "environment": "production",
        "log_level": "WARNING",
        "debug": False,
        "model": {
            "model_name": "gpt-4o",
            "temperature": 0.3,
            "max_tokens": 4096,
        },
    })

    prod_config = prod_manager.resolve()
    prod_validation = prod_manager.validate()

    # Compare dev vs prod
    differences = dev_manager.compare(prod_manager)

    # Dot-notation access demo
    dot_access = {
        "model.temperature": dev_manager.get_float("model.temperature"),
        "model.model_name": dev_manager.get_str("model.model_name"),
        "debug": dev_manager.get_bool("debug"),
        "model.max_tokens": dev_manager.get_int("model.max_tokens"),
    }

    # Source tracking
    source_report = dev_manager.get_source_report()

    return {
        "dev_summary": dev_manager.summary(),
        "prod_summary": prod_manager.summary(),
        "dev_valid": len(dev_validation) == 0,
        "prod_valid": len(prod_validation) == 0,
        "differences": differences,
        "dot_access": dot_access,
        "source_report": source_report,
        "dev_typed_config": dev_typed,
        "json_export": dev_manager.to_json(),
    }


# ============================================================================
# RUNNER: Full Demo of Configuration Management & CLI
# ============================================================================

def run_config_cli_demo():
    """Runs the complete configuration management and CLI demonstration."""
    print("=" * 70)
    print("  Day 13/150 — Configuration Management & CLI")
    print("  Agents That Adapt to Any Environment")
    print("=" * 70)

    # ------------------------------------------------------------------
    # DEMO 1: Environment Variables
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DEMO 1: Environment Variables — Secrets & Runtime Config")
    print("=" * 70)

    env_result = demonstrate_env_vars()

    print("\n  Typed environment variable reading:")
    for key, value in env_result["typed_config"].items():
        display_val = value
        if key == "api_key" and isinstance(value, str) and len(value) > 8:
            display_val = value[:8] + "..." + value[-4:]  # Mask secrets
        print(f"    {key:<15} = {display_val!r:<30} (type: {type(value).__name__})")

    print(f"\n  .env file parsed ({len(env_result['dotenv_vars'])} variables loaded):")
    for key, value in env_result["dotenv_vars"].items():
        print(f"    {key} = {value}")

    print(f"\n  .env file content:")
    for line in env_result["dotenv_content"].split("\n"):
        print(f"    {line}")

    # ------------------------------------------------------------------
    # DEMO 2: Configuration Dataclasses
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DEMO 2: Configuration Dataclasses — Typed & Validated")
    print("=" * 70)

    dc_result = demonstrate_config_dataclasses()
    config = dc_result["config"]

    print(f"\n  AgentConfig (frozen=True, immutable):")
    print(f"    Name:        {config.agent_name}")
    print(f"    Version:     {config.agent_version}")
    print(f"    Environment: {config.environment}")
    print(f"    Log Level:   {config.log_level.value}")
    print(f"    Debug:       {config.debug}")

    print(f"\n  Nested ModelConfig:")
    print(f"    Model:       {config.model.model_name}")
    print(f"    Provider:    {config.model.provider.value}")
    print(f"    Temperature: {config.model.temperature}")
    print(f"    Max Tokens:  {config.model.max_tokens}")

    print(f"\n  Tools ({len(config.tools)} configured):")
    for tool in config.tools:
        status = "✓ enabled" if tool.enabled else "✗ disabled"
        print(f"    [{status}] {tool.name:<15} timeout={tool.timeout_seconds}s retries={tool.max_retries}")

    print(f"\n  Validation catches errors early:")
    for err in dc_result["validation_errors"]:
        print(f"    ✗ {err}")

    print(f"\n  Immutability protection:")
    print(f"    {dc_result['immutability_error']}")

    # ------------------------------------------------------------------
    # DEMO 3: YAML/JSON Config Files
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DEMO 3: YAML/JSON Config Files — External Configuration")
    print("=" * 70)

    file_result = demonstrate_config_files()

    print(f"\n  JSON config loaded from: {file_result['json_path']}")
    jc = file_result["json_config"]
    print(f"    Agent:       {jc.agent_name}")
    print(f"    Model:       {jc.model.model_name} ({jc.model.provider.value})")
    print(f"    Environment: {jc.environment}")
    print(f"    Tools:       {len(jc.tools)}")

    print(f"\n  YAML config parsed:")
    yc = file_result["yaml_config"]
    print(f"    Agent:       {yc.agent_name}")
    print(f"    Model:       {yc.model.model_name} ({yc.model.provider.value})")
    print(f"    Temperature: {yc.model.temperature}")
    print(f"    Max Tokens:  {yc.model.max_tokens}")

    print(f"\n  Parsed YAML dictionary keys:")
    for key, value in file_result["parsed_yaml_dict"].items():
        if isinstance(value, dict):
            print(f"    {key}: {{...}} ({len(value)} keys)")
        else:
            print(f"    {key}: {value!r}")

    # ------------------------------------------------------------------
    # DEMO 4: Configuration Hierarchy
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DEMO 4: Configuration Hierarchy — 4-Layer Resolution")
    print("=" * 70)

    hier_result = demonstrate_config_hierarchy()

    print(f"\n  Temperature value through each layer:")
    print(f"  {'─' * 50}")
    for layer_name, temp_val in hier_result["resolution_chain"]:
        arrow = "→" if "+" in layer_name else "●"
        print(f"    {arrow} {layer_name:<20} temperature = {temp_val}")

    print(f"\n  Key differences layer by layer:")
    l1 = hier_result["layer_1_defaults"]
    l4 = hier_result["layer_4_final"]
    compare_keys = [
        ("agent_name", l1.get("agent_name"), l4.get("agent_name")),
        ("environment", l1.get("environment"), l4.get("environment")),
        ("model.model_name", l1["model"]["model_name"], l4["model"]["model_name"]),
        ("model.temperature", l1["model"]["temperature"], l4["model"]["temperature"]),
        ("model.max_tokens", l1["model"]["max_tokens"], l4["model"]["max_tokens"]),
        ("debug", l1.get("debug"), l4.get("debug")),
    ]
    print(f"    {'Key':<25} {'Default':<20} {'Final':<20}")
    print(f"    {'─'*25} {'─'*20} {'─'*20}")
    for key, default, final in compare_keys:
        changed = " ← changed" if default != final else ""
        print(f"    {key:<25} {str(default):<20} {str(final):<20}{changed}")

    # ------------------------------------------------------------------
    # DEMO 5: argparse CLI
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DEMO 5: argparse CLI — Professional Command-Line Interface")
    print("=" * 70)

    cli_result = demonstrate_argparse_cli()

    print(f"\n  CLI commands parsed ({cli_result['num_commands_parsed']}):")
    for i, result in enumerate(cli_result["test_results"], 1):
        cmd = result["command"]
        overrides = result.get("config_overrides", {})
        print(f"\n    {i}. $ agent {cmd}")
        parsed = result.get("parsed", {})
        print(f"       Command:  {parsed.get('command', 'N/A')}")
        if overrides:
            print(f"       Overrides: {json.dumps(overrides, default=str)}")

    print(f"\n  Help text preview (first 15 lines):")
    for line in cli_result["help_text"].split("\n")[:15]:
        print(f"    {line}")

    # ------------------------------------------------------------------
    # DEMO 6: AgentConfigManager — Complete Integration
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DEMO 6: AgentConfigManager — Multi-Source Config Manager")
    print("=" * 70)

    mgr_result = demonstrate_config_manager()

    print(f"\n  Dev Environment:")
    for line in mgr_result["dev_summary"].split("\n"):
        print(f"    {line}")
    print(f"    Valid: {'✓' if mgr_result['dev_valid'] else '✗'}")

    print(f"\n  Prod Environment:")
    for line in mgr_result["prod_summary"].split("\n"):
        print(f"    {line}")
    print(f"    Valid: {'✓' if mgr_result['prod_valid'] else '✗'}")

    print(f"\n  Dev vs Prod Differences:")
    print(f"  {'─' * 60}")
    for key, diff in mgr_result["differences"].items():
        print(f"    {key:<30} dev={str(diff['self']):<15} prod={str(diff['other']):<15}")

    print(f"\n  Dot-notation access:")
    for key, value in mgr_result["dot_access"].items():
        print(f"    manager.get(\"{key}\") → {value!r}")

    print(f"\n  Source tracking (where each value came from):")
    for key, source in list(mgr_result["source_report"].items())[:8]:
        print(f"    {key:<30} ← {source}")

    typed_config = mgr_result["dev_typed_config"]
    print(f"\n  Typed AgentConfig from manager:")
    print(f"    agent_name: {typed_config.agent_name}")
    print(f"    model:      {typed_config.model.model_name}")
    print(f"    frozen:     {True}  (immutable after creation)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Day 13 Complete!")
    print("=" * 70)
    print("""
  Key Takeaways:
    1. Environment variables are the standard for secrets — NEVER hardcode API keys.
    2. Frozen dataclasses give typed, validated, immutable config objects.
    3. YAML/JSON files let teams define config without touching Python code.
    4. The 4-layer hierarchy (defaults → file → env → CLI) follows 12-Factor App.
    5. argparse builds professional CLIs with subcommands and typed arguments.
    6. AgentConfigManager ties everything together with source tracking.

  Why This Matters for Agents:
    - An agent deployed to dev/staging/prod needs different configs each time.
    - API keys rotate — env vars ensure no secrets in source control.
    - Operators need CLI flags to adjust behavior without redeploying.
    - Config comparison catches why an agent behaves differently in prod vs dev.

  Next: Day 14 will wrap up Phase 1 with a capstone integration project!
""")


if __name__ == "__main__":
    run_config_cli_demo()
