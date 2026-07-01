# 🤖 Day 13/150 — Configuration Management & CLI — Agents That Adapt

![Day](https://img.shields.io/badge/Day-13%2F150-blue?style=flat-square)
![Phase](https://img.shields.io/badge/Phase-1%3A%20Foundations-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11-3572a5?style=flat-square&logo=python)

> **Key Insight:** An agent with hardcoded config is a prototype.
> An agent with a config hierarchy (defaults → file → env → CLI) is production-ready.
> The 12-Factor App methodology isn't optional for agents — it's the standard.

---

## 📌 What I Learned Today

| Concept | What It Does | Agent Application |
|---------|-------------|-------------------|
| **Environment Variables** | Load secrets and runtime config from OS/env | API keys, model names — never hardcoded, never committed |
| **dotenv Parsing** | Read `.env` files for local development | Zero-dependency secrets loading with override control |
| **Config Dataclasses** | Typed, validated, immutable config objects | `AgentConfig(frozen=True)` catches misconfig at startup |
| **YAML/JSON Loading** | External structured config files | Model settings, tool configs, memory params — all in one file |
| **Config Hierarchy** | Merge: defaults → file → env → CLI | Teams override only what they need at each layer |
| **argparse CLI** | Professional command-line interfaces | `python agent.py run --model gpt-4 --temperature 0.7` |

---

## 🔨 What I Built

### `day13_config_cli.py`
A complete configuration management system for AI agents:

- **`get_env()` / `get_env_int()` / `get_env_float()` / `get_env_bool()`** — Type-safe env var readers
- **`load_dotenv()`** — Zero-dependency .env file parser with override control
- **`AgentConfig` dataclass** — Frozen, validated, typed configuration object
- **`load_yaml_config()` / `load_json_config()`** — External config file loaders
- **`ConfigHierarchy`** — Merges all config sources with correct priority
- **`build_agent_cli()`** — Full argparse CLI with subcommands (run, config, test)
- **`AgentConfigManager`** — Production-ready manager combining all sources

### `sample_config.yaml`
A reference YAML configuration file showing agent, model, tools, memory, logging, and retry settings.

---

## 📂 Code Highlights

### Type-Safe Environment Variables
```python
def get_env(key: str, default=None, required=False) -> str:
    value = os.environ.get(key)
    if value is not None:
        return value
    if required and default is None:
        raise ValueError(f"Required env var '{key}' not set")
    return default or ""
```

### Frozen Config Dataclass
```python
@dataclass(frozen=True)
class AgentConfig:
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 2048
    tools: tuple = ("web_search", "calculator")

    def __post_init__(self):
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be 0.0-2.0")
```

### Configuration Hierarchy
```python
# Priority: defaults < file < env < CLI (highest)
final_config = {**defaults, **file_config, **env_config, **cli_args}
```

---

## ▶️ Run It

```bash
cd agentic-ai-journey/day-13
python day13_config_cli.py
```

---

## 🧠 Why This Matters for Agents

1. **Security**: API keys in source code = leaked keys. Env vars are the industry standard.
2. **Flexibility**: Same agent code runs in dev, staging, prod — only config changes.
3. **Team Workflows**: Shared defaults + personal `.env` overrides = no conflicts.
4. **12-Factor App**: Config in the environment, not in the code. This is how Docker, K8s, and CI/CD work.
5. **Operator Control**: CLIs let non-developers run agents with `--model gpt-4 --verbose`.

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| Python argparse | https://docs.python.org/3/library/argparse.html |
| Python dataclasses | https://docs.python.org/3/library/dataclasses.html |
| 12-Factor App Config | https://12factor.net/config |
| PyYAML docs | https://pyyaml.org/wiki/PyYAMLDocumentation |
| python-dotenv | https://pypi.org/project/python-dotenv/ |
| Python os.environ | https://docs.python.org/3/library/os.html#os.environ |
