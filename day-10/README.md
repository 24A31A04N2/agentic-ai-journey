# 🤖 Day 10/150 — Git Mastery & Professional Workflow

![Day](https://img.shields.io/badge/Day-10%2F150-blue?style=flat-square)
![Phase](https://img.shields.io/badge/Phase-2%3A%20Advanced%20Python-orange?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11-3572a5?style=flat-square&logo=python)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub--Actions-3fb950?style=flat-square&logo=github-actions)

> **Key Insight:** Raw code without quality control is a liability.
> Git structures your history, Pre-Commit shields your codebase locally, and CI/CD defends your main branch remotely.

---

## 📌 What I Learned Today

| Concept | What It Does | Agent Application |
|---------|-------------|-------------------|
| **Conventional Commits** | Semantic prefixes (`feat:`, `fix:`, `docs:`) | Generate changelogs and trace fixes in multi-agent tool setups |
| **Git Branching (GitFlow)** | Strict branch naming conventions | Keep production main stable while testing experimental agents |
| **Pre-Commit Hooks** | Runs checks locally before commits are allowed | Autodetect formatting errors and missing type-hints in tools |
| **GitHub Actions (CI/CD)** | Automates builds, lints, and test suites on push/PR | Never merge broken code; test tools dynamically on runner VMs |
| **AST Parser Linting** | Inspections via abstract syntax trees | Build custom code scanners verifying type safety & docstring compliance |

---

## 🔨 What I Built

### 1. Git & Code Compliance Simulator (`day10_workflow.py`)
A comprehensive validation suite designed to model a modern automated release pipeline:
- **`BranchValidator`**: Checks branch names against lowercase prefix standards (e.g. `feature/add-llm`, `bugfix/rate-limit`).
- **`CommitValidator`**: Parses commit headers according to Conventional Commit Specification 1.0.0.
- **`ASTCodeLinter`**: Visits Abstract Syntax Tree nodes to find missing class/method docstrings or missing argument/return type-hints.
- **`CISimulator`**: Implements stage logging simulating a GitHub Actions container run, validating commits and code compliance.

### 2. Repository Configuration files
- [**.github/workflows/ci.yml**](../.github/workflows/ci.yml): Automates `ruff` lint check, formatting verification, and day-09 pytest validation.
- [**.pre-commit-config.yaml**](../.pre-commit-config.yaml): Installs local hooks checking yaml syntax, trailing whitespace, and running ruff autofixes.

---

## 📂 Code Highlights

### Conventional Commits Check
```python
# Regex parsing type, optional scope, and description
pattern = r"^([a-z]+)(?:\(([^)]+)\))?\s*:\s+(.+)$"
match = re.match(pattern, header)

if not match:
    raise CommitValidationError("Format: '<type>: <desc>' or '<type>(<scope>): <desc>'")
```

### Pre-commit AST Validator (Type Hint Checker)
```python
def visit_FunctionDef(self, node: ast.FunctionDef):
    # Check arguments type hints
    for arg in node.args.args:
        if arg.arg in ("self", "cls"):
            continue
        if not arg.annotation:
            self.issues.append(LintIssue(
                line_no=arg.lineno, code="T201",
                message=f"Argument '{arg.arg}' in function '{node.name}' is missing type hint."
            ))
```

---

## ▶️ Run It

```bash
cd agentic-ai-journey/day-10
python day10_workflow.py
```

---

## 🔗 Resources

| Resource | Link |
|----------|------|
| Conventional Commits | https://www.conventionalcommits.org/ |
| pre-commit framework | https://pre-commit.com/ |
| GitHub Actions Docs | https://docs.github.com/en/actions |
| Ruff Python Linter | https://docs.astral.sh/ruff/ |
