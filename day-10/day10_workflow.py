"""
🤖 Day 10/150 — Git Mastery & Professional Workflow
===================================================
Week 2: Advanced Python + Dev Tools

Concepts:
  1. Conventional Commits — semantic commit messages (feat:, fix:, docs:, ci:)
  2. Git Branch Naming — structured branches (feature/, bugfix/, hotfix/)
  3. Pre-Commit Hooks — automated local checks (Ruff linting/formatting)
  4. GitHub Actions CI/CD — automated remote tests & compliance

Why this matters for Agentic AI:
  - Long-running agents require structured version control to manage state schemas and tool files.
  - Automated CI/CD (GitHub Actions) ensures broken agent tools never reach main.
  - Pre-commit hooks keep code format clean before hitting git.
"""

import re
import os
import sys
import ast
import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ============================================================================
# SECTION 1: Git Branch Name Validator
# ============================================================================
# Why: Structured branch names help tools and teams identify the intent
# of code changes (e.g. feature/..., bugfix/..., hotfix/...).

class BranchValidationError(ValueError):
    """Raised when a branch name violates formatting rules."""
    pass


class BranchValidator:
    """Validates Git branch names according to professional standards."""
    
    ALLOWED_PREFIXES = ["feature", "bugfix", "hotfix", "release", "docs", "ci", "chore"]
    
    @classmethod
    def validate(cls, branch_name: str) -> bool:
        """
        Validates if a branch name is professionally structured.
        Rules:
          1. Must start with an allowed prefix followed by a slash (e.g., feature/)
          2. Must be all lowercase
          3. Cannot contain spaces or special characters (only letters, numbers, hyphens, and slashes)
          4. Cannot end with a slash or hyphen
        """
        if not branch_name:
            raise BranchValidationError("Branch name cannot be empty.")

        # Check prefix
        parts = branch_name.split("/")
        if len(parts) < 2 or not parts[0]:
            raise BranchValidationError(
                f"Branch must start with a prefix: {', '.join(cls.ALLOWED_PREFIXES)} followed by '/'"
            )
            
        prefix = parts[0]
        if prefix not in cls.ALLOWED_PREFIXES:
            raise BranchValidationError(
                f"Invalid prefix '{prefix}'. Allowed: {', '.join(cls.ALLOWED_PREFIXES)}"
            )

        # Check case
        if branch_name != branch_name.lower():
            raise BranchValidationError("Branch name must be strictly lowercase.")

        # Check characters
        pattern = r"^[a-z0-9]+(?:/[a-z0-9\-]+)+$"
        if not re.match(pattern, branch_name):
            raise BranchValidationError(
                "Branch name must only contain alphanumeric characters and hyphens after the prefix."
            )

        # Check ending
        if branch_name.endswith("-") or branch_name.endswith("/"):
            raise BranchValidationError("Branch name cannot end with a hyphen or slash.")

        return True


# ============================================================================
# SECTION 2: Conventional Commits Checker
# ============================================================================
# Why: Semantic commit messages make commit history readable, searchable,
# and allow tools to auto-generate changelogs/versions.

class CommitValidationError(ValueError):
    """Raised when a commit message violates the Conventional Commit spec."""
    pass


class CommitValidator:
    """Validates commit messages against Conventional Commit Specification 1.0.0."""
    
    TYPES = [
        "feat", "fix", "docs", "style", "refactor", 
        "perf", "test", "build", "ci", "chore", "revert"
    ]
    
    @classmethod
    def validate(cls, commit_msg: str) -> Tuple[str, Optional[str], str]:
        """
        Validates a commit message header.
        Format: <type>(<scope>): <description> or <type>: <description>
        
        Returns:
            Tuple of (type, scope, description) if valid.
        """
        if not commit_msg:
            raise CommitValidationError("Commit message header cannot be empty.")
            
        lines = commit_msg.splitlines()
        header = lines[0].strip()
        
        # Regex for Conventional Commits
        # Group 1: type, Group 2: optional scope, Group 3: description
        pattern = r"^([a-z]+)(?:\(([^)]+)\))?\s*:\s+(.+)$"
        match = re.match(pattern, header)
        
        if not match:
            raise CommitValidationError(
                "Message must follow format: '<type>: <desc>' or '<type>(<scope>): <desc>'"
            )
            
        commit_type, scope, description = match.groups()
        
        # Validate type
        if commit_type not in cls.TYPES:
            raise CommitValidationError(
                f"Invalid commit type '{commit_type}'. Allowed types: {', '.join(cls.TYPES)}"
            )
            
        # Validate description length
        if len(description) < 10:
            raise CommitValidationError(
                f"Description too short ({len(description)} chars). Must be at least 10 characters."
            )
            
        if len(header) > 72:
            raise CommitValidationError(
                f"Commit header too long ({len(header)} chars). Must be 72 characters or less."
            )
            
        return commit_type, scope, description


# ============================================================================
# SECTION 3: AST Code Linter & Pre-Commit Simulator
# ============================================================================
# Why: Pre-commit hooks run checkers locally. Instead of running external CLI tools
# in this script, we build a mini-linter using Python's 'ast' library to check
# for compliance (missing type hints, missing docstrings).

@dataclass
class LintIssue:
    line_no: int
    column: int
    code: str  # E.g., E101, W201
    message: str
    severity: str = "warning"


class ASTCodeLinter(ast.NodeVisitor):
    """
    Parses a python file's abstract syntax tree (AST) to check:
      1. Missing docstrings in classes/functions (Code: D100, D101, D102)
      2. Missing type hints in function arguments/returns (Code: T201, T202)
    """
    def __init__(self, filename: str):
        self.filename = filename
        self.issues: List[LintIssue] = []
        self._source_lines: List[str] = []

    def check_file(self, file_content: str) -> List[LintIssue]:
        self.issues = []
        self._source_lines = file_content.splitlines()
        
        try:
            tree = ast.parse(file_content, filename=self.filename)
            
            # Run D100 (module-level docstring)
            if not ast.get_docstring(tree):
                self.issues.append(LintIssue(
                    line_no=1, column=0, code="D100", 
                    message="Missing module docstring at the top of the file."
                ))
                
            self.visit(tree)
        except SyntaxError as e:
            self.issues.append(LintIssue(
                line_no=e.lineno or 1, column=e.offset or 0, code="E901",
                message=f"Syntax Error: {e.msg}", severity="error"
            ))
            
        return self.issues

    def visit_ClassDef(self, node: ast.ClassDef):
        # Rule D101: Class docstring
        if not ast.get_docstring(node):
            self.issues.append(LintIssue(
                line_no=node.lineno, column=node.col_offset, code="D101",
                message=f"Class '{node.name}' is missing a docstring."
            ))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Ignore private/magic functions (e.g. __init__ doesn't strictly need a docstring)
        is_magic = node.name.startswith("__") and node.name.endswith("__")
        
        # Rule D102: Function docstring
        if not is_magic and not ast.get_docstring(node):
            self.issues.append(LintIssue(
                line_no=node.lineno, column=node.col_offset, code="D102",
                message=f"Function/Method '{node.name}' is missing a docstring."
            ))
            
        # Rule T201 & T202: Type hints check
        # Check arguments type hints
        for arg in node.args.args:
            if arg.arg == "self" or arg.arg == "cls":
                continue
            if not arg.annotation:
                self.issues.append(LintIssue(
                    line_no=arg.lineno, column=arg.col_offset, code="T201",
                    message=f"Argument '{arg.arg}' in function '{node.name}' is missing a type hint."
                ))
                
        # Check return type hint
        if not node.returns and node.name != "__init__":
            self.issues.append(LintIssue(
                line_no=node.lineno, column=node.col_offset, code="T202",
                message=f"Function/Method '{node.name}' is missing a return type hint."
            ))
            
        self.generic_visit(node)


# ============================================================================
# SECTION 4: CI/CD Pipeline Simulator
# ============================================================================
# Why: CI/CD validates code remotely. This simulator compiles all our checks
# and prints them out formatted like a real GitHub Actions runner terminal log.

class CISimulator:
    """Simulates a professional CI/CD runner executing workflow check stages."""
    
    def __init__(self, branch: str, commits: List[str], code_files: Dict[str, str]):
        self.branch = branch
        self.commits = commits
        self.code_files = code_files
        self.success = True

    def log_header(self, title: str):
        print("\n" + "=" * 80)
        print(f"⚙️  STAGE: {title.upper()}")
        print("=" * 80)

    def run_pipeline(self) -> bool:
        print("🚀 Starting simulated GitHub Actions runner...")
        print(f"📌 Runner info: OS=ubuntu-latest, Branch={self.branch}, CommitsCount={len(self.commits)}")
        time.sleep(0.5)

        # STAGE 1: Git Branch Check
        self.log_header("Git Branch Name Validation")
        try:
            BranchValidator.validate(self.branch)
            print(f"  ✅ Branch name '{self.branch}' conforms to standards.")
        except BranchValidationError as e:
            print(f"  ❌ Branch check FAILED: {e}")
            self.success = False

        # STAGE 2: Commit Messages Check
        self.log_header("Conventional Commits Check")
        for i, commit in enumerate(self.commits, 1):
            try:
                commit_type, scope, desc = CommitValidator.validate(commit)
                scope_str = f"({scope})" if scope else ""
                print(f"  ✅ Commit #{i}: Type={commit_type}{scope_str} | Message='{desc}'")
            except CommitValidationError as e:
                print(f"  ❌ Commit #{i} FAILED: '{commit}'")
                print(f"     Reason: {e}")
                self.success = False

        # STAGE 3: Pre-commit Linter Check
        self.log_header("Automated Code Linting (AST Parser)")
        for filepath, content in self.code_files.items():
            print(f"  Checking '{filepath}'...")
            linter = ASTCodeLinter(filepath)
            issues = linter.check_file(content)
            
            if not issues:
                print(f"    ✅ '{filepath}' passed all AST quality checks.")
            else:
                for issue in issues:
                    emoji = "🚨 ERROR" if issue.severity == "error" else "⚠️  WARN"
                    print(f"    {emoji} [Line {issue.line_no}:{issue.column}] [{issue.code}] {issue.message}")
                    if issue.severity == "error":
                        self.success = False
                    else:
                        # Warnings can warn in pre-commit but let's count them
                        # in our simulator as CI checks to guide the student.
                        self.success = False

        # STAGE 4: CI Status Report
        print("\n" + "=" * 80)
        print("📊 PIPELINE RUN SUMMARY")
        print("=" * 80)
        if self.success:
            print("  🎉 STATUS: SUCCESS")
            print("  All verification stages passed! Safe to merge into 'main'.")
        else:
            print("  ❌ STATUS: FAILED")
            print("  Please fix the lints, commit formatting, or branch name before merging.")
        print("=" * 80 + "\n")
        
        return self.success


# ============================================================================
# RUNNER: Demonstrations
# ============================================================================

def run_workflow_demo():
    """Runs tests and simulates CI logs for student verification."""
    print("🤖 Git Mastery & Professional Workflow Simulation")
    print("===================================================\n")
    
    # 1. Show branch validator behavior
    print("🧪 Testing Branch Name Validator:")
    valid_branches = ["feature/add-precommit", "bugfix/issue-42", "docs/update-readme"]
    invalid_branches = ["Feature/Add-Precommit", "feature-add-precommit", "bugfix/issue-42-"]
    
    for b in valid_branches:
        try:
            BranchValidator.validate(b)
            print(f"  ✅ Branch '{b}' is valid.")
        except BranchValidationError:
            print(f"  ❌ Branch '{b}' should be valid but failed!")
            
    for b in invalid_branches:
        try:
            BranchValidator.validate(b)
            print(f"  ❌ Branch '{b}' passed but should be invalid!")
        except BranchValidationError as e:
            print(f"  ✅ Branch '{b}' is correctly invalid. Error: {e}")
            
    # 2. Show commit message validator
    print("\n🧪 Testing Conventional Commits Validator:")
    valid_commits = [
        "feat(core): implement robust agent retry system",
        "fix: resolve rate-limit exception crash",
        "docs(readme): update day 10 details"
    ]
    invalid_commits = [
        "fixed rate limit crash",  # No type
        "feat: short",             # Desc too short
        "unknownType: commit desc test msg"  # Invalid type
    ]
    
    for c in valid_commits:
        try:
            CommitValidator.validate(c)
            print(f"  ✅ Commit '{c}' is valid.")
        except CommitValidationError:
            print(f"  ❌ Commit '{c}' should be valid but failed!")
            
    for c in invalid_commits:
        try:
            CommitValidator.validate(c)
            print(f"  ❌ Commit '{c}' passed but should be invalid!")
        except CommitValidationError as e:
            print(f"  ✅ Commit '{c}' is correctly invalid. Error: {e}")

    # 3. Code files to lint (One passing, one failing)
    passing_code = '''"""
Module docstring for valid agent tools.
"""

def process_agent_output(raw_text: str) -> dict:
    """Parses raw text and returns dict representation."""
    return {"data": raw_text.strip()}
'''

    failing_code = '''# No docstring at module level
def process_agent_output(raw_text):
    # No docstring, no type hints
    return {"data": raw_text}
'''

    # Run CI Simulation (Failing Branch Example)
    print("\n" + "#" * 60)
    print("🚩 RUNNING PIPELINE SIMULATION #1: FAILED COMPLIANCE RUN")
    print("#" * 60)
    
    bad_commits = [
        "feat: add tools",  # Too short description
        "fixed formatting bug"  # Non-conventional
    ]
    
    sim_fail = CISimulator(
        branch="Feature/bad-branch-Name",  # uppercase and slash
        commits=bad_commits,
        code_files={
            "agent_utils.py": failing_code
        }
    )
    sim_fail.run_pipeline()
    
    # Run CI Simulation (Passing Branch Example)
    print("\n" + "#" * 60)
    print("🏁 RUNNING PIPELINE SIMULATION #2: SUCCESS COMPLIANCE RUN")
    print("#" * 60)
    
    good_commits = [
        "feat(cli): add formatting utilities",
        "docs(readme): add execution commands to day 10"
    ]
    
    sim_pass = CISimulator(
        branch="feature/good-branch-name",
        commits=good_commits,
        code_files={
            "agent_utils.py": passing_code
        }
    )
    sim_pass.run_pipeline()


if __name__ == "__main__":
    run_workflow_demo()
