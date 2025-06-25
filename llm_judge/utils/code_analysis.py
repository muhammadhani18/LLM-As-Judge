"""
Code analysis utilities for syntax checking, style analysis, and security assessment.
"""
import py_compile
import re
import subprocess
import tempfile
import os
import json
from typing import Dict, Any
from radon.complexity import cc_visit

class CodeAnalyzer:
    """Handles code analysis including syntax, style, and security checks."""
    
    def __init__(self):
        pass
    
    def run_linter(self, code: str) -> Dict[str, Any]:
        """
        Run syntax and security analysis on code.
        
        Returns:
            Dict containing syntax_ok, security_issues, and bandit_output
        """
        # 1) Syntax check
        syntax_ok = True
        try:
            # Write code to a temp file so that Bandit can also see it
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(code)
                filename = f.name

            # Use py_compile to check syntax
            py_compile.compile(filename, doraise=True)
        except py_compile.PyCompileError as e:
            syntax_ok = False
        finally:
            # we'll let Bandit read the same file
            snippet_path = filename

        # 2) Security analysis via Bandit
        bandit_proc = subprocess.run(
            ["bandit", "-r", snippet_path, "-f", "json"],
            capture_output=True,
            text=True
        )
        try:
            bandit_report = json.loads(bandit_proc.stdout or "{}")
            security_issues = len(bandit_report.get("results", []))
        except json.JSONDecodeError:
            security_issues = 0

        # Clean up
        try:
            os.remove(snippet_path)
        except OSError:
            pass

        return {
            "syntax_ok": syntax_ok,
            "security_issues": security_issues,
            "bandit_output": bandit_proc.stdout,
        }
    
    def run_style_analysis(self, code: str) -> Dict[str, Any]:
        """
        Run style analysis including pylint and cyclomatic complexity.
        
        Returns:
            Dict containing pylint_score, average_complexity, and style_score
        """
        # 1) Write code to temp file
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code)
            path = f.name

        # 2) Pylint: get overall score
        pylint_proc = subprocess.run(
            ["pylint", path, "--disable=all", "--enable=score"],
            capture_output=True, text=True
        )
        # extract "Your code has been rated at 8.50/10"
        match = re.search(r"rated at ([0-9\.]+)/10", pylint_proc.stdout or "")
        pylint_score = float(match.group(1)) if match else 0.0

        # 3) Radon cyclomatic complexity
        try:
            blocks = cc_visit(code)
            # average complexity across functions
            complexities = [b.complexity for b in blocks]
            avg_cc = sum(complexities) / len(complexities) if complexities else 0.0
        except Exception:
            avg_cc = 0.0

        # 4) Normalize style: penalize high complexity
        # Let's define style_score = (pylint_score/10) * (1 / (1 + avg_cc/10))
        style_score = (pylint_score / 10) * (1 / (1 + avg_cc / 10))

        # Clean up
        try: 
            os.remove(path)
        except OSError: 
            pass

        return {
            "pylint_score": round(pylint_score, 2),
            "average_complexity": round(avg_cc, 2),
            "style_score": round(style_score, 3)
        } 