# llm_judge/llm_judge/code_pipeline.py
import py_compile

import subprocess
import tempfile
import os
import json
from typing import Dict, Any, List, Tuple

import openai
from .evaluator import Evaluator

# Configure your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

class CodePipeline(Evaluator):
    def __init__(self, models: List[str], judge_model: str = "gpt-4"):
        self.models = models
        self.judge_model = judge_model
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def evaluate(self, prompt: str, responses: Dict[str, str] = None) -> Dict[str, Any]:
        generated = responses or self._invoke_models(prompt)
        scores = {}

        for name, code in generated.items():
            print(f"\n🔍 Evaluating model: {name}")
            lint_info = self._run_linter(code)

            if not lint_info["syntax_ok"]:
                scores[name] = {
                    "status": "syntax_error",
                    "score": 0,
                    "details": lint_info
                }
                continue

            try:
                print("🤖 Asking Judge LLM to generate tests...")
                tests = self._generate_tests(prompt, code)

                print("⚙️ Running unit tests...")
                passed, failed = self._run_tests(code, tests)

                total = passed + failed
                pass_rate = passed / total if total > 0 else 0
                security_penalty = lint_info["security_issues"] * 0.1
                final_score = round(max(pass_rate - security_penalty, 0), 3)

                scores[name] = {
                    "status": "evaluated",
                    "score": final_score,
                    "tests": {"passed": passed, "failed": failed},
                    "lint": lint_info
                }
            except Exception as e:
                scores[name] = {
                    "status": "error",
                    "score": 0,
                    "error": str(e)
                }

        # Pick winner
        best = max(scores.items(), key=lambda kv: kv[1]["score"] if kv[1]["status"] == "evaluated" else -1, default=(None, {}))
        winner = best[0] if best[0] else "No valid code found"

        return {
            "prompt": prompt,
            "results": scores,
            "winner": winner
        }

    def _invoke_models(self, prompt: str) -> Dict[str, str]:
        """Call each model’s API to get Python code. Returns model_name→code."""
        responses = {}
        for model in self.models:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Generate a Python function only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
            )
            code = resp.choices[0].message.content
            responses[model] = code
        return responses

    def _run_linter(self, code: str) -> Dict[str, Any]:
        """
        1. Use Python's compile() to check syntax.
        2. Use Bandit for security issues.
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

    def _generate_tests(self, prompt: str, code: str) -> str:
        """Ask the judge LLM to write pytest tests for this code snippet."""
        test_prompt = f"""
You are an expert test writer. Given the prompt:
{prompt}

And this Python function:
{code}

Write pytest-style unit tests that thoroughly exercise edge cases.
Return only the test file content.
"""
        resp = openai.ChatCompletion.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content

    def _run_tests(self, code: str, tests: str) -> Tuple[int, int]:
        """Write code + tests to temp dir, run pytest, and return (passed, failed)."""
        with tempfile.TemporaryDirectory() as td:
            # Write snippet and tests
            snippet_path = os.path.join(td, "snippet.py")
            test_path = os.path.join(td, "test_snippet.py")
            with open(snippet_path, "w") as f:
                f.write(code)
            with open(test_path, "w") as f:
                f.write(tests)

            # Execute pytest
            proc = subprocess.run(
                ["pytest", "--maxfail=1", "--disable-warnings", "--json-report", "--json-report-file=report.json"],
                cwd=td,
                capture_output=True,
                text=True
            )
            report_path = os.path.join(td, "report.json")
            if not os.path.exists(report_path):
                return 0, 0

            report = json.loads(open(report_path).read())
            passed = report.get("summary", {}).get("passed", 0)
            failed = report.get("summary", {}).get("failed", 0)
            return passed, failed
