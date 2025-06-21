# llm_judge/llm_judge/code_pipeline.py
import py_compile
import re
import subprocess
import tempfile
import os
import json
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
import openai
from .evaluator import Evaluator
from openai import OpenAI

load_dotenv()

# Configure your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI() 


class CodePipeline(Evaluator):
    def __init__(self, models: List[str], judge_model: str = "gpt-4"):
        self.models = models
        self.judge_model = judge_model
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def evaluate(self, prompt: str, responses: Dict[str, str] = None) -> Dict[str, Any]:
        generated = responses or self._invoke_models(prompt)
        scores = {}
        
        print("\n📦 Generated responses:")
        for name, code in generated.items():
            print(f"\n{name}:\n{code}\n{'-'*40}")

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
        responses = {}
        for model in self.models:
            print(f"🎯 Invoking {model}...")
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Only return valid Python code. No markdown or text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
            )
            raw = resp.choices[0].message.content

            # Clean out markdown and extra explanation
            cleaned = self._extract_code(raw)
            responses[model] = cleaned
        return responses

    def _extract_code(self, text: str) -> str:
        # Removes triple backticks and any leading explanations
        # Handles ```python ... ``` or ``` ...
        code_blocks = re.findall(r"```(?:python)?(.*?)```", text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        return text.strip()

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
        print(f"prompt: {prompt}")
        print(f"code: {code}")
        test_prompt = f"""
You are an expert test writer. Given the prompt:
{prompt}

And this Python function:
{code}

Write a concise test function with exactly 10 test cases (5 valid, 5 invalid) that test key edge cases.
IMPORTANT: 
- Do NOT include any import statements
- Keep it very concise - maximum 10 test cases total
- Return only the test function content without any markdown formatting
- Focus on the most important edge cases

Example structure:
```python
def test_validate_email():
    # 5 valid cases
    assert validate_email('test@example.com') == True
    assert validate_email('user.name@domain.co.uk') == True
    assert validate_email('test+tag@example.com') == True
    assert validate_email('123@example.com') == True
    assert validate_email('test@sub.domain.com') == True
    
    # 5 invalid cases  
    assert validate_email('invalid') == False
    assert validate_email('test@') == False
    assert validate_email('test@.com') == False
    assert validate_email('test@example') == False
    assert validate_email('test@example..com') == False
```

Return ONLY the test function, no imports, no markdown, exactly 10 test cases.
"""
        response = client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0,
            max_tokens=500  # Limit the response size even more
        )
        tests = response.choices[0].message.content
        
        # Clean up the response to remove any markdown and imports
        cleaned_tests = self._extract_code(tests)
        # Remove any import statements
        cleaned_tests = re.sub(r'^import.*\n?', '', cleaned_tests, flags=re.MULTILINE)
        cleaned_tests = re.sub(r'^from.*import.*\n?', '', cleaned_tests, flags=re.MULTILINE)
        
        print(f"test cases: {cleaned_tests}")
        return cleaned_tests

    def _run_tests(self, code: str, tests: str) -> Tuple[int, int]:
        """Write code + tests to temp dir, run tests directly, and return (passed, failed)."""
        with tempfile.TemporaryDirectory() as td:
            # Create a combined test file that includes both the function and tests
            combined_test_content = f"""
{code}

def _run_individual_tests():
    #Run each assertion individually to count passed/failed.
    passed = 0
    failed = 0
    
    # Define test cases to run individually
    test_cases = [
        # Valid cases
        ('test@example.com', True),
        ('user.name@domain.co.uk', True),
        ('test+tag@example.com', True),
        ('123@example.com', True),
        ('test@sub.domain.com', True),
        # Invalid cases
        ('invalid', False),
        ('test@', False),
        ('test@.com', False),
        ('test@example', False),
        ('test@example..com', False),
    ]
    
    for email, expected in test_cases:
        try:
            result = validate_email(email)
            if result == expected:
                passed += 1
                print(f"PASS: {{email}} returned {{result}}")
            else:
                failed += 1
                print(f"FAIL: {{email}} returned {{result}}, expected {{expected}}")
        except Exception as e:
            failed += 1
            print(f"ERROR: {{email}} - {{e}}")
    
    return passed, failed

{self._extract_code(tests)}

# Add test runner
if __name__ == "__main__":
    passed = 0
    failed = 0
    total_tests = 0
    
    # Count total assertions in the test function
    test_lines = {repr(self._extract_code(tests))}.split('\\n')
    for line in test_lines:
        if line.strip().startswith('assert '):
            total_tests += 1
    
    print(f"Running {{total_tests}} test cases...")
    
    try:
        test_validate_email()
        print("All tests passed!")
        passed = total_tests
        failed = 0
    except AssertionError as e:
        print(f"Test failed: {{e}}")
        print("Running individual tests to get detailed results...")
        # Try to run tests individually to count passed/failed
        passed, failed = _run_individual_tests()
    except Exception as e:
        print(f"Error running tests: {{e}}")
        failed = total_tests
        passed = 0
    
    print(f"PASSED: {{passed}}")
    print(f"FAILED: {{failed}}")
"""
            
            test_path = os.path.join(td, "test_snippet.py")
            with open(test_path, "w") as f:
                f.write(combined_test_content)

            print(f"Test file content:\n{combined_test_content}")

            # Execute the test file directly with Python
            proc = subprocess.run(
                ["python", test_path],
                cwd=td,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            print(f"Python return code: {proc.returncode}")
            print(f"Python stdout: {proc.stdout}")
            print(f"Python stderr: {proc.stderr}")
            
            # Parse the output to get test results
            passed = 0
            failed = 0
            
            if "PASSED:" in proc.stdout:
                for line in proc.stdout.split('\n'):
                    if line.startswith("PASSED:"):
                        passed = int(line.split(":")[1].strip())
                    elif line.startswith("FAILED:"):
                        failed = int(line.split(":")[1].strip())
            
            print(f"Parsed results: passed={passed}, failed={failed}")
            return passed, failed
