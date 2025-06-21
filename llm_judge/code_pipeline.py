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
import re
from radon.complexity import cc_visit

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
            style_info = self._run_style_analysis(code)    
            if not lint_info["syntax_ok"]:
                scores[name] = {
                    "status": "syntax_error",
                    "score": 0,
                    "details": lint_info,
                    "style": style_info 
                }
                continue
            else:
                tests = self._generate_tests(prompt, code)
                passed, failed = self._run_tests(code, tests)

                total = passed + failed
                pass_rate = passed / total if total > 0 else 0
                security_penalty = lint_info["security_issues"] * 0.1

                # combine functionality & style
                w_func, w_style = 0.7, 0.3
                func_comp = max(pass_rate - security_penalty, 0)
                style_comp = style_info["style_score"]
                final_score = round(w_func * func_comp + w_style * style_comp, 3)

                scores[name] = {
                    "status": "evaluated",
                    "score": final_score,
                    "tests": {"passed": passed, "failed": failed},
                    "lint": lint_info,
                    "style": style_info
                }
                
            try:
                print("🤖 Asking Judge LLM to generate tests...")
                tests = self._generate_tests(prompt, code)

                print("⚙️ Running unit tests...")
                passed, failed = self._run_tests(code, tests)

                total = passed + failed
                pass_rate = passed / total if total > 0 else 0
                security_penalty = lint_info["security_issues"] * 0.1
                                # weights (tune as you wish)
                w_func = 0.7    # weight for functionality (tests & security)
                w_style = 0.3   # weight for style

                func_component = max(pass_rate - security_penalty, 0)
                style_component = style_info["style_score"]

                combined_score = round((w_func * func_component) + (w_style * style_component), 3)

                final_score = combined_score


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
    
    def _run_style_analysis(self, code: str) -> Dict[str, Any]:
        """
        Returns:
          - pylint_score: float (0–10)
          - average_complexity: float (cyclomatic complexity)
          - style_score: float (0–1), higher is better
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
        # extract “Your code has been rated at 8.50/10”
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
        # Let’s define style_score = (pylint_score/10) * (1 / (1 + avg_cc/10))
        style_score = (pylint_score / 10) * (1 / (1 + avg_cc / 10))

        # Clean up
        try: os.remove(path)
        except OSError: pass

        return {
            "pylint_score": round(pylint_score, 2),
            "average_complexity": round(avg_cc, 2),
            "style_score": round(style_score, 3)
        }

    def _generate_tests(self, prompt: str, code: str) -> str:
        """Ask the judge LLM to write pytest tests for this code snippet."""
        
        test_prompt = f"""
You are an expert test writer. Given the prompt:
{prompt}

And this Python function:
{code}

Write a concise test function with exactly 10 test cases (5 valid, 5 invalid) that test key edge cases.
CRITICAL REQUIREMENTS:
- Do NOT include any import statements
- Do NOT include any markdown formatting (no ```python or ```)
- Keep it very concise - maximum 10 test cases total
- Return only the test function content
- Focus on the most important edge cases
- Start with "def test_" followed by a descriptive name

Example structure:
def test_function_name():
    # 5 valid cases
    assert function_call('valid_input') == expected_output
    assert function_call('another_valid') == expected_output
    
    # 5 invalid cases  
    assert function_call('invalid_input') == expected_output
    assert function_call('another_invalid') == expected_output

Return ONLY the test function definition and body, no imports, no markdown, exactly 10 test cases.
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
        # Remove any remaining markdown
        cleaned_tests = re.sub(r'^```.*\n?', '', cleaned_tests, flags=re.MULTILINE)
        cleaned_tests = re.sub(r'^```$', '', cleaned_tests, flags=re.MULTILINE)
        
        print(f"test cases: {cleaned_tests}")
        return cleaned_tests

    def _run_tests(self, code: str, tests: str) -> Tuple[int, int]:
        """Write code + tests to temp dir, run tests directly, and return (passed, failed)."""
        with tempfile.TemporaryDirectory() as td:
            # Extract the test function name
            test_function_name = "test_function"
            test_lines = self._extract_code(tests).split('\n')
            for line in test_lines:
                if line.strip().startswith('def test_'):
                    test_function_name = line.strip().split('def ')[1].split('(')[0]
                    break
            
            # Create a combined test file that includes both the function and tests
            combined_test_content = f"""
{code}

def _run_individual_tests():
    #Run each assertion individually to count passed/failed.
    passed = 0
    failed = 0
    
    # Try to run the test function and catch individual failures
    try:
        {test_function_name}()
        # If we get here, all tests passed
        passed = 10  # Assume 10 tests if all pass
        failed = 0
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {{e}}")
        # Count this as one failure
        passed = 9
        failed = 1
    except Exception as e:
        print(f"Error running tests: {{e}}")
        failed = 10
        passed = 0
    
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
        {test_function_name}()
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
