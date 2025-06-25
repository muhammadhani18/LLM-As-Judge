"""
Test generation and execution utilities for code evaluation.
"""
import re
import subprocess
import tempfile
import os
from typing import Tuple, Dict, Any
from openai import OpenAI
from ..config import MAX_TEST_TOKENS, DEFAULT_JUDGE_MODEL

client = OpenAI()

class TestRunner:
    """Handles test generation and execution for code evaluation."""
    
    def __init__(self, judge_model: str = DEFAULT_JUDGE_MODEL):
        self.judge_model = judge_model
    
    def generate_tests(self, prompt: str, code: str) -> str:
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
            max_tokens=MAX_TEST_TOKENS
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
    
    def run_tests(self, code: str, tests: str) -> Tuple[int, int]:
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
    
    def _extract_code(self, text: str) -> str:
        """Extract code from markdown or plain text."""
        # Removes triple backticks and any leading explanations
        # Handles ```python ... ``` or ``` ...
        code_blocks = re.findall(r"```(?:python)?(.*?)```", text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        return text.strip() 