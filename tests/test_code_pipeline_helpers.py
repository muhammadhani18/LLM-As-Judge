import pytest
from llm_judge.code_pipeline import CodePipeline

# Initialize a pipeline (models list unused in these tests)
pipeline = CodePipeline(models=[])

def test_run_linter_valid_code():
    # A simple, secure Python snippet
    code = "def add(a, b):\n    return a + b\n"
    result = pipeline._run_linter(code)
    assert result["syntax_ok"] is True
    # No security issues expected
    assert result["security_issues"] == 0
    # We should at least have a bandit_output key
    assert "bandit_output" in result

def test_run_linter_syntax_error():
    # Code with a syntax error
    code = "def oops(:\n    pass\n"
    result = pipeline._run_linter(code)
    assert result["syntax_ok"] is False
    # security_issues should still be an int (likely 0)
    assert isinstance(result["security_issues"], int)

def test_run_tests_all_pass(tmp_path, monkeypatch):
    # A trivial function and matching test that should pass
    code = """
def add(a, b):
    return a + b
"""
    tests = """
import pytest
from snippet import add

def test_add_positive():
    assert add(1, 2) == 3
"""
    passed, failed = pipeline._run_tests(code, tests)
    assert passed == 1
    assert failed == 0

def test_run_tests_some_fail(tmp_path):
    # Function with bug and a test that exposes it
    code = """
def subtract(a, b):
    return a + b  # wrong!
"""
    tests = """
import pytest
from snippet import subtract

def test_subtract():
    assert subtract(5, 2) == 3
"""
    passed, failed = pipeline._run_tests(code, tests)
    assert failed >= 1

if __name__ == "__main__":
    pytest.main()
