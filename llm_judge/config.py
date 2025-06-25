"""
Configuration settings for LLM Judge evaluation framework.
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Cost per 1K tokens (as of 2024)
COST_PER_1K_TOKENS = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004}
}

# Default Models
DEFAULT_JUDGE_MODEL = "gpt-4"
DEFAULT_MODELS = ["gpt-4", "gpt-3.5-turbo"]

# Evaluation Weights
CODE_EVALUATION_WEIGHTS = {
    "functionality": 0.7,
    "style": 0.3
}

# Security Penalty
SECURITY_PENALTY_PER_ISSUE = 0.1

# Test Generation Settings
MAX_TEST_TOKENS = 500
MAX_TEST_CASES = 10

# File Paths
EVALUATIONS_DIR = "evaluations"
BATCH_PROMPTS_DIR = "batch_prompts"

# CLI Settings
CLI_SETTINGS = {
    "default_temperature": 0.7,
    "judge_temperature": 0,
    "timeout_seconds": 30
} 