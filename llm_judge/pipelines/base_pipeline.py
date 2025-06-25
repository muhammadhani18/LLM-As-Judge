"""
Base pipeline class with common functionality for all evaluation pipelines.
"""
import re
from typing import Dict, Any, List
from openai import OpenAI
from ..config import OPENAI_API_KEY, CLI_SETTINGS
from ..utils.cost_tracker import CostTracker

client = OpenAI()

class BasePipeline:
    """Base class for all evaluation pipelines."""
    
    def __init__(self, models: List[str], judge_model: str = "gpt-4"):
        self.models = models
        self.judge_model = judge_model
        self.cost_tracker = CostTracker()
    
    def _invoke_models(self, prompt: str, system_message: str = "Answer clearly and concisely.") -> Dict[str, str]:
        """Invoke multiple models with the same prompt."""
        responses = {}
        for model in self.models:
            print(f"🎯 Invoking {model}...")
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=CLI_SETTINGS["default_temperature"],
            )
            
            # Track cost
            self.cost_tracker.track_cost(
                model,
                resp.usage.prompt_tokens,
                resp.usage.completion_tokens,
                "model"
            )
            
            responses[model] = resp.choices[0].message.content.strip()
        return responses
    
    def _extract_code(self, text: str) -> str:
        """Extract code from markdown or plain text."""
        # Removes triple backticks and any leading explanations
        # Handles ```python ... ``` or ``` ...
        code_blocks = re.findall(r"```(?:python)?(.*?)```", text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        return text.strip()
    
    def display_responses(self, responses: Dict[str, str]):
        """Display generated responses from all models."""
        print("\n📦 Generated responses:")
        for name, response in responses.items():
            print(f"\n{name}:\n{response}\n{'-'*40}")
    
    def get_cost_breakdown(self) -> Dict[str, Any]:
        """Get the cost breakdown."""
        return self.cost_tracker.get_breakdown()
    
    def display_cost_summary(self):
        """Display cost summary."""
        self.cost_tracker.display_summary()
    
    def reset_cost_tracker(self):
        """Reset the cost tracker."""
        self.cost_tracker.reset() 