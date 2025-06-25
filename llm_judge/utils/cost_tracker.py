"""
Cost tracking utilities for LLM API calls.
"""
from typing import Dict, Any
from ..config import COST_PER_1K_TOKENS

class CostTracker:
    """Tracks and calculates costs for LLM API calls."""
    
    def __init__(self):
        self.total_cost = 0.0
        self.cost_breakdown = {
            "model_calls": {},
            "judge_calls": {},
            "total": 0.0
        }
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost for a model call."""
        if model not in COST_PER_1K_TOKENS:
            # Default to gpt-3.5-turbo pricing for unknown models
            model = "gpt-3.5-turbo"
        
        costs = COST_PER_1K_TOKENS[model]
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        return input_cost + output_cost
    
    def track_cost(self, model: str, input_tokens: int, output_tokens: int, call_type: str = "model"):
        """Track the cost for a model call."""
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        self.total_cost += cost
        
        if call_type == "judge":
            if model not in self.cost_breakdown["judge_calls"]:
                self.cost_breakdown["judge_calls"][model] = {"calls": 0, "cost": 0.0}
            self.cost_breakdown["judge_calls"][model]["calls"] += 1
            self.cost_breakdown["judge_calls"][model]["cost"] += cost
        else:
            if model not in self.cost_breakdown["model_calls"]:
                self.cost_breakdown["model_calls"][model] = {"calls": 0, "cost": 0.0}
            self.cost_breakdown["model_calls"][model]["calls"] += 1
            self.cost_breakdown["model_calls"][model]["cost"] += cost
        
        self.cost_breakdown["total"] = self.total_cost
        
        print(f"💰 Cost for {model} ({call_type}): ${cost:.4f} ({input_tokens} input, {output_tokens} output tokens)")
    
    def display_summary(self):
        """Display a summary of all costs incurred during the evaluation."""
        print("\n" + "="*60)
        print("💰 COST SUMMARY")
        print("="*60)
        
        print(f"\n📊 MODEL CALLS:")
        for model, data in self.cost_breakdown["model_calls"].items():
            print(f"   {model}: {data['calls']} calls, ${data['cost']:.4f}")
        
        print(f"\n🧠 JUDGE CALLS:")
        for model, data in self.cost_breakdown["judge_calls"].items():
            print(f"   {model}: {data['calls']} calls, ${data['cost']:.4f}")
        
        print(f"\n💵 TOTAL COST: ${self.total_cost:.4f}")
        print("="*60)
    
    def get_breakdown(self) -> Dict[str, Any]:
        """Get the cost breakdown dictionary."""
        return self.cost_breakdown.copy()
    
    def reset(self):
        """Reset the cost tracker."""
        self.total_cost = 0.0
        self.cost_breakdown = {
            "model_calls": {},
            "judge_calls": {},
            "total": 0.0
        } 