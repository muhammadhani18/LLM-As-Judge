"""
Refactored text evaluation pipeline using modular utilities.
"""
from typing import Dict, Any, List
from openai import OpenAI
from ..evaluator import Evaluator
from ..config import CLI_SETTINGS
from ..utils.judge_analyzer import JudgeAnalyzer
from .base_pipeline import BasePipeline

client = OpenAI()

class TextPipeline(Evaluator, BasePipeline):
    """Text evaluation pipeline with modular architecture."""
    
    def __init__(self, models: List[str], judge_model: str = "gpt-4"):
        BasePipeline.__init__(self, models, judge_model)
        self.judge_analyzer = JudgeAnalyzer(judge_model)
    
    def evaluate(self, prompt: str, responses: Dict[str, str] = None) -> Dict[str, Any]:
        """Evaluate text generation from multiple models."""
        # Step 1: Generate responses if not given
        generated = responses or self._invoke_models(prompt)

        # Step 2: Display responses
        self.display_responses(generated)

        # Step 3: Ask Judge LLM to score the answers
        judge_prompt = self.judge_analyzer.analyze_text_quality(prompt, generated)

        judge_response = client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=CLI_SETTINGS["judge_temperature"]
        )

        # Track cost for judge model
        self.cost_tracker.track_cost(
            self.judge_model,
            judge_response.usage.prompt_tokens,
            judge_response.usage.completion_tokens,
            "judge"
        )

        try:
            judgment = judge_response.choices[0].message.content
            print("\n🧠 Judge Output:\n", judgment)

            # Step 4: Extract JSON
            parsed = self.judge_analyzer._extract_json(judgment)
            
            # Step 5: Display detailed reasoning
            self.judge_analyzer.display_text_reasoning(parsed, generated)
            
            # Step 6: Display cost summary
            self.display_cost_summary()
            
            return {
                "prompt": prompt,
                "responses": generated,
                "scores": parsed,
                "winner": parsed.get("winner"),
                "cost_breakdown": self.get_cost_breakdown()
            }
        except Exception as e:
            return {"error": "Failed to evaluate text", "details": str(e)} 