"""
Refactored code evaluation pipeline using modular utilities.
"""
from typing import Dict, Any, List, Tuple
from openai import OpenAI
from ..evaluator import Evaluator
from ..config import CODE_EVALUATION_WEIGHTS, SECURITY_PENALTY_PER_ISSUE, CLI_SETTINGS
from ..utils.cost_tracker import CostTracker
from ..utils.code_analysis import CodeAnalyzer
from ..utils.test_runner import TestRunner
from ..utils.judge_analyzer import JudgeAnalyzer
from .base_pipeline import BasePipeline

client = OpenAI()

class CodePipeline(Evaluator, BasePipeline):
    """Code evaluation pipeline with modular architecture."""
    
    def __init__(self, models: List[str], judge_model: str = "gpt-4"):
        BasePipeline.__init__(self, models, judge_model)
        self.code_analyzer = CodeAnalyzer()
        self.test_runner = TestRunner(judge_model)
        self.judge_analyzer = JudgeAnalyzer(judge_model)
    
    def evaluate(self, prompt: str, responses: Dict[str, str] = None) -> Dict[str, Any]:
        """Evaluate code generation from multiple models."""
        # Step 1: Generate responses if not given
        generated = responses or self._invoke_models(
            prompt, 
            system_message="Only return valid Python code. No markdown or text."
        )
        
        # Step 2: Display responses
        self.display_responses(generated)
        
        # Step 3: Evaluate each model
        scores = {}
        for name, code in generated.items():
            print(f"\n🔍 Evaluating model: {name}")
            
            # Clean the code
            cleaned_code = self._extract_code(code)
            
            # Run analysis
            lint_info = self.code_analyzer.run_linter(cleaned_code)
            style_info = self.code_analyzer.run_style_analysis(cleaned_code)
            
            if not lint_info["syntax_ok"]:
                scores[name] = {
                    "status": "syntax_error",
                    "score": 0,
                    "details": lint_info,
                    "style": style_info
                }
                continue

            try:
                print("🤖 Asking Judge LLM to generate tests...")
                tests = self.test_runner.generate_tests(prompt, cleaned_code)
                
                # Track cost for test generation
                # Note: Cost tracking is handled in the test_runner
                
                print("⚙️ Running unit tests...")
                passed, failed = self.test_runner.run_tests(cleaned_code, tests)

                # Calculate scores
                total = passed + failed
                pass_rate = passed / total if total > 0 else 0
                security_penalty = lint_info["security_issues"] * SECURITY_PENALTY_PER_ISSUE

                # Combine functionality & style
                func_comp = max(pass_rate - security_penalty, 0)
                style_comp = style_info["style_score"]
                final_score = round(
                    CODE_EVALUATION_WEIGHTS["functionality"] * func_comp + 
                    CODE_EVALUATION_WEIGHTS["style"] * style_comp, 
                    3
                )

                scores[name] = {
                    "status": "evaluated",
                    "score": final_score,
                    "tests": {"passed": passed, "failed": failed},
                    "lint": lint_info,
                    "style": style_info
                }
            except Exception as e:
                scores[name] = {
                    "status": "error",
                    "score": 0,
                    "error": str(e)
                }

        # Step 4: Get detailed reasoning from judge
        judge_analysis = self.judge_analyzer.analyze_code_quality(prompt, generated, scores)
        
        # Step 5: Display detailed reasoning
        self.judge_analyzer.display_code_reasoning(scores, judge_analysis)

        # Step 6: Display cost summary
        self.display_cost_summary()

        # Step 7: Pick winner
        best = max(
            scores.items(), 
            key=lambda kv: kv[1]["score"] if kv[1]["status"] == "evaluated" else -1, 
            default=(None, {})
        )
        winner = best[0] if best[0] else "No valid code found"

        return {
            "prompt": prompt,
            "results": scores,
            "winner": winner,
            "judge_analysis": judge_analysis,
            "cost_breakdown": self.get_cost_breakdown()
        } 