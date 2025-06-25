# llm_judge/llm_judge/text_pipeline.py

from openai import OpenAI
from .evaluator import Evaluator
from typing import Dict, Any, List

client = OpenAI()

# Cost per 1K tokens (as of 2024)
COST_PER_1K_TOKENS = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004}
}

class TextPipeline(Evaluator):
    def __init__(self, models: List[str], judge_model: str = "gpt-4"):
        self.models = models
        self.judge_model = judge_model
        self.total_cost = 0.0
        self.cost_breakdown = {
            "model_calls": {},
            "judge_calls": {},
            "total": 0.0
        }

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost for a model call."""
        if model not in COST_PER_1K_TOKENS:
            # Default to gpt-3.5-turbo pricing for unknown models
            model = "gpt-3.5-turbo"
        
        costs = COST_PER_1K_TOKENS[model]
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        return input_cost + output_cost

    def _track_cost(self, model: str, input_tokens: int, output_tokens: int, call_type: str = "model"):
        """Track the cost for a model call."""
        cost = self._calculate_cost(model, input_tokens, output_tokens)
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

    def _display_cost_summary(self):
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

    def evaluate(self, prompt: str, responses: Dict[str, str] = None) -> Dict[str, Any]:
        # Step 1: Generate responses if not given
        generated = responses or self._invoke_models(prompt)

        print("\n📦 Generated responses:")
        for name, text in generated.items():
            print(f"\n{name}:\n{text}\n{'-'*40}")

        # Step 2: Ask Judge LLM to score the answers
        judge_prompt = self._build_judge_prompt(prompt, generated)

        judge_response = client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0
        )

        # Track cost for judge model
        self._track_cost(
            self.judge_model,
            judge_response.usage.prompt_tokens,
            judge_response.usage.completion_tokens,
            "judge"
        )

        try:
            judgment = judge_response.choices[0].message.content
            print("\n🧠 Judge Output:\n", judgment)

            # Step 3: Extract JSON
            parsed = self._extract_json(judgment)
            
            # Step 4: Display detailed reasoning
            self._display_reasoning(parsed, generated)
            
            # Step 5: Display cost summary
            self._display_cost_summary()
            
            return {
                "prompt": prompt,
                "responses": generated,
                "scores": parsed,
                "winner": parsed.get("winner"),
                "cost_breakdown": self.cost_breakdown
            }
        except Exception as e:
            return {"error": "Failed to evaluate text", "details": str(e)}

    def _invoke_models(self, prompt: str) -> Dict[str, str]:
        responses = {}
        for model in self.models:
            print(f"🎯 Invoking {model}...")
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Answer clearly and concisely."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
            )
            
            # Track cost
            self._track_cost(
                model,
                resp.usage.prompt_tokens,
                resp.usage.completion_tokens,
                "model"
            )
            
            responses[model] = resp.choices[0].message.content.strip()
        return responses

    def _build_judge_prompt(self, prompt: str, responses: Dict[str, str]) -> str:
        instruction = f"""
            You are an expert evaluator. Given the following prompt:

            PROMPT:
            {prompt}

            And the following model responses:

            """
        for name, response in responses.items():
            instruction += f"{name}:\n{response}\n\n"

        instruction += """
            Score each response on a scale of 1–5 for:
            - Correctness: How accurate and factually correct is the response?
            - Relevance: How well does the response address the prompt?

            Then select the best response and provide detailed reasoning.

            Return the output in **this exact JSON format**:

            {
            "gpt-4": { 
                "correctness": 5, 
                "relevance": 5,
                "reasoning": "Detailed explanation of strengths and weaknesses"
            },
            "gpt-3.5-turbo": { 
                "correctness": 3, 
                "relevance": 4,
                "reasoning": "Detailed explanation of strengths and weaknesses"
            },
            "winner": "gpt-4",
            "winner_reasoning": "Detailed explanation of why this response is the best",
            "loser_reasoning": "Detailed explanation of why the losing response fell short"
            }

            For the reasoning fields:
            - Be specific about what makes each response good or bad
            - Point out specific strengths and weaknesses
            - Explain how well each response addresses the prompt
            - Highlight any factual errors, logical gaps, or missed points

            Only return valid JSON.
        """
        return instruction.strip()

    def _extract_json(self, text: str) -> Dict[str, Any]:
        import json, re
        # Try to extract the first valid JSON block
        match = re.search(r"{.*}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("No valid JSON in judge output")

    def _display_reasoning(self, parsed: Dict[str, Any], responses: Dict[str, str]):
        """Display detailed reasoning from the judge in a readable format."""
        print("\n" + "="*60)
        print("🏆 JUDGE'S DETAILED ANALYSIS")
        print("="*60)
        
        winner = parsed.get("winner")
        winner_reasoning = parsed.get("winner_reasoning", "")
        loser_reasoning = parsed.get("loser_reasoning", "")
        
        # Display individual model analysis
        for model_name, model_data in parsed.items():
            if model_name in ["winner", "winner_reasoning", "loser_reasoning"]:
                continue
                
            print(f"\n📊 {model_name.upper()} ANALYSIS:")
            print(f"   Correctness: {model_data.get('correctness', 'N/A')}/5")
            print(f"   Relevance: {model_data.get('relevance', 'N/A')}/5")
            
            reasoning = model_data.get('reasoning', '')
            if reasoning:
                print(f"   Reasoning: {reasoning}")
        
        # Display winner/loser reasoning
        if winner and winner_reasoning:
            print(f"\n🥇 WINNER ({winner}):")
            print(f"   {winner_reasoning}")
        
        if loser_reasoning:
            # Find the loser (the model that's not the winner)
            loser = None
            for model_name in responses.keys():
                if model_name != winner:
                    loser = model_name
                    break
            
            if loser:
                print(f"\n🥈 RUNNER-UP ({loser}):")
                print(f"   {loser_reasoning}")
        
        print("\n" + "="*60)
