# llm_judge/llm_judge/text_pipeline.py

from openai import OpenAI
from .evaluator import Evaluator
from typing import Dict, Any, List

client = OpenAI()

class TextPipeline(Evaluator):
    def __init__(self, models: List[str], judge_model: str = "gpt-4"):
        self.models = models
        self.judge_model = judge_model

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

        try:
            judgment = judge_response.choices[0].message.content
            print("\n🧠 Judge Output:\n", judgment)

            # Step 3: Extract JSON
            parsed = self._extract_json(judgment)
            
            # Step 4: Display detailed reasoning
            self._display_reasoning(parsed, generated)
            
            return {
                "prompt": prompt,
                "responses": generated,
                "scores": parsed,
                "winner": parsed.get("winner")
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
