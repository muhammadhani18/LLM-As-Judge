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
            - Correctness
            - Relevance

            Then select the best response and return the output in **this exact JSON format**:

            {
            "gpt-4": { "correctness": 5, "relevance": 5 },
            "gpt-3.5-turbo": { "correctness": 3, "relevance": 4 },
            "winner": "gpt-4"
            }
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
