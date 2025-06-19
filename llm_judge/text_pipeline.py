from .evaluator import Evaluator
from typing import Dict, Any, List

class TextPipeline(Evaluator):
    def __init__(self, models: List[str]):
        self.models = models

    def evaluate(self, prompt: str, responses: Dict[str, str] = None) -> Dict[str, Any]:
        # TODO: invoke models, prompt judge LLM, aggregate scores
        return {"status": "not implemented"}
