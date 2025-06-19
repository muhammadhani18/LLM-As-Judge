from abc import ABC, abstractmethod
from typing import Dict, Any

class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, prompt: str, responses: Dict[str, str]) -> Dict[str, Any]:
        """
        Given a prompt and a dict of model-name→response,
        return a dict containing scores, winner, and any metadata.
        """
        ...
