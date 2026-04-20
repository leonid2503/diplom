from abc import ABC, abstractmethod
from typing import Dict

from ..domain.query import Answer


class Evaluator(ABC):
    """Interface for evaluating answer quality."""

    @abstractmethod
    def evaluate(self, answer: Answer, expected: str, question: str) -> Dict[str, float]:
        """
        Evaluate an answer against the expected answer.

        Returns a dict of metric_name -> score, e.g.:
          {"faithfulness": 0.9, "answer_relevance": 0.85, "f1": 0.72}
        """
        pass
