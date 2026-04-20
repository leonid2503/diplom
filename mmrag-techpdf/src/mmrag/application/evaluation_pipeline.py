"""
Evaluation pipeline: run the query pipeline over a golden dataset and
report precision, recall, F1, and faithfulness metrics.

Golden dataset format (JSONL, one record per line)
---------------------------------------------------
{
  "question": "What activation function does the paper use?",
  "expected_answer": "ReLU"
}

Metrics computed
----------------
* **Token F1**       – standard SQuAD-style F1 between predicted and
                       expected answer tokens.
* **Exact Match**    – 1 if the expected answer appears verbatim in the
                       predicted answer (case-insensitive), else 0.
* **Answer Length**  – number of tokens in the predicted answer.

For a more sophisticated evaluation (faithfulness, context precision /
recall via an LLM judge) wire in an LLM-based Evaluator adapter.
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

from .query_pipeline import QueryPipeline
from ..domain.query import Answer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token-level metrics (SQuAD style)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text


def _tokenize(text: str) -> List[str]:
    return _normalize(text).split()


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(ground_truth)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(_normalize(ground_truth) in _normalize(prediction))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class EvaluationPipeline:
    """
    Runs the QueryPipeline over each record in a golden JSONL file and
    aggregates metrics.
    """

    def __init__(self, query_pipeline: QueryPipeline):
        self.qp = query_pipeline

    def run(self, golden_path: str, output_path: str | None = None) -> Dict[str, float]:
        """
        Evaluate against *golden_path* (JSONL).

        Parameters
        ----------
        golden_path  : Path to the golden dataset JSONL file.
        output_path  : Optional path to write per-question results JSONL.

        Returns
        -------
        Dict with aggregated metric averages.
        """
        records = self._load_golden(golden_path)
        if not records:
            logger.warning("No records found in %s", golden_path)
            return {}

        results: List[dict] = []
        for record in records:
            question = record["question"]
            expected = record.get("expected_answer", record.get("answer", ""))

            logger.info("Evaluating: %s", question[:80])
            try:
                answer: Answer = self.qp.run(question)
                prediction = answer.text
            except Exception as exc:
                logger.error("Query failed: %s", exc)
                prediction = ""

            f1 = token_f1(prediction, expected)
            em = exact_match(prediction, expected)

            results.append(
                {
                    "question": question,
                    "expected": expected,
                    "predicted": prediction,
                    "f1": f1,
                    "exact_match": em,
                    "answer_length": len(prediction.split()),
                    "artifact_ids": answer.artifacts if prediction else [],
                }
            )
            logger.info("  F1=%.3f  EM=%.3f", f1, em)

        # Aggregate
        avg_f1 = sum(r["f1"] for r in results) / len(results)
        avg_em = sum(r["exact_match"] for r in results) / len(results)
        avg_len = sum(r["answer_length"] for r in results) / len(results)

        aggregated = {
            "num_questions": len(results),
            "avg_f1": avg_f1,
            "avg_exact_match": avg_em,
            "avg_answer_length": avg_len,
        }

        logger.info(
            "Evaluation complete — F1: %.3f  EM: %.3f  (n=%d)",
            avg_f1,
            avg_em,
            len(results),
        )

        if output_path:
            self._save_results(results, aggregated, output_path)

        return aggregated

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_golden(path: str) -> List[dict]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Golden dataset not found: {path}")
        records = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _save_results(results: List[dict], aggregated: Dict, output_path: str) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary_path = out.with_suffix(".summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)
        logger.info("Results saved to %s", output_path)
