from statistics import mean

from aenum import extend_enum
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_corpus import CorpusLevelComputation
from lighteval.metrics.sample_preparator import GenerativePreparator
from lighteval.metrics.utils.metric_utils import (
    CorpusLevelMetric,
    SamplingMethod,
)
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc

from think2sql.configs import EvaluateArgs
from think2sql.grpo.rewards import get_reward_funcs


class EXPreparator(GenerativePreparator):
    @staticmethod
    def prepare(doc: Doc, model_response: ModelResponse, **kwargs):
        return {
            'golds': doc.get_golds()[0],
            'preds': model_response.final_text if len(model_response.final_text) > 1 else model_response.final_text,
            'formatted_doc': doc,
        }


class CorpusLevelEX(CorpusLevelComputation):
    def __init__(self, evaluate_args: EvaluateArgs):
        setattr(evaluate_args, "reward_funcs", ["EX"])
        self.evaluate_args = evaluate_args

    def compute_corpus(self, items: list[dict]) -> float:
        ex_fn = get_reward_funcs(script_args=self.evaluate_args)[0]

        golds = [item['golds'] for item in items]
        preds = [item['preds'] for item in items]
        db_ids = [item['formatted_doc'].specific["db_id"] for item in items]
        rewards = ex_fn(completions=preds,
                        target_sql=golds,
                        db_id=db_ids)
        return mean(rewards)


def extend_enum_metrics(evaluate_args: EvaluateArgs):
    execution_accuracy = CorpusLevelMetric(
        metric_name='execution_accuracy@3',
        higher_is_better=True,
        sample_level_fn=EXPreparator(),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=CorpusLevelEX(evaluate_args),
    )

    # Adds the metric to the metric list!
    extend_enum(Metrics, 'execution_accuracy', execution_accuracy)
    return Metrics.execution_accuracy


if __name__ == "__main__":
    print("Imported metric")
