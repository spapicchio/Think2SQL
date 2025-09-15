import sys
from types import ModuleType

from lighteval.tasks.lighteval_task import LightevalTaskConfig

from think2sql.configs import EvaluateArgs
from think2sql.evaluate.metrics import extend_enum_metrics
from think2sql.evaluate.prompt import get_prompt

TASKS_TABLE = []


def create_and_add_task_config(eval_args: EvaluateArgs):
    execution_accuracy = extend_enum_metrics(evaluate_args=eval_args)
    if 'bird' in eval_args.dataset_name:
        task = LightevalTaskConfig(
            name="bird_dev",
            prompt_function=get_prompt(evaluate_args=eval_args),
            hf_repo=eval_args.dataset_name,
            hf_subset="default",
            suite=["community"],
            hf_avail_splits=["train", "dev", "minidev"],
            evaluation_splits=["dev"],  # List of dataset splits to process (e.g. ["train", "dev"])
            few_shots_split=None,
            few_shots_select=None,
            metrics=[execution_accuracy],
            num_samples=None
        )
        TASKS_TABLE.append(task)
        return task
    else:
        raise KeyError(f"Dataset {eval_args.dataset_name} not supported.")


def build_tasks_module(eval_args: EvaluateArgs) -> ModuleType:
    """Populate TASKS_TABLE and return the module object itself."""
    create_and_add_task_config(eval_args)
    return sys.modules[__name__]


if __name__ == "__main__":
    eval_args = EvaluateArgs()
    create_and_add_task_config(eval_args)
    for task in TASKS_TABLE:
        print(task)
