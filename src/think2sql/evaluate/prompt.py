from functools import partial
from typing import Optional, Literal

from lighteval.tasks.requests import Doc

from think2sql.configs import EvaluateArgs
from think2sql.data_processor import build_messages
from think2sql.data_processor.get_ddl import get_schema


def prompt_process_messages_bird(line,
                                 task_name,
                                 user_prompt_name,
                                 system_prompt,
                                 prompt_folder,
                                 add_sample_rows_strategy: Optional[Literal["append", "inline"]],
                                 relative_base_path: str) -> Doc:
    """Defines how to go from a dataset line to a doc object. This assumes the target is into a column named "SQL"
    Note that task_name is automatically passed by lighteval"""
    line["evidence"] = line["evidence"] if "evidence" in line else ""

    schema = get_schema(line, add_sample_rows_strategy, relative_base_path)['schema']
    line["schema"] = schema

    choices = [line["SQL"]]

    specific_to_save = {
        "db_id": line["db_id"],
        "question": line["question"],
        "evidence": line["evidence"],
        "SQL": line["SQL"],
        "difficulty": line["difficulty"] if "difficulty" in line else None,
        "schema": schema
    }

    # System prompt is passed in the VLLM config in main_eval.py
    prompt = build_messages(
        row=line,
        user_prompt_name=user_prompt_name,
        system_prompt_name=system_prompt,
        assistant_response_col_name=None,
        prompt_folder=prompt_folder,
    )['prompt'][-1]['content']

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=choices,
        gold_index=0,
        specific=specific_to_save
    )


def get_prompt(evaluate_args: EvaluateArgs):
    if evaluate_args.task_name == 'bird_dev':
        # Lighteval automatically passes one line of the dataset and the  task_name to the function
        return partial(prompt_process_messages_bird,
                       user_prompt_name=evaluate_args.user_prompt_name,
                       system_prompt=evaluate_args.system_prompt_name,
                       prompt_folder=evaluate_args.prompt_folder,
                       add_sample_rows_strategy=evaluate_args.add_sample_rows_strategy,
                       relative_base_path=evaluate_args.relative_db_base_path
                       )
    else:
        raise KeyError(f"Task not recognized: {evaluate_args.task_name}")
