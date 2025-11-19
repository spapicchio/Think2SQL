import statistics
from dataclasses import dataclass, asdict
from pathlib import Path

from NL2SQLEvaluator.db_executor_nodes.db_executor_protocol import extract_sql_or_same
from NL2SQLEvaluator.evaluator_nodes import BirdEXEvaluator
from trl import TrlParser
from vllm import SamplingParams

from think2sql.configs import EvaluateArgs
from think2sql.data_processor import build_messages
from think2sql.evaluate.configs import EvalVLLMConfig, EvalGenerationParams
from think2sql.evaluate.predictors import Predictor, VLLMPredictor, LiteLLMPredictor
from think2sql.evaluate.saver import DataframeSaver, JSONSaver
from think2sql.grpo.rewards.rewards_answer_content import nl2sql_reward
from think2sql.logger import get_logger
from think2sql.utils.data import get_dataset

logger = get_logger(__name__)


@dataclass
class SummaryResults:
    number_of_completions: int
    model_name: str
    dataset_name: str
    ex: float
    std: float


def main_eval(
        vllm_config: EvalVLLMConfig,
        generation_params: EvalGenerationParams,
        evaluate_args: EvaluateArgs,
        predictor: Predictor,
        saver: DataframeSaver,
):
    logger.info(asdict(vllm_config))
    logger.info(asdict(generation_params))
    logger.info(asdict(evaluate_args))

    # Read dataset
    dataset = get_dataset(evaluate_args)
    dataset = dataset['train'] if 'test' not in dataset else dataset['test']
    initial_len = len(dataset)
    dataset = dataset.filter(lambda x: x['sql_execution_time'] != -1)
    logger.info(f"Filtered {initial_len - len(dataset)} samples that have sql_execution_time != -1 (Wrong target SQL)")

    # dataset = dataset.select(range(10))

    dataset = dataset.map(
        build_messages,
        fn_kwargs={
            "user_prompt_name": evaluate_args.user_prompt_name,
            "system_prompt_name": evaluate_args.system_prompt_name,
            "assistant_response_col_name": evaluate_args.assistant_response_col_name,
            "prompt_folder": evaluate_args.prompt_folder,
        },
        num_proc=16,
    )

    # store some examples to check the data processing
    for i in range(3):
        logger.info("***** Example *****")
        logger.info(f"Example {i} of the processed dataset")
        logger.info(f"{dataset[i]['prompt']}")

    # create Sampling Params for generation
    sampling_params = SamplingParams(
        n=generation_params.number_of_completions,
        repetition_penalty=generation_params.repetition_penalty,
        temperature=generation_params.temperature,
        top_p=generation_params.top_p,
        top_k=generation_params.top_k,
        max_tokens=generation_params.max_new_tokens,
    )
    logger.info(f"Sampling Params: {sampling_params}")

    model_name = vllm_config.model_name.split('/')[-1].replace('.json', '').replace('.', '_')
    dataset_name = "_".join(evaluate_args.dataset_name.replace('.json', '').split('/'))
    strategy = 'greedy' if generation_params.number_of_completions == 1 else 'majority_voting'

    num_experiments = evaluate_args.num_of_experiments if generation_params.temperature > 0 else 1
    logger.info(f'Running {num_experiments} experiments to calculate standard deviation.')

    messages = [line["prompt"] for line in dataset] * num_experiments
    target_queries = [line[evaluate_args.target_sql_col_name] for line in dataset] * num_experiments
    db_ids = [line["db_id"] for line in dataset] * num_experiments

    logger.info(f"Total number of messages: {len(messages)}: {len(dataset)} * {num_experiments}")
    predictions_str = predictor.infer(
        messages=messages,
        sampling_params=sampling_params,
        max_model_len=vllm_config.max_model_length,
        tp=vllm_config.tensor_parallel_size,
        dp=vllm_config.data_parallel_size,
        **asdict(vllm_config),
        **asdict(evaluate_args),
    )

    logger.info(f'Prediction Example: {predictions_str[0]}')

    if generation_params.number_of_completions == 1:
        predictions = [[{'content': pred}] for pred in predictions_str]
    else:
        predictions = [[{'content': p} for p in pred] for pred in predictions_str]

    results = nl2sql_reward(
        completions=predictions,
        target_sql=target_queries,
        db_id=db_ids,
        evaluator=BirdEXEvaluator(),
        relative_db_base_path=evaluate_args.relative_db_base_path,
        sql_execution_time=[500] * len(predictions)
    )

    df = dataset.to_pandas()

    ex_n = []
    for i in range(num_experiments):
        start = i * len(dataset)
        end = start + len(dataset)
        n_pred = predictions_str[start:end]
        n_results = results[start:end]
        df[f'{model_name}_{i}'] = n_pred

        sql_prediction = [
            extract_sql_or_same(pred) if generation_params.number_of_completions == 1
            else [extract_sql_or_same(p) for p in pred]
            for pred in n_pred
        ]
        ex_n.append(statistics.mean(n_results))

        df[f'SQL_{model_name}_{i}'] = sql_prediction
        df[f'EX_{model_name}_{i}'] = n_results

    mean_ex = statistics.mean(ex_n) * 100
    std_ex = (statistics.stdev(ex_n) * 100) if len(ex_n) > 1 else 0.0
    summary_results = SummaryResults(
        number_of_completions=generation_params.number_of_completions,
        model_name=model_name,
        dataset_name=dataset_name,
        ex=round(mean_ex, 3),
        std=round(std_ex, 3),
    )
    logger.warning(summary_results)
    saver.save(
        folder=Path(evaluate_args.save_folder_path) / strategy,
        df=df,
        configs=(vllm_config, generation_params, sampling_params, evaluate_args, summary_results),
    )


if __name__ == "__main__":
    parser = TrlParser((EvalVLLMConfig, EvalGenerationParams, EvaluateArgs))
    vllm_config, generation_params, evaluate_args = parser.parse_args_and_config()
    main_eval(
        vllm_config, generation_params, evaluate_args,
        VLLMPredictor() if vllm_config.litellm_provider is None else LiteLLMPredictor(),
        JSONSaver()
    )
