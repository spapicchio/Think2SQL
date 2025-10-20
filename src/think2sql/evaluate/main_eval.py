import os
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path

from trl import TrlParser
from vllm import SamplingParams

from think2sql.configs import EvaluateArgs
from think2sql.evaluate.configs import EvalVLLMConfig, EvalGenerationParams
from think2sql.evaluate.data_readers import DataReader, JsonDataReader
from think2sql.evaluate.evaluators import Evaluator, SqliteEvaluatorEX
from think2sql.evaluate.message_builders import BuildMessagesBirdDev, BuildMessagesOmniSQLData
from think2sql.evaluate.predictors import Predictor, VLLMPredictor, LiteLLMPredictor
from think2sql.evaluate.saver import DataframeSaver, JSONSaver
from think2sql.logger import get_logger
from think2sql.utils.sql import get_sql_from_generation

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
        data_reader: DataReader,
        messages_builder: BuildMessagesBirdDev,
        predictor: Predictor,
        evaluator: Evaluator,
        saver: DataframeSaver,
):
    logger.info(asdict(vllm_config))
    logger.info(asdict(generation_params))
    logger.info(asdict(evaluate_args))

    dataset = data_reader.read(evaluate_args.dataset_name)
    dataset = messages_builder.build(dataset, evaluate_args)

    # create Sampling Params
    sampling_params = SamplingParams(
        n=generation_params.number_of_completions,
        repetition_penalty=generation_params.repetition_penalty,
        temperature=generation_params.temperature,
        top_p=generation_params.top_p,
        top_k=generation_params.top_k,
        max_tokens=generation_params.max_new_tokens,
    )
    logger.info(f"Sampling Params: {sampling_params}")
    model_name = vllm_config.model_name.replace("/", "_")
    dataset_name = "_".join(evaluate_args.dataset_name.replace('.json', '').split('/'))
    strategy = 'greedy' if generation_params.number_of_completions == 1 else 'majority_voting'

    num_experiments = evaluate_args.num_of_experiments if generation_params.temperature > 0 else 1
    logger.info(f'Running {num_experiments} experiments to calculate standard deviation.')
    messages = [line["messages"] for line in dataset] * num_experiments
    target_queries = [line[evaluate_args.target_sql_col_name] for line in dataset] * num_experiments
    db_files = [os.path.join(evaluate_args.relative_db_base_path, line["db_id"], f"{line['db_id']}.sqlite")
                for line in dataset
                ] * num_experiments
    logger.info(f"Total number of messages: {len(messages)}: {len(dataset)} * {num_experiments}")
    predictions = predictor.infer(
        messages=messages,
        sampling_params=sampling_params,
        max_model_len=vllm_config.max_model_length,
        tp=vllm_config.tensor_parallel_size,
        dp=vllm_config.data_parallel_size,
        **asdict(vllm_config)
    )

    # run evaluations
    results = evaluator.evaluate(
        target_queries=target_queries,
        llm_predictions=predictions,
        db_files=db_files,
        relative_db_base_path=evaluate_args.relative_db_base_path,
    )

    df = dataset.to_pandas()

    ex_n = []
    for i in range(num_experiments):
        start = i * len(dataset)
        end = start + len(dataset)
        n_pred = predictions[start:end]
        n_results = results[start:end]
        df[f'{model_name}_{i}'] = n_pred
        sql_prediction = [
            get_sql_from_generation(get_sql_from_generation(pred)) if generation_params.number_of_completions == 1
            else [get_sql_from_generation(get_sql_from_generation(p)) for p in pred]
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
        folder=Path('./results') / dataset_name / model_name / strategy,
        df=df,
        configs=(vllm_config, generation_params, sampling_params, evaluate_args, summary_results),
    )


if __name__ == "__main__":
    parser = TrlParser((EvalVLLMConfig, EvalGenerationParams, EvaluateArgs))
    vllm_config, generation_params, evaluate_args = parser.parse_args_and_config()
    main_eval(
        vllm_config, generation_params, evaluate_args,
        JsonDataReader(),
        BuildMessagesOmniSQLData() if evaluate_args.omnisql_file_db_id_json_path is not None else BuildMessagesBirdDev(),
        VLLMPredictor() if vllm_config.litellm_provider is None else LiteLLMPredictor(),
        SqliteEvaluatorEX(),
        JSONSaver()
    )
