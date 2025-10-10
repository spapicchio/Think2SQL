import os
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

    # run predictions
    predictions = predictor.infer(
        messages=[line["messages"] for line in dataset],
        sampling_params=sampling_params,
        **asdict(vllm_config)
    )

    # run evaluations
    results = evaluator.evaluate(
        target_queries=[line[evaluate_args.target_sql_col_name] for line in dataset],
        llm_predictions=predictions,
        db_files=[
            os.path.join(evaluate_args.relative_db_base_path, line["db_id"], f"{line['db_id']}.sqlite")
            for line in dataset
        ],
        relative_db_base_path=evaluate_args.relative_db_base_path,
    )

    logger.info(f"EX: {sum(results) / len(results): .4f}")
    df = dataset.to_pandas()
    model_name = vllm_config.model_name.replace("/", "_")
    df[model_name] = predictions

    sql_prediction = [
        get_sql_from_generation(get_sql_from_generation(pred)) if generation_params.number_of_completions == 1
        else [get_sql_from_generation(get_sql_from_generation(p)) for p in pred]
        for pred in predictions
    ]

    df[f'SQL_{model_name}'] = sql_prediction
    df[f'EX_{model_name}'] = results
    dataset_name = "_".join(evaluate_args.dataset_name.replace('.json', '').split('/'))
    strategy = 'greedy' if generation_params.number_of_completions == 1 else 'majority_voting'

    summary_results = SummaryResults(
        number_of_completions=generation_params.number_of_completions,
        model_name=model_name,
        dataset_name=dataset_name,
        ex=round(sum(results) / len(results), 4)
    )
    logger.warning(summary_results)
    saver.save(
        folder=Path('./results') / dataset_name / model_name / strategy,
        df=dataset.to_pandas().assign(model_name=predictions, results=results),
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
