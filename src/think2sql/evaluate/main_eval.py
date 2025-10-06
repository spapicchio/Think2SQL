import os

from trl import TrlParser
from vllm import SamplingParams

from think2sql.configs import EvaluateArgs
from think2sql.data_processor import build_messages
from think2sql.data_processor.get_ddl import get_schema
from think2sql.evaluate.configs import EvalVLLMConfig, EvalGenerationParams
from think2sql.evaluate.data_readers import DataReader, HFDataReader, JsonDataReader
from think2sql.evaluate.evaluators import Evaluator, SqliteEvaluatorEX
from think2sql.evaluate.predictors import Predictor, VLLMPredictor
from think2sql.evaluate.saver import DataframeSaver, JSONSaver
from think2sql.evaluate.message_builders import BuildMessagesBirdDev
from think2sql.logger import get_logger
from think2sql.utils.sql import get_sql_from_generation, check_crud_sql

logger = get_logger(__name__)


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
    dataset = data_reader.read(evaluate_args.dataset_name)

    dataset = messages_builder.build(dataset)

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
        model_name=vllm_config.model_name,
        messages=[line["messages"] for line in dataset],
        sampling_params=sampling_params,
        tp=vllm_config.tensor_parallel_size,
        dp=vllm_config.data_parallel_size,
        max_model_len=vllm_config.max_model_length,
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
    df[f'SQL_{model_name}'] = [check_crud_sql(get_sql_from_generation(pred)) for pred in predictions]
    df[f'EX_{model_name}'] = results
    saver.save(
        folder='./results',
        file_name=f'eval_{vllm_config.model_name.replace("/", "_")}_{evaluate_args.dataset_name}',
        df=dataset.to_pandas().assign(model_name=predictions, results=results)
    )


if __name__ == "__main__":
    parser = TrlParser((EvalVLLMConfig, EvalGenerationParams, EvaluateArgs))
    vllm_config, generation_params, evaluate_args = parser.parse_args_and_config()
    main_eval(
        vllm_config, generation_params, evaluate_args,
        JsonDataReader(),
        BuildMessagesBirdDev(),
        VLLMPredictor(),
        SqliteEvaluatorEX(),
        JSONSaver()
    )
