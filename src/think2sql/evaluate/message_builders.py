from typing import Protocol

import pandas as pd
from datasets import Dataset

from think2sql.configs import EvaluateArgs
from think2sql.data_processor import build_messages
from think2sql.data_processor.get_ddl import get_schema
from think2sql.logger import get_logger

logger = get_logger(__name__)


class BuildMessages(Protocol):
    @staticmethod
    def build(dataset: Dataset, evaluate_args: EvaluateArgs, *args, **kwargs) -> Dataset:
        ...


class BuildMessagesBirdDev:
    @staticmethod
    def build(dataset: Dataset, evaluate_args: EvaluateArgs, *args, **kwargs) -> Dataset:
        logger.info(f"Creating column `messages` in dataset with `BuildMessagesBirdDev`")
        logger.info("Inserting evidence and schema if not present in dataset")
        dataset = dataset.map(
            lambda line: {
                'evidence': line.get("evidence", ""),
                "schema": line.get("schema", get_schema(line, 'inline', evaluate_args.relative_db_base_path)["schema"])
            },
            num_proc=64,
            load_from_cache_file=False
        )
        logger.info("Building messages HF format")
        dataset = dataset.map(
            lambda line: {
                "messages": build_messages(
                    row=line,
                    user_prompt_name=evaluate_args.user_prompt_name,
                    system_prompt_name=evaluate_args.system_prompt_name,
                    assistant_response_col_name=None,
                    prompt_folder=evaluate_args.prompt_folder,
                )["prompt"],
            },
            num_proc=64,
            load_from_cache_file=False
        )
        return dataset


class BuildMessagesOmniSQLData:
    @staticmethod
    def build(dataset: Dataset,
              evaluate_args: EvaluateArgs,
              *args, **kwargs) -> Dataset:
        path_dev_db_id_json = evaluate_args.omnisql_file_db_id_json_path
        df = pd.read_json(path_dev_db_id_json)
        df_omnisql = dataset.to_pandas()
        df_omnisql['db_id'] = df['db_id']
        df_omnisql['SQL'] = df_omnisql['output_seq']
        df_omnisql['messages'] = df_omnisql.apply(
            lambda row: [{"role": "user", "content": row['input_seq']}],
            axis=1
        )
        return Dataset.from_pandas(df_omnisql)
