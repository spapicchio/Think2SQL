from typing import Protocol

from datasets import Dataset

logger = get_logger(__name__)


class BuildMessages(Protocol):
    @staticmethod
    def build(dataset: Dataset) -> Dataset:
        ...


class BuildMessagesBirdDev:
    @staticmethod
    def build(dataset: Dataset) -> Dataset:
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
