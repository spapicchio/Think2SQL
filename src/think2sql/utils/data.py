import os
from functools import lru_cache

import datasets
from NL2SQLEvaluator.db_executor import BaseSQLDBExecutor
from NL2SQLEvaluator.orchestrator_state import AvailableDialect
from datasets import DatasetDict, concatenate_datasets

from think2sql.configs import SFTScriptArguments
from think2sql.logger import get_logger

logger = get_logger(__name__)


def get_dataset(args: SFTScriptArguments) -> DatasetDict:
    """Load a dataset or a mixture of datasets based on the configuration.

    Args:
        args (SFTScriptArguments): Script arguments containing dataset configuration.

    Returns:
        DatasetDict: The loaded datasets.
    """
    if args.dataset_name and not args.dataset_mixture:
        logger.info(f"Loading dataset: {args.dataset_name}")
        return datasets.load_dataset(args.dataset_name, args.dataset_config)

    elif args.dataset_mixture:
        logger.info(f"Creating dataset mixture with {len(args.dataset_mixture.datasets)} datasets")
        seed = args.dataset_mixture.seed
        datasets_list = []

        for dataset_config in args.dataset_mixture.datasets:
            logger.info(f"Loading dataset for mixture: {dataset_config.id} (config: {dataset_config.config})")
            ds = datasets.load_dataset(
                dataset_config.id,
                dataset_config.config,
                split=dataset_config.split,
            )
            if dataset_config.columns is not None:
                ds = ds.select_columns(dataset_config.columns)
            if dataset_config.weight is not None:
                ds = ds.shuffle(seed=seed).select(range(int(len(ds) * dataset_config.weight)))
                logger.info(
                    f"Subsampled dataset '{dataset_config.id}' (config: {dataset_config.config}) with weight={dataset_config.weight} to {len(ds)} examples"
                )

            datasets_list.append(ds)

        if datasets_list:
            combined_dataset = concatenate_datasets(datasets_list)
            combined_dataset = combined_dataset.shuffle(seed=seed)
            logger.info(f"Created dataset mixture with {len(combined_dataset)} examples")

            if args.dataset_mixture.test_split_size is not None:
                combined_dataset = combined_dataset.train_test_split(
                    test_size=args.dataset_mixture.test_split_size, seed=seed
                )
                logger.info(
                    f"Split dataset into train and test sets with test size: {args.dataset_mixture.test_split_size}"
                )
                return combined_dataset
            else:
                return DatasetDict({"train": combined_dataset})
        else:
            raise ValueError("No datasets were loaded from the mixture configuration")

    else:
        raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")


@lru_cache(maxsize=100)
def utils_get_engine(relative_base_path, db_executor: AvailableDialect, db_id: str, cache=None, *args,
                     **kwargs) -> BaseSQLDBExecutor:
    try:
        if db_executor == AvailableDialect.sqlite:
            from NL2SQLEvaluator.db_executor.sqlite_executor import SqliteDBExecutor
            engine = SqliteDBExecutor.from_uri(
                relative_base_path=os.path.join(relative_base_path, db_id, f"{db_id}.sqlite"), *args, **kwargs,
            )
            engine.cache_db = cache
            return engine
    except Exception as e:
        raise ValueError(f"Error initializing database executor for {relative_base_path}: {e}")
    raise ValueError(f"Database executor not supported: {db_executor}. Supported: {list(AvailableDialect)}")
