from dataclasses import dataclass, field
from typing import Optional

import trl


@dataclass
class DatasetConfig:
    """Configuration for a dataset in a mixture."""
    id: str
    config: Optional[str] = None
    split: str = "train"
    columns: Optional[list[str]] = None
    weight: Optional[float] = None


@dataclass
class DatasetMixtureConfig:
    """Configuration for a mixture of datasets."""

    datasets: list[DatasetConfig]
    seed: int = 0
    test_split_size: Optional[float] = None


@dataclass
class SFTScriptArguments(trl.ScriptArguments):
    """
    Extended version of ScriptArguments with support for dataset mixtures.

    Args:
        dataset_mixture (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Configuration for creating dataset mixtures with advanced options.
            Format:
              dataset_mixture:
                datasets:
                  - id: dataset_id1
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                  - id: dataset_id2
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                seed: 42
                test_split_size: 0.1
    """

    # Override the dataset_name to make it optional
    dataset_name: Optional[str] = field(
        default='simone-papicchio/bird',
        metadata={"help": "Dataset name. Can be omitted if using dataset_mixture."}
    )

    target_sql_col_name: str = field(
        default='SQL',
        metadata={"help": "The target SQL column name in the dataset. Default to SQL as in the BIRD dataset."},
    )

    relative_db_base_path: str = field(
        default="data/bird/train_databases",
        metadata={"help": "Relative path to the database files directory"}
    )

    dataset_test_split: Optional[str] = field(
        default='dev',
        metadata={"help": "The dataset split to use for evaluation."},
    )
    dataset_train_split: Optional[str] = field(
        default='train',
        metadata={"help": "The dataset split to use for training."},
    )

    dataset_mixture: Optional[DatasetMixtureConfig] = field(
        default=None,
        metadata={"help": "Configuration for creating dataset mixtures with advanced options like shuffling."},
    )

    assistant_response_col_name: str = field(
        default=None,
        metadata={"help": "The assistant response column name in the dataset used for SFT."},
    )

    add_sample_rows_strategy: Optional[str] = field(
        default='inline',
        metadata={
            "help": "Strategy to add sample rows to the prompt. Options: 'random', 'similarity', or None."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.dataset_mixture is None:
            raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")

        if self.dataset_mixture is not None:
            if not isinstance(self.dataset_mixture, dict) or "datasets" not in self.dataset_mixture:
                raise ValueError(
                    "dataset_mixture must be a dictionary with a 'datasets' key. "
                    "Expected format: {'datasets': [...], 'seed': int}"
                )

            datasets_list = []
            datasets_data = self.dataset_mixture.get("datasets", [])

            if isinstance(datasets_data, list):
                for dataset_config in datasets_data:
                    datasets_list.append(
                        DatasetConfig(
                            id=dataset_config.get("id"),
                            config=dataset_config.get("config"),
                            split=dataset_config.get("split", "train"),
                            columns=dataset_config.get("columns"),
                            weight=dataset_config.get("weight", 1.0),
                        )
                    )
            else:
                raise ValueError("'datasets' must be a list of dataset configurations")

            self.dataset_mixture = DatasetMixtureConfig(
                datasets=datasets_list,
                seed=self.dataset_mixture.get("seed", 0),
                test_split_size=self.dataset_mixture.get("test_split_size", None),
            )

            # Check that column names are consistent across all dataset configs
            columns_sets = [set(dataset.columns) for dataset in datasets_list if dataset.columns is not None]
            if columns_sets:
                first_columns = columns_sets[0]
                if not all(columns == first_columns for columns in columns_sets):
                    raise ValueError(
                        "Column names must be consistent across all dataset configurations in a mixture. "
                        f"Found different column sets: {[list(cols) for cols in columns_sets]}"
                    )


@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    num_completions_to_print: int = field(default=0, metadata={"help": "Number of completions to print."})
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})

    wandb_log_unique_prompts: bool = field(
        default=True,
        metadata={
            "help": "Whether to log the unique prompts to wandb. This will create a new run for each unique prompt."
        },
    )
    log_level: str = field(
        default='INFO',
        metadata={"help": "The logging level to use. Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'."}
    )

    wandb_entity: Optional[str] = field(
        default='spapicchio-politecnico-di-torino',
        metadata={"help": "The entity to store runs under."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "The project to store runs under."},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": "The group to store runs under."},
    )

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.sync_ref_model, str):
            if self.sync_ref_model.lower() in ("true", "1", "yes"):
                self.sync_ref_model = True
            elif self.sync_ref_model.lower() in ("false", "0", "no"):
                self.sync_ref_model = False
            else:
                raise ValueError(
                    "`sync_ref_model` must be a boolean or a string representing a boolean ('true', 'false', '1', '0', 'yes', 'no')"
                )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The callbacks to run during training."},
    )

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})

    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default='spapicchio-politecnico-di-torino',
        metadata={"help": "The entity to store runs under."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "The project to store runs under."},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": "The group to store runs under."},
    )


@dataclass
class EvaluateArgs(SFTScriptArguments):
    task_name: str = field(
        default='EX',
        metadata={"help": "The name of the evaluation task."},
    )
    omnisql_file_db_id_json_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the JSON file mapping database IDs to file paths for OmniSQL."},
    )

    prompt_folder: str = field(
        default="prompts",
        metadata={"help": "The folder where the jinja prompts are stored."},
    )

    user_prompt_name: str = field(
        default="base_think_user_prompt.jinja",
        metadata={"help": "The user prompt name to use from the chat template. "
                          "The available prompts are in `prompt_folder`"}
    )
    system_prompt_name: str = field(
        default="base_think_system_prompt.jinja",
        metadata={"help": "The system prompt name to use from the chat template. "
                          "The available prompts are in `prompt_folder`"}
    )

    cache_db_connections_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path of the folder where to store the executed query. In case of multi-gpu, one database per process will be created. "
                    "It has to be manually merged into one cache DB"
        },
    )

    target_sql_cache_db_path: Optional[str] = field(
        default='.think2sql_cache/target_cached_query.sqlite',
        metadata={
            "help": "Path to the cache database with the target queries. Can be different from the db of the prediction."},
    )

    pred_sql_cache_db_path: Optional[str] = field(
        default='.think2sql_cache/pred_cached_query.sqlite',
        metadata={
            "help": "Path to the cache database with the predicted queries. Can be different from the db of the target."},
    )

    num_of_experiments: int = field(
        default=3,
        metadata={"help": "Number of experiments for calculating standard deviation."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.system_prompt_name is not None and self.system_prompt_name.lower() in ('none', 'null', ''):
            self.system_prompt_name = None
        if self.omnisql_file_db_id_json_path is not None and self.omnisql_file_db_id_json_path.lower() in ('none',
                                                                                                           'null', ''):
            self.omnisql_file_db_id_json_path = None


@dataclass
class GRPOScriptArguments(EvaluateArgs):
    """Script arguments for the GRPO training script."""

    reward_funcs: list[str] = field(
        default_factory=lambda: ["EX", "QATCH", "format", "tag_count"],
        metadata={
            "help": "List of reward functions."
                    ' Possible values: "EX", "QATCH", "format", "tag_count"'
        },
    )
