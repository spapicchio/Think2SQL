import os

import datasets
import pandas as pd
import transformers
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig, SFTTrainer, get_peft_config

from think2sql.configs import SFTConfig, SFTScriptArguments
from think2sql.data_processor import build_messages
from think2sql.logger import get_logger
from think2sql.utils.callbacks import get_callbacks
from think2sql.utils.data import get_dataset
from think2sql.utils.hf_model import get_tokenizer, get_model
from think2sql.utils.wandb import init_wandb_training


class SFTCustomTrainer:
    def __init__(
            self,
            script_args: SFTScriptArguments,
            training_args: SFTConfig,
            model_args: ModelConfig,
    ):
        if script_args.assistant_response_col_name is None:
            raise ValueError('To run SFT need to specify the assistant response column name.')

        ###############
        # Set seed for reproducibility
        ###############
        transformers.set_seed(training_args.seed)
        ###############
        # Logger
        ###############
        logger = self._setup_logger(training_args.log_level)
        # Log on each process a small summary
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Model parameters {model_args}")
        logger.info(f"Script parameters {script_args}")
        logger.info(f"Training parameters {training_args}")
        self.logger = logger
        ###############
        # Check for last checkpoint
        ###############
        self.last_checkpoint = None
        if training_args.output_dir and os.path.isdir(training_args.output_dir):
            self.last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if self.last_checkpoint is not None and training_args.resume_from_checkpoint:
            if training_args.overwrite_output_dir:
                logger.warning(
                    f"Checkpoint detected, starting training from {training_args.output_dir}"
                )
            logger.info(
                f"Checkpoint detected, resuming training at {self.last_checkpoint=}."
            )
        elif (
                self.last_checkpoint is not None
                and training_args.resume_from_checkpoint is None
        ):
            raise ValueError(
                f"Checkpoint detected, please pass --resume_from_checkpoint {self.last_checkpoint} "
                "to continue training. If you want to start from scratch, please delete the directory "
                f"{training_args.output_dir} and pass --overwrite_output_dir to train a new model."
            )
        elif training_args.resume_from_checkpoint is not None and self.last_checkpoint is None:
            logger.info(
                f"No checkpoint found in {training_args.output_dir}, starting training from scratch."
            )
            training_args.resume_from_checkpoint = False

        ###############
        # Set wandb if present
        ###############
        report_to = training_args.report_to
        if report_to is not None and (
                (isinstance(report_to, str) and "wandb" in report_to)
                or (isinstance(report_to, list) and "wandb" in report_to)
        ):
            init_wandb_training(training_args)

        self.script_args = script_args
        self.training_args = training_args
        self.model_args = model_args

    def train(self):
        filter_fn = lambda x: x['ex_gemini-3-flash-preview_0'] == 1
        dataset = get_dataset(self.script_args, filter_fn)


        wrong_queries = pd.read_json('train_wrong_queries.json').SQL.values
        initial_len = len(dataset)
        dataset = dataset.filter(
            lambda x: x['SQL'] not in wrong_queries
        )
        self.logger.info(f'Removed {initial_len - len(dataset)} wrong queries from the training set')

        tokenizer = get_tokenizer(self.model_args, self.training_args)
        model = get_model(self.model_args, self.training_args)

        dataset = dataset.map(
            build_messages,
            fn_kwargs={
                "user_prompt_name": self.script_args.user_prompt_name,
                "system_prompt_name": self.script_args.system_prompt_name,
                "assistant_response_col_name": self.script_args.assistant_response_col_name,
                "prompt_folder": self.script_args.prompt_folder
            },
            num_proc=16,
            load_from_cache_file=False
        )
        # store some examples to check the data processing
        for i in range(3):
            self.logger.info("***** Example *****")
            self.logger.info(f"Example {i} of the TRAIN processed dataset")
            self.logger.info(f"{dataset['train'][i]['prompt']}")
            if 'completion' in dataset['train'][i]:
                self.logger.info(f"{dataset['train'][i]['completion']}")

        self.logger.info(
            f"Process dataset with user/system/assistant messages "
            f"{self.script_args.user_prompt_name}/{self.script_args.system_prompt_name}/{self.script_args.assistant_response_col_name}"
        )

        if "messages" in dataset.column_names:
            dataset = dataset.remove_columns("messages")

        #############################
        # Initialize the SFT trainer
        #############################
        trainer = SFTTrainer(
            model=model,
            args=self.training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"]
            if self.training_args.eval_strategy != "no"
            else None,
            processing_class=tokenizer,
            peft_config=get_peft_config(self.model_args),
            callbacks=get_callbacks(self.training_args, self.model_args),
        )

        ###############
        # Training loop
        ###############
        self.logger.info("*** Train ***")

        checkpoint = None
        if self.training_args.resume_from_checkpoint not in [None, "", 'false', 'False']:
            checkpoint = self.training_args.resume_from_checkpoint
            self.logger.info(
                f"Resuming training from checkpoint specified in config `{self.training_args.resume_from_checkpoint}`"
            )

        elif self.last_checkpoint is not None:
            self.logger.info(
                f"Resuming training from checkpoint found in `output_dir`: `{self.last_checkpoint}`"
            )
            checkpoint = self.last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        ##################################
        # Save model and create model card
        ##################################
        self.logger.info("*** Save model ***")
        # Align the model's generation config with the tokenizer's eos token
        # to avoid unbounded generation in the transformers `pipeline()` function
        trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
        trainer.save_model(self.training_args.output_dir)
        self.logger.info(f"Model saved to {self.training_args.output_dir}")

        # Save everything else on main process
        kwargs = {
            "dataset_name": self.script_args.dataset_name,
        }
        if trainer.accelerator.is_main_process:
            trainer.create_model_card(**kwargs)
            # Restore k,v cache for fast inference
            trainer.model.config.use_cache = True
            trainer.model.config.save_pretrained(self.training_args.output_dir)

    def _setup_logger(self, log_level="INFO"):
        log_level = log_level.upper()
        logger = get_logger(self.__class__.__name__, log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        return logger
