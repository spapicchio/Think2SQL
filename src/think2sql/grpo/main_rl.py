from trl import ModelConfig, TrlParser

from think2sql.configs import GRPOScriptArguments, GRPOConfig
from think2sql.grpo.think2sql_trainer import Think2SQLTrainer


def main_rl(script_args: GRPOScriptArguments, training_args: GRPOConfig, model_args: ModelConfig):
    trainer = Think2SQLTrainer(script_args, training_args, model_args)
    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main_rl(script_args, training_args, model_args)
