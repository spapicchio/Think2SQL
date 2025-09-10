from trl import ModelConfig, TrlParser

from think2sql.configs import GRPOScriptArguments, GRPOConfig
from think2sql.grpo.rl_trainer import GRPOCustomTrainer


def main_rl(script_args: GRPOScriptArguments, training_args: GRPOConfig, model_args: ModelConfig):
    trainer = GRPOCustomTrainer(script_args, training_args, model_args)
    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main_rl(script_args, training_args, model_args)
