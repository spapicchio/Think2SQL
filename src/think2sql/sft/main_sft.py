from trl import ModelConfig, TrlParser

from think2sql.configs import GRPOScriptArguments, SFTConfig, SFTScriptArguments
from think2sql.sft.sft_trainer import SFTCustomTrainer


def main_rl(script_args: GRPOScriptArguments, training_args: SFTConfig, model_args: ModelConfig):
    trainer = SFTCustomTrainer(script_args, training_args, model_args)
    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main_rl(script_args, training_args, model_args)
