import torch
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM
from trl import ModelConfig, get_quantization_config, get_kbit_device_map

from think2sql.configs import SFTConfig, GRPOConfig
from think2sql.logger import get_logger

logger = get_logger(__name__)


def get_tokenizer(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        local_files_only=training_args.local_files_only,
        
    )
    logger.info(f"Tokenizer loaded for: {model_args.model_name_or_path}")

    if training_args.chat_template is not None:
        logger.warning('Override default ChatTemplate used for training')
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_model(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig):
    """Get the model"""
    torch_dtype = (
        model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    )

    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    logger.info(
        f"Model loaded: {model_args.model_name_or_path}; dtype: {torch_dtype}; quantization: {quantization_config}"
    )
    return model


