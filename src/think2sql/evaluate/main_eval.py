from pathlib import Path

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_input import GenerationParameters
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_accelerate_available
from trl import TrlParser

from think2sql.configs import EvaluateArgs
from think2sql.evaluate.task import build_tasks_module

if is_accelerate_available():
    from datetime import timedelta
    from accelerate import Accelerator, InitProcessGroupKwargs

    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None

from dataclasses import dataclass, field, asdict


# Nested block for generation parameters (good with HfArgumentParser).
@dataclass
class GenerationParams:
    temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature; lower is more deterministic (≈0.1–0.7)."},
    )
    top_p: float = field(
        default=0.8,
        metadata={"help": "Nucleus sampling: keep smallest set of tokens whose cumulative prob ≥ top_p (0–1]."},
    )
    top_k: int = field(
        default=20,
        metadata={"help": "Top-k sampling: consider only the top_k most likely tokens (0 disables)."},
    )
    repetition_penalty: float = field(
        default=1.1,
        metadata={"help": "Penalty >1.0 discourages repeating the same tokens."},
    )
    frequency_penalty: float = field(
        default=0.0,
        metadata={"help": "Downweights tokens that appear frequently in the output so far (0–2, often 0 for code)."},
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={"help": "Penalizes tokens already present; encourages exploring new tokens (0–2, often 0 for code)."},
    )
    max_new_tokens: int = field(
        default=8000,
        metadata={"help": "Maximum number of new tokens to generate."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for generation (sampling)."},
    )


@dataclass
class InferenceConfig:
    # --- Model & runtime ---
    model_name: str = field(
        default="simone-papicchio/Think2SQL-7B",
        metadata={"help": "Hugging Face repo or local path to the model."},
    )
    dtype: str = field(
        default="bfloat16",
        metadata={"help": "Computation dtype (e.g., float16, bfloat16, float32, auto)."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of GPUs to shard weights across (tensor parallel)."},
    )
    data_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of data-parallel replicas (usually handled by launcher)."},
    )
    pipeline_parallel_size: int = field(
        default=1,
        metadata={"help": "Pipeline parallel stages (rarely needed unless model is huge)."},
    )
    gpu_memory_utilization: float = field(
        default=0.80,
        metadata={"help": "Target fraction of GPU memory to use (vLLM/Accelerate style runtimes)."},
    )
    max_model_length: int = field(
        default=30_000,
        metadata={"help": "Max total context (prompt + generated) tokens to allow."},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Allow models/repos with custom code to execute (HF `trust_remote_code`)."},
    )
    override_chat_template: bool = field(
        default=True,
        metadata={"help": "Use the repo’s chat template override when available."},
    )
    system_prompt: str = field(
        default="",
        metadata={"help": "Optional system prompt to prepend for chat-style templates."},
    )


def _get_system_prompt_from_jinja(eval_args: EvaluateArgs) -> str:
    system_prompt_jinja_file = Path(eval_args.prompt_folder) / eval_args.system_prompt_name
    if system_prompt_jinja_file.exists():
        return system_prompt_jinja_file.read_text()
    raise ValueError(f"System prompt Jinja file not found: {system_prompt_jinja_file}")


def main(vllm_config, generation_params, evaluate_args):
    evaluation_tracker = EvaluationTracker(
        output_dir="./results",
        save_details=True,
        push_to_hub=False,
        hub_results_org="...",
    )

    custom_module_task = build_tasks_module(eval_args=evaluate_args)
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        custom_tasks_directory=custom_module_task,  # Set to path if using custom tasks
        # Remove the parameter below once your configuration is tested
        # max_samples=10
    )

    vllm_config.system_prompt_name = _get_system_prompt_from_jinja(eval_args=evaluate_args)
    vllm_config = VLLMModelConfig(
        **asdict(vllm_config),
        generation_parameters=GenerationParameters(**asdict(generation_params)),
    )

    task = "community|bird_dev|0"

    pipeline = Pipeline(
        tasks=task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=vllm_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()


if __name__ == "__main__":
    parser = TrlParser((InferenceConfig, GenerationParams, EvaluateArgs))
    vllm_config, generation_params, evaluate_args = parser.parse_args_and_config()
    main(vllm_config, generation_params, evaluate_args)
