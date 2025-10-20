from dataclasses import dataclass, field


@dataclass
class EvalGenerationParams:
    number_of_completions: int = field(
        default=1,
        metadata={"help": "Number of completions to generate per prompt."},
    )
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
class EvalVLLMConfig:
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
    vllm_server_host: str = field(
        default="localhost",
        metadata={"help": "vLLM server host (if using vLLM server)."},
    )
    vllm_server_port: int = field(
        default=8000,
        metadata={"help": "vLLM server port (if using vLLM server)."},
    )
    litellm_provider: str | None = field(
        default=None,
        metadata={"help": "Optional provider for LiteLLM. If vllm online serving, use `hosted_vllm`"},
    )

    def __post_init__(self):
        if self.litellm_provider is not None and self.litellm_provider.lower() in ['none', 'null', '']:
            self.litellm_provider = None
