import time
from collections import defaultdict
from typing import Protocol, Literal
from typing import TypedDict

import requests
from dotenv import load_dotenv
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponse
from openai_harmony import ReasoningEffort
from transformers import AutoTokenizer
from vllm import SamplingParams, LLM

from think2sql.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def check_server_availability(base_url, total_timeout: float = 320, retry_interval: float = 60):
    """
    Check server availability with retries on failure. This function is only used with hosted_vllm provider.
    If the server is not up after the total timeout duration, raise a `ConnectionError`.
    """
    url = f"{base_url}/health/"
    start_time = time.time()  # Record the start time

    while True:
        try:
            response = requests.get(url)
        except requests.exceptions.RequestException as exc:
            # Check if the total timeout duration has passed
            elapsed_time = time.time() - start_time
            if elapsed_time >= total_timeout:
                raise ConnectionError(
                    f"The vLLM server can't be reached at {base_url} after {total_timeout} seconds. Make "
                    "sure the server is running by running `vllm serve`."
                ) from exc
        else:
            if response.status_code == 200:
                logger.info(f"Server is up at `{url}`!")
                return None

        # Retry logic: wait before trying again
        logger.info(f"Server is not up yet at `{url}`. Retrying in {retry_interval} seconds...")
        time.sleep(retry_interval)


# ---- Types ----
class ChatTurn(TypedDict):
    role: Literal["system", "user", "assistant", "tool", "function"]  # trim if your models need fewer
    content: str


ChatMessageHF = list[ChatTurn]  # one conversation
BatchChatMessagesHF = list[ChatMessageHF]  # batch of conversations


class Predictor(Protocol):
    def infer(self,
              model_name: str,
              messages: BatchChatMessagesHF | ChatMessageHF,
              sampling_params: SamplingParams,
              *args,
              **kwargs) -> list[list[str]] | list[str]:
        pass


class VLLMPredictor:
    def infer(self,
              model_name,
              messages: BatchChatMessagesHF | ChatMessageHF,
              sampling_params: SamplingParams,
              tp=1,
              dp=1,
              max_model_len=8192,
              *args,
              **kwargs) -> list[list[str]] | list[str]:
        enable_thinking_mode = kwargs.get('enable_thinking_mode', False)
        self.model_name = model_name
        if sampling_params.stop_token_ids is None or len(sampling_params.stop_token_ids) == 0:
            logger.info(f'Adding stop_token_ids to sampling_params: {self.stop_token_ids}.')
            sampling_params.stop_token_ids = self.stop_token_ids
        # Normalize and validate messages against HF chat template
        logger.info(f'Loading model {model_name} with tp={tp}, dp={dp}, max_model_len={max_model_len}.')
        llm = self._load_model(tp, dp, max_model_len)
        tokenizer = self._load_tokenizer()
        # reasoning effort: https://huggingface.co/docs/trl/main/dataset_formats#harmony
        if 'gpt' in self.model_name:
            chat_prompts = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                reasoning_effort=ReasoningEffort.HIGH
            )
        else:
            chat_prompts = tokenizer.apply_chat_template(messages,
                                                         tokenize=False,
                                                         add_generation_prompt=True,
                                                         enable_thinking=enable_thinking_mode,
                                                         )
        output = llm.generate(chat_prompts, sampling_params)
        responses = [o.outputs[0].text for o in output]
        return responses

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        return tokenizer

    def _load_model(self, tp, dp, max_model_len):
        # https://docs.vllm.ai/en/latest/configuration/optimization.html#performance-tuning-with-chunked-prefill
        llm = LLM(
            model=self.model_name,
            dtype="bfloat16",
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.92,
            swap_space=42,
            # enforce_eager=True,
            # disable_custom_all_reduce=True,
            trust_remote_code=True,
            data_parallel_size=dp,
            max_num_batched_tokens=16000,
        )
        return llm

    @property
    def stop_token_ids(self):
        name = self.model_name.lower()
        mapping = [
            (["deepseek-coder-"], [32021]),
            (["deepseek-coder-v2"], [100001]),
            (["gpt-oss"], [200002, 199999, 200012]),
            (["opencoder-"], [96539]),
            (["meta-llama-"], [128009, 128001]),
            (["granite-", "starcoder2-"], [0]),
            (["codestral-", "mixtral-"], [2]),
            (["omnisql-", "arctic-", "sql-r1-", "qwen2.5-", "qwen3-"], [151645]),
        ]
        for keys, ids in mapping:
            if any(k in name for k in keys):
                return ids

        logger.info(f'No stop_token_ids found for model {self.model_name}, using default of [151645].')
        return [151645]


class LiteLLMPredictor:
    def infer(self,
              model_name,
              messages: BatchChatMessagesHF | ChatMessageHF,
              sampling_params: SamplingParams,
              litellm_provider: str,
              *args,
              **kwargs) -> list[list[str]] | list[str]:
        import litellm
        # litellm._turn_on_debug()
        litellm.request_timeout = 6000  # increase request timeout to 6000 seconds

        logger.info(f"Inferring with LiteLLM. Provider: {litellm_provider}, Model: {model_name}")

        # litellm.batch_completion expects a list of requests;
        is_single = False
        if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
            messages = [messages]  # wrap single conversation into a list
            is_single = True

        api_base = None
        if litellm_provider == 'hosted_vllm':
            base_url = f"http://{kwargs.get('vllm_server_host', 'localhost')}:{kwargs.get('vllm_server_port', 8000)}"
            api_base = f"{base_url}/v1"
            logger.info(f'Using hosted_vllm with api_base: {api_base}')
            check_server_availability(base_url, total_timeout=500, retry_interval=20)

        # stream=True makes each item in the returned list a streaming iterator
        model_answer = litellm.batch_completion(
            model=f"{litellm_provider}/{model_name}",
            messages=messages,
            temperature=sampling_params.temperature,
            max_tokens=sampling_params.max_tokens if 'gpt' not in model_name.lower() else None,
            top_p=sampling_params.top_p,
            n=sampling_params.n,
            max_completion_tokens=sampling_params.max_tokens,
            presence_penalty=sampling_params.presence_penalty,
            frequency_penalty=sampling_params.frequency_penalty,
            api_base=api_base,
            # rpm=kwargs.get('rpm', None),
            drop_params=True,
            reasoning_effort='high' if 'gpt' in model_name.lower() else None,
            max_workers=200,
            chat_template_kwargs={"enable_thinking": kwargs.get('enable_thinking_mode', False)}
            # stream=True if sampling_params.n > 1 or sampling_params.max_tokens > 15000 else False,
        )

        parsed_responses = self.parse_model_output(model_answer)
        return parsed_responses if not is_single else parsed_responses[0]

    def parse_model_output(
            self, model_answer: list[ModelResponse | CustomStreamWrapper]
    ) -> list[list[str] | str]:

        parsed_response: list[list[str]] = []
        for out in model_answer:
            if isinstance(out, CustomStreamWrapper):
                choices_response = self._parse_single_stream_response(out)
            else:
                if not isinstance(out, ModelResponse):
                    if isinstance(out, BaseException):
                        raise out
                    raise ValueError(f'The output is not of type ModelResponse but type {type(out)}: {out}')

                choices_response = [choice['message']['content'] for choice in out['choices']]

            parsed_response.append(choices_response)

        parsed_response = [
            response if len(response) > 1 else response[0]
            for response in parsed_response
        ]

        return parsed_response

    def _parse_single_stream_response(self, stream_model_response: CustomStreamWrapper) -> list[str]:
        buckets = defaultdict(list)
        # stream contains N different streams one for each decoding strategy N requested
        for chunk in stream_model_response:
            for choice in chunk.choices:
                buckets[choice.index].append(choice.delta.content or "")
        # combine all chunks for the different choices
        choices = ["".join(choice) for choice in buckets.values()]
        return choices


if __name__ == "__main__":
    predictor = LiteLLMPredictor()
    sampling_params = SamplingParams(
        n=3,
        temperature=1.0,
        max_tokens=2000,
    )
    messages: BatchChatMessagesHF = [
        [{"role": "user", "content": "Write a Python function to add two numbers."}],
        [{"role": "user", "content": "Write a Python function to add two numbers."}],
        [{"role": "user", "content": "Write a Python function to add two numbers."}],
    ]

    output = predictor.infer(
        model_name='Qwen/Qwen3-1.7B',
        litellm_provider='hosted_vllm',
        sampling_params=sampling_params,
        messages=messages,
        vllm_server_host='0.0.0.0',
        vllm_server_port=8000
    )

    print(output)
