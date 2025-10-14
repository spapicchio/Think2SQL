import time
from typing import Protocol, Literal
from typing import TypedDict

import requests
from dotenv import load_dotenv
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
            chat_prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        output = llm.generate(chat_prompts, sampling_params)
        responses = [o.outputs[0].text for o in output]
        return responses

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        return tokenizer

    def _load_model(self, tp, dp, max_model_len):
        llm = LLM(
            model=self.model_name,
            dtype="bfloat16",
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.92,
            swap_space=42,
            enforce_eager=False,
            disable_custom_all_reduce=True,
            trust_remote_code=True,
            data_parallel_size=dp,
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
        raise ValueError(f"stop_token_ids for model {self.model_name} is not defined.")

        return stop_token_ids


class LiteLLMPredictor:
    def infer(self,
              model_name,
              messages: BatchChatMessagesHF | ChatMessageHF,
              sampling_params: SamplingParams,
              litellm_provider: str,
              *args,
              **kwargs) -> list[list[str]] | list[str]:
        from litellm import batch_completion
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
            check_server_availability(base_url, total_timeout=500, retry_interval=60)

        model_answer = batch_completion(
            model=f"{litellm_provider}/{model_name}",
            messages=messages,
            temperature=sampling_params.temperature,
            max_tokens=sampling_params.max_tokens,
            top_p=sampling_params.top_p,
            n=sampling_params.n,
            max_completion_tokens=sampling_params.max_tokens,
            presence_penalty=sampling_params.presence_penalty,
            frequency_penalty=sampling_params.frequency_penalty,
            api_base=api_base,
            rpm=kwargs.get('rpm', None),
            drop_params=True,
            reasoning_effort='high' if 'gpt' in model_name.lower() else None,
        )

        parsed_responses = self.parse_model_output(model_answer)
        return parsed_responses if not is_single else parsed_responses[0]

    def parse_model_output(self, model_answer: list) -> list[list[str] | str]:
        parsed_response = []
        for out in model_answer:
            choices_response = []
            for choice in out['choices']:
                message = choice['message']['content']
                if 'reasoning_content' in choice['message']:
                    message = choice['message']['reasoning_content'] + "\n" + message
                choices_response.append(message)
            parsed_response.append(choices_response)

        parsed_response = [response if len(response) > 1 else response[0] for response in parsed_response]
        return parsed_response


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
        model_name='gpt-5-mini-2025-08-07',
        litellm_provider='openai',
        sampling_params=sampling_params,
        messages=messages,
        vllm_server_host='127.0.0.1',
        vllm_server_port=24879
    )

    print(output)
