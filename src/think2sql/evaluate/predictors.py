from typing import Protocol

from transformers import AutoTokenizer
from vllm import SamplingParams, LLM

from think2sql.logger import get_logger

logger = get_logger(__name__)


class Predictor(Protocol):
    def infer(self, model_name, messages, *args, **kwargs):
        pass


class VLLMPredictor:
    def infer(self,
              messages,
              model_name,
              sampling_params: SamplingParams,
              tp=1,
              dp=1,
              max_model_len=8192,
              *args,
              **kwargs):
        self.model_name = model_name
        if sampling_params.stop_token_ids is None or len(sampling_params.stop_token_ids) == 0:
            logger.info(f'Adding stop_token_ids to sampling_params: {self.stop_token_ids}.')
            sampling_params.stop_token_ids = self.stop_token_ids

        logger.info(f'Loading model {model_name} with tp={tp}, dp={dp}, max_model_len={max_model_len}.')
        llm = self._load_model(tp, dp, max_model_len)
        tokenizer = self._load_tokenizer()
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
        if "qwen2.5-" in self.model_name:
            stop_token_ids = [151645]  # 151645 is the token id of <|im_end|> (end of turn token in Qwen2.5)
        elif "deepseek-coder-" in self.model_name.lower():
            stop_token_ids = [32021]
        elif "deepseek-coder-v2" in self.model_name.lower():
            stop_token_ids = [100001]
        elif "opencoder-" in self.model_name.lower():
            stop_token_ids = [96539]
        elif "meta-llama-" in self.model_name.lower():
            stop_token_ids = [128009, 128001]
        elif "granite-" in self.model_name.lower():
            stop_token_ids = [0]  # <|end_of_text|> is the end token of granite-3.1 and granite-code
        elif "starcoder2-" in self.model_name.lower():
            stop_token_ids = [0]  # <|end_of_text|> is the end token of starcoder2
        elif "codestral-" in self.model_name.lower():
            stop_token_ids = [2]
        elif "mixtral-" in self.model_name.lower():
            stop_token_ids = [2]
        elif "omnisql-" in self.model_name.lower() or "arctic-" in self.model_name.lower():
            stop_token_ids = [151645]  # OmniSQL uses the same tokenizer as Qwen2.5
        elif "sql-r1-" in self.model_name.lower():
            stop_token_ids = [151645, 151643]
        elif "qwen2.5-" in self.model_name.lower():
            stop_token_ids = [151645, 151643]
        else:
            raise ValueError(f"stop_token_ids for model {self.model_name} is not defined.")

        return stop_token_ids
