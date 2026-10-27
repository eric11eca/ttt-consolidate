"""Text generation with a Hugging Face model, using the same prompt format as training."""

import logging
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import logging as hf_logging

from perk.datasets.meta_dataset import PROMPT_COMPLETION_SEPARATOR, uses_chat_template

hf_logging.set_verbosity_info()

util_logger = logging.getLogger("perk.inference")
util_logger.setLevel(logging.INFO)


def _format_chat_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    """Format a generation prompt the same way training formats the prompt."""
    if not uses_chat_template(tokenizer):
        return f"{prompt}{PROMPT_COMPLETION_SEPARATOR}"
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Some tokenizers do not support enable_thinking.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _set_if_not_none(target: Dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


class HFGenerator:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cuda:0",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            self._pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
            )
        return self._pipeline

    @torch.no_grad()
    def generate(self, queries, **kwargs):
        text_generator = self._get_pipeline()

        generation_kwargs = {
            "num_return_sequences": 1,
            "return_full_text": False,
        }
        _set_if_not_none(generation_kwargs, "pad_token_id", kwargs.get("pad_token_id"))
        _set_if_not_none(generation_kwargs, "do_sample", kwargs.get("do_sample"))
        _set_if_not_none(generation_kwargs, "top_p", kwargs.get("top_p"))
        _set_if_not_none(generation_kwargs, "temperature", kwargs.get("temperature"))
        _set_if_not_none(generation_kwargs, "max_new_tokens", kwargs.get("max_new_tokens"))

        for query in queries:
            prompt = _format_chat_prompt(self.tokenizer, query["prompt"])
            response = text_generator(prompt, **generation_kwargs)
            query["output"] = response[0]["generated_text"] if response else ""
        return queries

