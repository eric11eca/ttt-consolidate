"""Causal language model with LoRA adapters and an optional learned token-weighting model."""

import os
import torch
import logging
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType
)
from perk.weight_model import build_weight_model

logger = logging.getLogger("perk.model")


def _cache_dir():
    """Hub cache under ``$HF_HOME`` when set; otherwise the library default."""
    home = os.environ.get("HF_HOME")
    return os.path.join(home, "hub") if home else None


class CausalLM(nn.Module):
    """
    Decoder-only language model wrapped with LoRA adapters and an optional
    learned per-token loss-weighting model.
    """

    def __init__(self, model, tokenizer, weight_model, model_config, global_config):
        super().__init__()
        self.lm_model = model
        self.weight_model = weight_model
        self.tokenizer = tokenizer
        self.config = model_config
        self.global_config = global_config

    @classmethod
    def from_config(cls, config):
        """
        Loads a pretrained decoder-only causal LLM from configuration

        :param config: the global configuration
        :rtype CausalLM
        """
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            padding_side="left")
        model_config = AutoConfig.from_pretrained(config.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            config=model_config,
            torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
            attn_implementation=config.attn_implementation,
            trust_remote_code=False,
            cache_dir=_cache_dir())
        model.train()

        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id
        if "llama" in config.model_name_or_path:
            tokenizer.pad_token_id = 128004
            model.config.pad_token_id = 128004
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

        if config.peft_lora:
            target_modules = list(getattr(config, "lora_target_modules", None) or [
                "q_proj", "o_proj", "k_proj", "v_proj",
                "gate_proj", "up_proj", "down_proj",
                "c_attn", "c_proj", "c_fc"])

            num_layers = model.config.num_hidden_layers
            mem_depth = config.mem_depth
            layers_to_transform = list(range(num_layers - mem_depth, num_layers))
            logger.info(f"Layers adapted: {layers_to_transform[0]} - {layers_to_transform[-1]}")

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                use_rslora=bool(getattr(config, "use_rslora", True)),
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=float(getattr(config, "lora_dropout", 0.1)),
                target_modules=target_modules,
                layers_to_transform=layers_to_transform
            )
            model = get_peft_model(model, peft_config, autocast_adapter_dtype=True)

            trainable_params, all_param = model.get_nb_trainable_parameters()
            logger.info("PEFT model loaded")
            logger.info(f"trainable parameters: {trainable_params:,d} || all params: {all_param:,d}")
            logger.info(f"trainable%: {100 * trainable_params / all_param:.4f}")

        weight_model = build_weight_model(config, model.config.hidden_size)

        return cls(
            model,
            tokenizer,
            weight_model,
            model_config,
            config,
        )

    def compute_weights(self, outputs,features):
        attn_mask = features.get("attention_mask")
        labels = features.get("labels")

        token_mask = None
        if torch.is_tensor(attn_mask) and attn_mask.dim() <= 2:
            token_mask = attn_mask.bool()
        if labels is not None:
            label_mask = labels != -100
            token_mask = label_mask if token_mask is None else (token_mask & label_mask)
        if token_mask is None:
            token_mask = torch.ones_like(features["input_ids"], dtype=torch.bool)

        token_mask = token_mask.to(
            device=outputs["hidden_states"][-1].device,
            dtype=outputs["hidden_states"][-1].dtype,
        )

        weights = self.weight_model(outputs["hidden_states"][-1], token_mask)
        return weights

    def weighted_loss(self, outputs, features, weights):
        logits = outputs["logits"]
        loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=-100, reduction='none')
        batch_size = len(features["input_ids"])
        reshaped_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        reshaped_labels = features["labels"][:, 1:].reshape(-1)
        loss = loss_fn(reshaped_logits, reshaped_labels)
        loss = (loss.reshape(batch_size, -1) * weights[:, 1:]).mean()
        return loss

    def forward(
        self,
        features,
        weights=None,
        is_inner=False
    ):
        """A modified version of forward pass for the language model.

        :param features: the target inputs
        :param weights: the weights
        :param is_inner: whether the forward is for inner loop
        :rtype: dict
        :returns: the outputs of the language model
        """
        outputs = self.lm_model(
            **features,
            return_dict=True,
            output_hidden_states=True)

        if self.global_config.weighted_loss and is_inner:
            weights = self.compute_weights(outputs, features)
            outputs["loss"] = self.weighted_loss(outputs, features, weights)

        return outputs