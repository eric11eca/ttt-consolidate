"""Learned inner learning-rate table and the functional AdamW used by the inner loop."""

import math
import hashlib
import json
import torch
import torch.nn as nn
from typing import Any, List, Tuple, Optional, Sequence

# Saved in every checkpoint (with the `row_names` below); the checkpoint loader
# compares them, so changing either value makes existing checkpoints unloadable.
LR_TABLE_FORMAT = "one_based_v1"

# =============================================================================
# Utility functions
# =============================================================================

class LRParameters(nn.Module):
    """Learned inner-step table with a one-based layout."""

    def __init__(
        self,
        num_inner_iter,
        init_lr=1e-5,
        init_decay=5e-4,
        params=None,
        row_names: Optional[Sequence[str]] = None,
    ):
        super(LRParameters, self).__init__()
        params = list(params or [])
        self.num_inner_iter = int(num_inner_iter)
        self.init_lr = float(init_lr)
        self.init_decay = init_decay
        self._strict_registry = row_names is not None
        if row_names is None:
            row_names = tuple(f"legacy_row_{index}" for index in range(len(params)))
        self.row_names = tuple(str(name) for name in row_names)
        if len(self.row_names) != len(params) or len(set(self.row_names)) != len(params):
            raise ValueError("learned LR row registry must be unique and match params")
        values = torch.full((len(params), self.num_inner_iter + 1), self.init_lr)
        values[:, 0] = 0.0
        self.lr = nn.Parameter(values, requires_grad=True)
        self.register_buffer(
            "lr_sentinel",
            torch.tensor(0.0, dtype=torch.float32),
            persistent=True,
        )
        payload = {
            "abi": LR_TABLE_FORMAT,
            "row_names": list(self.row_names),
            "num_inner_iter": self.num_inner_iter,
            "sentinel_column": 0,
        }
        self.registry_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def registry_metadata(self) -> dict[str, Any]:
        return {
            "abi": LR_TABLE_FORMAT,
            "row_names": list(self.row_names),
            "num_inner_iter": self.num_inner_iter,
            "sentinel_column": 0,
            "registry_hash": self.registry_hash,
        }

    def get_extra_state(self):
        return self.registry_metadata()

    def set_extra_state(self, state):
        if state is None and not self._strict_registry:
            return
        self._validate_registry_state(state)

    def _validate_registry_state(self, state) -> None:
        if not isinstance(state, dict) or state != self.registry_metadata():
            raise RuntimeError("learned LR row registry checkpoint mismatch")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        registry = state_dict.get(prefix + "_extra_state")
        if registry is None:
            if self._strict_registry:
                raise RuntimeError("learned LR row registry is missing from checkpoint")
        else:
            self._validate_registry_state(registry)
        lr = state_dict.get(prefix + "lr")
        if lr is not None:
            if tuple(lr.shape) != tuple(self.lr.shape):
                raise RuntimeError("one_based learned LR table shape mismatch")
            if lr.numel() and not torch.equal(
                lr[:, 0].detach().cpu(), torch.zeros_like(lr[:, 0].detach().cpu())
            ):
                raise RuntimeError("one_based sentinel column must remain zero")
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


# =============================================================================
# AdamW optimizer (functional)
# =============================================================================

def adamw(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    beta1: float,
    beta2: float,
    weight_decay: float,
    lr_param: torch.Tensor,
    step: int,
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    eps: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Functional AdamW optimizer step.

    Args:
        params: List of parameters
        grads: List of gradients
        beta1: First moment decay
        beta2: Second moment decay
        weight_decay: Weight decay coefficient (AdamW-style)
        lr_param: Learning rate parameter tensor
        step: Current optimization step (0-indexed, will be incremented internally)
        exp_avgs: List of first moment estimates
        exp_avg_sqs: List of second moment estimates
        eps: Numerical stability term

    Returns:
        Tuple of (updates, new_exp_avgs, new_exp_avg_sqs)
    """
    step += 1
    r_updates, r_exp_avgs, r_exp_avg_sqs = [], [], []

    for _i, (p, g, exp_avg, exp_avg_sq) in enumerate(zip(params, grads, exp_avgs, exp_avg_sqs)):
        if g is None:
            raise RuntimeError('AdamW does not support None gradients!')

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        # Use lerp for more efficient EMA updates
        exp_avg = torch.lerp(g, exp_avg, beta1)
        exp_avg_sq = torch.lerp(g * g, exp_avg_sq, beta2)

        denom = torch.sqrt(exp_avg_sq + 1e-8) / math.sqrt(bias_correction2) + eps
        step_size = lr_param[_i, step] / bias_correction1
        update = -step_size * exp_avg / denom

        if len(p.shape) > 1:
            update -= p * lr_param[_i, step] * weight_decay

        r_updates.append(update)
        r_exp_avgs.append(exp_avg)
        r_exp_avg_sqs.append(exp_avg_sq)

    return r_updates, r_exp_avgs, r_exp_avg_sqs
