"""Outer-loop strategies for meta-learning (cross-entropy objective)."""

from typing import Dict, Type

import torch

from perk.modules.outer_loop.base import OuterLoopStrategy
from perk.modules.outer_loop.ce import CrossEntropyOuterLoop

__all__ = [
    "OuterLoopStrategy",
    "CrossEntropyOuterLoop",
    "OUTER_LOOP_STRATEGIES",
    "create_outer_loop",
]

OUTER_LOOP_STRATEGIES: Dict[str, Type[OuterLoopStrategy]] = {
    "ce": CrossEntropyOuterLoop,
}


def create_outer_loop(
    outer_loop_type: str,
    model,
    hparams,
    device: torch.device,
    local_rank: int = 0,
    **kwargs,
) -> OuterLoopStrategy:
    """Create an outer-loop strategy by name."""
    if outer_loop_type not in OUTER_LOOP_STRATEGIES:
        raise ValueError(
            f"Unknown outer loop type: {outer_loop_type}. "
            f"Available types: {list(OUTER_LOOP_STRATEGIES.keys())}"
        )
    return OUTER_LOOP_STRATEGIES[outer_loop_type](
        model=model,
        hparams=hparams,
        device=device,
        local_rank=local_rank,
        **kwargs,
    )
