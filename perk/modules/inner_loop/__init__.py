"""Inner-loop strategies for meta-learning (recursive gradient unrolling)."""

import torch
from typing import Dict, List, Type

from perk.modules.inner_loop.base import InnerLoopStrategy
from perk.modules.inner_loop.rgu import RGUInnerLoop

__all__ = [
    "InnerLoopStrategy",
    "RGUInnerLoop",
    "INNER_LOOP_STRATEGIES",
    "create_inner_loop",
]

INNER_LOOP_STRATEGIES: Dict[str, Type[InnerLoopStrategy]] = {
    "rgu": RGUInnerLoop,
}


def create_inner_loop(
    inner_loop_type: str,
    model,
    inner_params_idx: List[int],
    lrs,
    hparams,
    device: torch.device,
    verbose: bool = False,
    local_rank: int = 0,
    **kwargs,
) -> InnerLoopStrategy:
    """Create an inner-loop strategy by name."""
    if inner_loop_type not in INNER_LOOP_STRATEGIES:
        raise ValueError(
            f"Unknown inner loop type: {inner_loop_type}. "
            f"Available types: {list(INNER_LOOP_STRATEGIES.keys())}"
        )
    return INNER_LOOP_STRATEGIES[inner_loop_type](
        model=model,
        inner_params_idx=inner_params_idx,
        lrs=lrs,
        hparams=hparams,
        device=device,
        verbose=verbose,
        local_rank=local_rank,
        **kwargs,
    )
