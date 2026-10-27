"""Base class for outer-loop strategies."""

import torch
from abc import ABC, abstractmethod
from typing import Any, Dict


class OuterLoopStrategy(ABC):
    """Abstract base class for outer loop optimization strategies."""

    def __init__(
        self,
        model,
        hparams,
        device: torch.device,
        local_rank: int = 0,
    ):
        """
        Initialize outer loop strategy.

        Args:
            model: The model
            hparams: Hyperparameters configuration
            device: Device to run on
            local_rank: Local rank for distributed training
        """
        self.model = model
        self.hparams = hparams
        self.device = device
        self.local_rank = local_rank

    @abstractmethod
    def compute_loss(
        self,
        inner_result: Dict[str, Any],
        outer_batches: Dict,
        print_out: Dict,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute outer loop loss and perform backward pass if needed.

        Args:
            inner_result: Result from inner loop (contains params, fmodel, buffers, etc.)
            outer_batches: Outer loop batch data
            print_out: Print output data (prompts, responses, etc.)
            **kwargs: Additional strategy-specific arguments

        Returns:
            Dictionary containing:
                - loss: The outer loop loss tensor
                - metrics: Dictionary of additional metrics to be logged
                - manual_backward: Whether backward was already performed
        """
        pass
