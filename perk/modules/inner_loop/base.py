"""Base class for inner-loop strategies, with the shared functional forward and update."""

import torch
import functorch

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
from torch.nn.attention import SDPBackend, sdpa_kernel

from perk.modules.optimizers import adamw

def print_mem(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"{prefix} => allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")

class InnerLoopStrategy(ABC):
    """Abstract base class for inner loop optimization strategies."""

    def __init__(
        self,
        model,
        inner_params_idx: List[int],
        lrs,
        hparams,
        device: torch.device,
        verbose: bool = False,
        local_rank: int = 0,
    ):
        """
        Initialize inner loop strategy.

        Args:
            model: The model to optimize
            inner_params_idx: List of parameter indices to update in inner loop
            lrs: Learning rate scheduler/parameters
            hparams: Hyperparameters configuration
            device: Device to run on
            verbose: Whether to print debug information
            local_rank: Local rank for distributed training
        """
        self.model = model
        self.inner_params_idx = inner_params_idx
        self.lrs = lrs
        self.hparams = hparams
        self.device = device
        self.verbose = verbose
        self.local_rank = local_rank

        self.inner_opt = getattr(hparams, "inner_opt", "adamw").lower()
        if self.inner_opt != "adamw":
            raise ValueError(f"Unknown inner_opt: {self.inner_opt}. Supported: adamw")

    def _make_functional(self):
        """Make model functional for differentiation."""
        fmodel, params, buffers = functorch.make_functional_with_buffers(self.model)
        params = list(self.model.parameters())
        return fmodel, params, buffers

    def _filter_inner_params(self, params: List[torch.Tensor]) -> List[torch.Tensor]:
        """Filter parameters to only include inner loop parameters."""
        return [
            p if _i in self.inner_params_idx else torch.tensor(
                [], requires_grad=True, device=self.device)
            for _i, p in enumerate(params)
        ]

    def _init_optimizer_states(self, inner_params: List[torch.Tensor]) -> Dict[str, Any]:
        """
        Initialize the AdamW moment estimates.

        Args:
            inner_params: List of inner loop parameters

        Returns:
            Dictionary with the `exp_avgs` and `exp_avg_sqs` tensors
        """
        return {
            "exp_avgs": [torch.zeros_like(p, requires_grad=False) for p in inner_params],
            "exp_avg_sqs": [torch.zeros_like(p, requires_grad=False) for p in inner_params],
        }

    def inner_forward(
        self,
        fmodel,
        params: List[torch.Tensor],
        buffers,
        batch: Dict,
        step_index: int,
    ) -> torch.Tensor:
        """
        Compute inner loop forward pass loss.

        Args:
            fmodel: Functional model
            params: Model parameters
            buffers: Model buffers
            batch: Input batch
            step_index: Current inner loop step index

        Returns:
            Inner loop loss tensor
        """
        if self.verbose and self.local_rank == 0:
            print_mem(f"  [Inner Forward {step_index}] Start")

        with sdpa_kernel(backends=[SDPBackend.MATH]):
            inner_loss = fmodel(
                params, buffers, batch,
                None,  # inner_weights
                is_inner=True
            )["loss"]

        if self.verbose and self.local_rank == 0:
            print_mem(f"  [Inner Forward {step_index}] End")

        torch.cuda.empty_cache()
        return inner_loss

    def inner_update(
        self,
        params: List[torch.Tensor],
        inner_params: List[torch.Tensor],
        grads: List[Optional[torch.Tensor]],
        step_index: int,
        opt_states: Dict[str, Any],
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Apply an AdamW update to the inner parameters.

        Args:
            params: All model parameters
            inner_params: Inner loop parameters (filtered)
            grads: Gradients for inner params
            step_index: Current inner loop step index
            opt_states: Dictionary containing optimizer state tensors

        Returns:
            Tuple of (updated_params, updated_opt_states)
        """
        if step_index < self.hparams.unroll_start:
            grads = [g.detach() if g is not None else None for g in grads]

        if self.verbose and self.local_rank == 0:
            print_mem(f"  [Inner Update {step_index}]")

        updates, exp_avgs, exp_avg_sqs = adamw(
            inner_params,
            grads,
            beta1=getattr(self.hparams, "inner_beta1", 0.9),
            beta2=getattr(self.hparams, "inner_beta2", 0.95),
            weight_decay=self.hparams.weight_decay,
            lr_param=self.lrs.lr,
            step=step_index,
            exp_avgs=opt_states["exp_avgs"],
            exp_avg_sqs=opt_states["exp_avg_sqs"],
            eps=getattr(self.hparams, "inner_eps", 1e-10),
        )
        new_opt_states = {"exp_avgs": exp_avgs, "exp_avg_sqs": exp_avg_sqs}

        params = [
            p + u if (u is not None and u.numel() > 0) else p
            for p, u in zip(params, updates)
        ]
        torch.cuda.empty_cache()
        return params, new_opt_states

    @abstractmethod
    def run(
        self,
        inner_batches: List[Dict],
        outer_batches: Optional[Dict] = None,
        return_parameters: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the inner loop optimization.

        Args:
            inner_batches: List of inner loop training batches
            outer_batches: Optional outer batch for computing outer loss
            return_parameters: Whether to return adapted parameters
            **kwargs: Additional strategy-specific arguments

        Returns:
            Dictionary containing:
                - inner_loss: Final inner loop loss
                - outer_loss: Outer loop loss (if outer_batches provided)
                - params: Adapted parameters (if return_parameters=True)
                - fmodel: Functional model (optional)
                - buffers: Model buffers (optional)
        """
        pass
