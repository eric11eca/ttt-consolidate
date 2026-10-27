"""Recursive gradient unrolling (RGU): inner-loop adaptation with checkpointed second-order gradients."""

import torch
from typing import Any, Dict, List, Optional
from torch.utils.checkpoint import checkpoint

from perk.modules.inner_loop.base import InnerLoopStrategy, print_mem


class RGUInnerLoop(InnerLoopStrategy):
    """
    Recursive Gradient Unrolling inner loop.

    Uses gradient checkpointing to save memory while maintaining
    differentiability through the inner loop.
    """

    def run(
        self,
        inner_batches: List[Dict],
        outer_batches: Optional[Dict] = None,
        return_parameters: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run recursive gradient unrolling inner loop.

        Args:
            inner_batches: List of inner loop training batches
            outer_batches: Not used (kept for API consistency)
            return_parameters: Whether to return adapted parameters

        Returns:
            Dictionary with inner_loss, params, fmodel, and buffers
        """
        fmodel, params, buffers = self._make_functional()
        inner_params = self._filter_inner_params(params)
        opt_states = self._init_optimizer_states(inner_params)

        def compute_grad(inner_loss, inner_params):
            if self.verbose:
                print_mem("Computing inner-grads")
            grads = torch.autograd.grad(
                inner_loss, inner_params,
                create_graph=not return_parameters,
                allow_unused=True,
                materialize_grads=True
            )
            torch.cuda.empty_cache()
            return grads

        # Inner loop iterations
        for _iter in range(self.hparams.n_inner_iter):
            if self.verbose:
                print(f"inner loop step: {_iter}")

            accum_grads = [None] * len(params)

            for batch in inner_batches:
                inner_loss = checkpoint(
                    self.inner_forward,
                    fmodel,
                    params,
                    buffers,
                    batch,
                    _iter,
                    use_reentrant=False,
                )
                inner_loss = inner_loss / len(inner_batches)

                inner_params = self._filter_inner_params(params)

                grads = checkpoint(
                    compute_grad,
                    inner_loss,
                    inner_params,
                    use_reentrant=False
                )

                for i, g in enumerate(grads):
                    if g is not None:
                        if accum_grads[i] is None:
                            accum_grads[i] = g
                        else:
                            accum_grads[i] = accum_grads[i] + g

            params, opt_states = checkpoint(
                self.inner_update,
                params,
                inner_params,
                accum_grads,
                _iter,
                opt_states,
                use_reentrant=False,
            )

        return {
            "inner_loss": inner_loss,
            "params": params,
            "fmodel": fmodel,
            "buffers": buffers,
        }
