"""Cross-entropy outer loop: loss of the adapted model on the query, plus a KL term."""

import torch
import torch.nn.functional as F
from typing import Any, Dict

from torch.nn.attention import SDPBackend, sdpa_kernel

from perk.modules.outer_loop.base import OuterLoopStrategy


def kl_loc_loss(pre, post, mask=None):
    if pre.shape != post.shape:
        raise ValueError("pre/post KL logits must have identical shape")
    pre = pre.detach().to(torch.float32)
    post = post.to(torch.float32)

    sequence = pre.dim() == 3
    pre_ = pre.view(-1, pre.shape[-1])
    post_ = post.view(-1, pre.shape[-1])

    if not sequence:
        if pre_.shape[-1] == 1: # No masking needed for binary classification
            return pre.sigmoid() * (F.logsigmoid(pre) - F.logsigmoid(post)).mean() + (
                (-pre).sigmoid() * (F.logsigmoid(-pre) - F.logsigmoid(-post))
            ).mean()
        raise ValueError("non-sequence KL supports binary logits only")
    if pre_.shape[-1] <= 1:
        raise ValueError("sequence KL requires a non-binary vocabulary")
    if mask is None:
        raise ValueError("canonical KL requires an explicit task attention mask")
    if mask.shape != pre.shape[:-1]:
        raise ValueError("canonical KL mask must match the logits leading shape")
    if mask.dtype != torch.bool:
        raise ValueError("canonical KL mask dtype must be bool")
    if mask.device != pre.device:
        raise ValueError("canonical KL mask and logits must share one device")
    mask_ = mask.reshape(pre_.shape[0])
    count = mask_.sum(dtype=torch.int64)
    if int(count.detach().cpu().item()) <= 0:
        raise ValueError("canonical KL task attention mask must be non-empty")
    pre_logp = pre_.log_softmax(-1)
    post_logp = post_.log_softmax(-1)
    per_token = (pre_logp.exp() * (pre_logp - post_logp)).sum(-1)
    numerator = torch.where(mask_, per_token, torch.zeros_like(per_token)).sum(
        dtype=torch.float32
    )
    return numerator / count.to(dtype=torch.float32)


class CrossEntropyOuterLoop(OuterLoopStrategy):
    """
    Standard Cross-Entropy outer loop with optional KL regularization.

    Computes cross-entropy loss on the outer batch using the adapted model,
    with optional KL divergence regularization to prevent drift from initial model.

    This class handles:
    - Outer loss computation on adapted parameters
    - Outer gain metric (loss improvement after adaptation)
    - L2 regularization on adapted parameters
    - KL divergence regularization to prevent drift
    """

    def compute_loss(
        self,
        inner_result: Dict[str, Any],
        outer_batches: Dict,
        print_out: Dict,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute cross-entropy loss with regularization and metrics.

        Args:
            inner_result: Result from inner loop (params, fmodel, buffers)
            outer_batches: Outer loop batch data
            print_out: Print output data

        Returns:
            Dictionary with loss, metrics, and manual_backward flag
        """
        fmodel = inner_result["fmodel"]
        params = inner_result["params"]
        buffers = inner_result["buffers"]

        init_outer_loss = inner_result.get("init_outer_loss")
        init_outer_logits = inner_result.get("init_outer_logits")

        metrics = {}
        if inner_result.get("forward_mode_done", False):
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                target_loss = inner_result["outer_loss"]
            if init_outer_loss is not None:
                outer_gain = ((init_outer_loss - target_loss) / init_outer_loss).detach()
                metrics['outer_gain'] = outer_gain.item() if isinstance(outer_gain, torch.Tensor) else outer_gain

            return {
                "loss": target_loss,
                "metrics": metrics,
                "manual_backward": True,
            }

        c_kl = getattr(self.hparams, 'c_kl', 0.0)
        if init_outer_loss is None or (c_kl > 0 and init_outer_logits is None):
            with torch.no_grad():
                with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    init_outer = self.model(outer_batches)
                init_outer_loss = init_outer["loss"]
                init_outer_logits = init_outer["logits"]

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            target = fmodel(params, buffers, outer_batches)
        target_loss = target["loss"]
        target_logits = target["logits"]

        if init_outer_loss is not None:
            outer_gain = ((target_loss - init_outer_loss) / init_outer_loss).detach()
            metrics['outer_gain'] = outer_gain.item() if isinstance(outer_gain, torch.Tensor) else outer_gain

        outer_reg_lambda = getattr(self.hparams, 'outer_reg_lambda', 0.0)
        if outer_reg_lambda > 0:
            l2_reg = 0.0
            for param in params:
                if param.requires_grad and param.numel() > 0:
                    l2_reg += torch.norm(param, p=2)**2
            target_loss = target_loss + outer_reg_lambda * l2_reg
            metrics['l2_reg'] = l2_reg.item() if isinstance(l2_reg, torch.Tensor) else l2_reg

        c_kl = getattr(self.hparams, 'c_kl', 0.0)
        if c_kl > 0 and init_outer_logits is not None:
            kl_div = kl_loc_loss(
                init_outer_logits, target_logits,
                outer_batches['attention_mask'].bool())
            target_loss = target_loss + c_kl * kl_div / len(outer_batches)
            metrics['kl_div'] = kl_div.item() if isinstance(kl_div, torch.Tensor) else kl_div

        return {
            "loss": target_loss,
            "metrics": metrics,
            "manual_backward": False,
        }
