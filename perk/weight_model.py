"""Learned per-token loss weights: a small network on the final hidden states."""
from typing import Optional

import torch
import torch.nn as nn


class LinearWeightingModel(nn.Module):
    """Two-layer network that maps each token's hidden state to a positive loss weight."""

    def __init__(
        self,
        normalize=False,
        non_linearity="gelu",
        embed_dim=768,
    ):
        super().__init__()
        self.normalize = normalize
        self.non_linearities = {
            'sigmoid': torch.sigmoid,
            'exp': torch.exp,
            'softplus': nn.Softplus(beta=3),
            'gelu': nn.GELU()
        }
        self.non_linearity = non_linearity
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, 1)
        with torch.no_grad():
            self.fc2.weight.fill_(0)
            self.fc2.bias.fill_(0)

    def forward(self, x, attention_mask=None):
        """Per-token weights from hidden states.

        Args:
            x: ``(B, T, D)`` hidden states.
            attention_mask: ``(B, T)`` valid-token mask.

        With ``normalize=True`` the numerator is clamped to ``1e-6`` so the
        weights never become NaN when they all collapse to zero.
        """
        if attention_mask is None:
            attention_mask = torch.ones(x.shape[:2], device=x.device)
        # The weighting parameters stay FP32 even when the base model emits BF16.
        x = x.to(dtype=self.fc1.weight.dtype)
        nl = self.non_linearities[self.non_linearity]
        gelu = nn.GELU()
        x = gelu(self.fc1(x))
        x = self.fc2(x).squeeze(-1)
        x = nl(x)
        x = x * attention_mask
        if self.normalize:
            num = x.sum(1, keepdim=True).clamp_min(1e-6)
            x = x / num * attention_mask.sum(1, keepdim=True)
        return x


def build_weight_model(config, embed_dim) -> Optional[nn.Module]:
    """Build the configured weighting model, or ``None`` when weighting is off."""
    if not bool(getattr(config, "weighted_loss", False)):
        return None
    name = str(getattr(config, "weighting_model", None) or "linear").lower()
    if name != "linear":
        raise ValueError(f"Unknown weighting_model: {name}. Available: ['linear']")
    return LinearWeightingModel(
        normalize=bool(getattr(config, "weight_normalize", False)),
        non_linearity=str(getattr(config, "weight_non_linearity", "gelu")),
        embed_dim=embed_dim,
    )
