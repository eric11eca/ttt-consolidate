"""Entry point: benchmark evaluation with test-time training.

Loads a training checkpoint, adapts it on each test example's context, generates
the answers with the adapted weights (Hugging Face ``generate``), and scores them.

    python run_eval.py checkpoint=<run_name>/latest-step-step=00001650.ckpt \
        eval_subsets=all eval_metrics=pass@1

    # generate on GPUs, score anywhere
    python run_eval.py stages=generate ...
    python run_eval.py stages=score run_name=<previous run name> ...
"""
from __future__ import annotations

import logging
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from opt_math.eval_config import build_eval_config
from opt_math.eval_pipeline import run_eval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def seed_everything(seed: int) -> None:
    """Seed the process RNGs, as ``train.py`` does, so eval-time sampling
    (e.g. ``max_eval_data`` subsetting) is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(args: DictConfig) -> None:
    config = build_eval_config(args)
    seed_everything(config.seed)
    run_eval(config)


if __name__ == "__main__":
    main()
