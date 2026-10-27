"""Entry point: meta-learning (test-time-learning) training.

    torchrun --nproc_per_node=4 train.py
    python train.py n_gpu=1 output_dir=./outputs/debug     # single GPU
"""
import logging
import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.logging import RichHandler
from rich.traceback import install

from perk.datasets.meta_dataset import DATASET_NAME
from perk.runner import run


def setup_logging(log_path):
    """Log to the console (rich) and to ``log_path``."""
    options = {"level": "INFO", "show_time": True, "show_level": True, "show_path": True,
               "markup": False, "rich_tracebacks": True, "tracebacks_show_locals": False}
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[logging.FileHandler(log_path), RichHandler(**options)],
        force=True,
    )
    logging.captureWarnings(True)


@hydra.main(version_base="1.3", config_path="./configs", config_name="train")
def main(args: DictConfig) -> None:
    install(show_locals=False)
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(os.path.join(args.output_dir, "train_log.txt" if args.do_train else "eval_log.txt"))

    logger = logging.getLogger("perk.train")
    logger.info("***** Running training/evaluation *****")
    logger.info(OmegaConf.to_container(args, resolve=True))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    run_dir = f"{args.output_dir}/{args.wandb_name}"
    with open_dict(args):
        args.run_dir = run_dir
        args.gen_dir = f"{run_dir}/outputs"
        args.eval_dir = f"{run_dir}/evals"

    # `wandb_name` stays the run-folder name above; the W&B display name carries the dataset.
    args.wandb_name = f"{DATASET_NAME}-{args.wandb_name}"

    if args.do_eval and not args.do_train:
        args.num_train_epochs = 1

    run(args)


if __name__ == "__main__":
    main()
