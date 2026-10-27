"""Runs training and/or evaluation: builds the trainer and the meta-learning module, then fits."""

import os
import json
import math
import logging
from pathlib import Path

from perk.train import setup_trainer
from perk.modules import MetaLearningModule

util_logger = logging.getLogger("perk.runner")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _write_train_progress(trainer, args):
    """Write `<run_dir>/train_progress.json`: steps done against steps per
    epoch, so a caller can tell whether a time-limited run finished its epochs
    or should be extended by resuming."""
    if not getattr(trainer, "is_global_zero", True) or not getattr(args, "run_dir", None):
        return
    batches = getattr(trainer, "num_training_batches", None)
    accumulate = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
    steps_per_epoch = (
        math.ceil(int(batches) / accumulate)
        if isinstance(batches, (int, float)) and math.isfinite(batches) else None
    )
    epochs = int(getattr(args, "num_train_epochs", 1) or 1)
    global_step = int(trainer.global_step)
    progress = {
        "global_step": global_step,
        "steps_per_epoch": steps_per_epoch,
        "num_train_epochs": epochs,
        "max_time": getattr(args, "max_time", None),
        "finished_all_epochs": (
            None if not steps_per_epoch else global_step >= steps_per_epoch * epochs
        ),
    }
    path = Path(str(args.run_dir)) / "train_progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    util_logger.info("Train progress: %s", progress)


def run(args):
    """Train (and/or evaluate) with the meta-learning module."""
    util_logger.info("Setting up configuration for model runner...")

    if getattr(args, "use_wandb", False):
        for key, value in (("WANDB_NAME", args.wandb_name), ("WANDB_PROJECT", args.wandb_project),
                           ("WANDB_ENTITY", getattr(args, "wandb_entity", None))):
            if value:
                os.environ[key] = str(value)

    trainer = setup_trainer(args)
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = os.path.join(args.output_dir, args.checkpoint)

    if args.load_checkpoint and args.resume_from_checkpoint:
        raise ValueError("Load and resume from checkpoint are not allowed at the same time")

    if args.load_checkpoint:
        assert args.checkpoint is not None, "Checkpoint path is not provided"
        util_logger.info(f"Loading model from: {checkpoint_path}")
        model = MetaLearningModule.load_from_checkpoint(
            checkpoint_path, config=args, map_location="cpu")
    else:
        util_logger.info("Initializing from a pre-trained model")
        model = MetaLearningModule(args)

    if not args.do_train:
        # Eval-only job: nothing to train.
        util_logger.info("do_train=false: skipping trainer.fit")
    elif args.resume_from_checkpoint:
        assert args.checkpoint is not None
        util_logger.info(f"Resuming training from: {checkpoint_path}")
        trainer.fit(model, ckpt_path=checkpoint_path)
    else:
        util_logger.info("Train and Evaluation start...")
        trainer.fit(model)
    if args.do_train:
        _write_train_progress(trainer, args)

    if not args.do_train and bool(getattr(args, "do_eval", False)):
        trainer.validate(model, verbose=False)
