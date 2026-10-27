"""Builds the Lightning trainer: checkpoint callbacks, precision, distributed strategy and logging."""

import os
import logging
import lightning as pl

from pprint import pformat
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)

util_logger = logging.getLogger("perk.trainer")
util_logger.setLevel(logging.INFO)

class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description("running validation...")
        return bar

def setup_trainer(args) -> pl.Trainer:
    """Sets up the trainer and associated callbacks from the configuration.

    :param args: the run configuration
    :rtype: a trainer instance
    """
    if "loss" in args.callback_monitor:
        mode = "min"
    else:
        mode = "max"
    util_logger.info("mode=%s via %s" % (mode, args.callback_monitor))

    if not os.path.isdir(args.output_dir):
        util_logger.info("making target directory: %s" % args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_callback_val = ModelCheckpoint(
        dirpath=args.run_dir,
        filename="val-{epoch:02d}-{step:08d}-{" + args.callback_monitor + ":.4f}",
        monitor=args.callback_monitor,
        mode=mode,
        save_top_k=5,
        verbose=True,
        auto_insert_metric_name=True,
    )

    checkpoint_callback_latest = ModelCheckpoint(
        dirpath=args.run_dir,
        filename="latest-step-{step:08d}",
        every_n_train_steps=getattr(args, "latest_checkpoint_every_n_train_steps", 50),
        save_top_k=1,
        verbose=True,
        save_last=True,
        save_on_exception=True,
    )

    # keep_last > 0 keeps only the newest N archives (ranked by step); -1 keeps all.
    archive_keep = int(getattr(args, "archive_checkpoint_keep_last", -1))
    checkpoint_callback_archive = ModelCheckpoint(
        dirpath=args.run_dir,
        filename="archive-step-{step:08d}",
        every_n_train_steps=getattr(args, "archive_checkpoint_every_n_train_steps", 500),
        save_top_k=archive_keep,
        monitor="step" if archive_keep > 0 else None,
        mode="max",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    progress_bar = LitProgressBar()

    callbacks = [lr_monitor, progress_bar]

    if args.do_train:
        if args.do_eval:
            callbacks.append(checkpoint_callback_val)
        callbacks.append(checkpoint_callback_latest)
        callbacks.append(checkpoint_callback_archive)

    train_params = dict(
        accelerator=str(getattr(args, "accelerator", "gpu")),
        devices=args.n_gpu,
        max_epochs=args.num_train_epochs,
        callbacks=callbacks,
        log_every_n_steps=1,
        val_check_interval=args.val_check_interval,
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=0,
        inference_mode=False,
    )
    # Wall-clock stop for the training loop (Lightning `max_time`, a
    # "DD:HH:MM:SS" string). Unset (None) keeps the default behavior.
    max_time = getattr(args, "max_time", None)
    if max_time is not None:
        train_params["max_time"] = str(max_time)
    max_steps = getattr(args, "max_steps", None)
    if max_steps is not None:
        max_steps = int(max_steps)
        if max_steps > 0:
            train_params["max_steps"] = max_steps

    if args.val_check_interval == 0.0:
        train_params["check_val_every_n_epoch"] = 1
        del train_params["val_check_interval"]

    # A train-only job skips validation.
    if args.do_train and not args.do_eval:
        train_params["limit_val_batches"] = 0

    if args.do_train and str(getattr(args, "outer_loss_type", "ce")).lower() == "ce":
        train_params["gradient_clip_val"] = args.max_grad_norm
        train_params["gradient_clip_algorithm"] = "norm"

    train_params["accumulate_grad_batches"] = args.gradient_accumulation_steps
    if args.bf16:
        train_params["precision"] = "bf16-mixed"
    else:
        train_params["precision"] = 32

    if args.n_gpu > 1:
        util_logger.info("using DDP mode for inner loop")
        train_params["strategy"] = "ddp"
        train_params["use_distributed_sampler"] = True

    if args.do_eval and not args.do_train:
        args.n_nodes = 1
        util_logger.info("using single node for evaluation")
    elif args.n_nodes > 1:
        util_logger.info(f"Multi-node training with {args.n_nodes} nodes")
        train_params["num_nodes"] = args.n_nodes
    else:
        util_logger.info("using single node for training")

    if getattr(args, "use_wandb", False) and args.wandb_project:
        from lightning.pytorch.loggers import WandbLogger

        train_params["logger"] = WandbLogger(
            project=args.wandb_project,
            entity=getattr(args, "wandb_entity", None) or None,
            name=args.wandb_name,
            log_model=False,
        )
        train_params["logger"].log_hyperparams(args)

    util_logger.info("\n===========\n" + pformat(train_params) + "\n===========")

    trainer = pl.Trainer(**train_params)

    return trainer
