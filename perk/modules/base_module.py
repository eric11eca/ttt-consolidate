"""Lightning module base: data loading, scheduler, checkpointing hooks and evaluation epoch logic."""

import json
import math
import os
import time
import hashlib
import shutil
import yaml
import torch
import warnings
import lightning as pl
import torch.distributed as dist

from typing import Any, Dict
from omegaconf import OmegaConf
from transformers import get_cosine_schedule_with_warmup
from lightning.pytorch.utilities import grad_norm

from perk.evaluate import evaluate
from perk.datasets.meta_dataset import DATASET_NAME, MetaKnowledgeDataset, create_dataloader

warnings.filterwarnings("ignore")

# Non-zero ranks wait this long for rank 0 to prepare the shared dataset cache.
DATASET_LOAD_TIMEOUT_SECONDS = 7200


def _scheduler_total_steps(
    dataset_size,
    num_train_epochs,
    train_batch_size,
    grad_accum,
    num_devices,
    num_nodes,
):
    """Compute the LR-scheduler horizon (total optimizer steps) for the manual fallback."""
    total_gpus = max(1, int(num_devices) * int(num_nodes))
    effective_batch_size = (
        int(train_batch_size) * int(grad_accum) * total_gpus
    )
    return math.ceil(
        dataset_size * num_train_epochs / effective_batch_size
    )


class BaseModule(pl.LightningModule):
    def __init__(self, config, logger):
        """
        :param config: the run configuration
        :param logger: logger used for this module's messages
        """
        super().__init__()
        self.model_logger = logger
        self.hparams.update(
            OmegaConf.to_container(config))
        self.global_training_step = 0
        self.global_epoch_counter = 0
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Validate or optionally resize learned inner-LR checkpoint tensors."""
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            return

        key = "lrs.lr"
        checkpoint_lr = state_dict.get(key)
        current_lr = getattr(getattr(self, "lrs", None), "lr", None)
        if not isinstance(checkpoint_lr, torch.Tensor) or current_lr is None:
            return
        if tuple(checkpoint_lr.shape) == tuple(current_lr.shape):
            return
        if not bool(getattr(self.hparams, "allow_lrs_shape_mismatch", False)):
            raise RuntimeError(
                "Checkpoint lrs.lr shape mismatch "
                f"(checkpoint={tuple(checkpoint_lr.shape)}, "
                f"current={tuple(current_lr.shape)}). This usually means the "
                "model parameter list differs between training and eval, for "
                "example because tie_word_embeddings changed. Set "
                "tie_word_embeddings to match the training run; only set "
                "allow_lrs_shape_mismatch=true if you intentionally want to "
                "resize the learned inner-loop LR table."
            )

        resized_lr = current_lr.detach().clone()
        rows = min(int(resized_lr.shape[0]), int(checkpoint_lr.shape[0]))
        cols = min(int(resized_lr.shape[1]), int(checkpoint_lr.shape[1]))
        resized_lr[:rows, :cols] = checkpoint_lr[:rows, :cols].to(
            device=resized_lr.device,
            dtype=resized_lr.dtype,
        )
        state_dict[key] = resized_lr
        self._rewrite_lrs_extra_state(state_dict)

        self.model_logger.warning(
            "Resized checkpoint %s from shape %s to %s while loading. "
            "Copied overlapping %dx%d learned LR values and kept defaults for "
            "new positions.",
            key,
            tuple(checkpoint_lr.shape),
            tuple(resized_lr.shape),
            rows,
            cols,
        )

    def _rewrite_lrs_extra_state(self, state_dict: Dict[str, Any]) -> None:
        """After resizing `lrs.lr` to another `n_inner_iter`, make the saved
        registry state (`lrs._extra_state`: `num_inner_iter`, `registry_hash`)
        describe the resized table, or `LRParameters` rejects the load. Only
        done when the checkpoint's rows are the current rows (same names, same
        format): a registry for different rows still fails, as it should."""
        extra_key = "lrs._extra_state"
        saved = state_dict.get(extra_key)
        lrs = getattr(self, "lrs", None)
        if not isinstance(saved, dict) or not hasattr(lrs, "registry_metadata"):
            return
        current = lrs.registry_metadata()
        if saved.get("abi") != current["abi"] or list(saved.get("row_names", ())) != current["row_names"]:
            return
        state_dict[extra_key] = current

    @staticmethod
    def _runtime_global_rank() -> int:
        if dist.is_initialized():
            return int(dist.get_rank())
        for env_name in ("RANK",):
            env_value = os.environ.get(env_name)
            if env_value is not None:
                try:
                    return int(env_value)
                except ValueError:
                    pass
        return 0

    @staticmethod
    def _runtime_world_size() -> int:
        if dist.is_initialized():
            return int(dist.get_world_size())
        for env_name in ("WORLD_SIZE",):
            env_value = os.environ.get(env_name)
            if env_value is not None:
                try:
                    return int(env_value)
                except ValueError:
                    pass
        return 1

    def _dataset_cache_ready_path(self) -> str:
        cache_key = "|".join([
            DATASET_NAME,
            str(getattr(self.hparams, "dataset_subset", "")),
            str(getattr(self.hparams, "source_model", "")),
            str(getattr(self.hparams, "num_thinking", "")),
            str(getattr(self.hparams, "conditioned", "")),
            str(getattr(self.hparams, "strip_generated_boxed_answers", True)),
            str(getattr(self.hparams, "seed", "")),
            str(getattr(self.hparams, "max_eval_data", "")),
            str(getattr(self.hparams, "do_train", "")),
            str(getattr(self.hparams, "do_eval", "")),
            str(getattr(self.hparams, "dataset_cache_version", "v1")),
            *(
                str(getattr(self.hparams, name, None))
                for name in (
                    "epoch_sample_fractions",
                    "use_openthoughts",
                    "openthoughts_config",
                    "openthoughts_fraction",
                    "train_repo",
                    "train_revision",
                    "openthoughts_repo",
                    "openthoughts_revision",
                    "eval_repo_prefix",
                    "eval_revision",
                )
            ),
            "prepared-dataset-cache-v2",
            str(os.environ.get("HF_HOME", "")),
        ])
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
        output_dir = getattr(self.hparams, "output_dir", ".")
        marker_name = f"{digest}.ready"
        return os.path.join(output_dir, ".dataset_cache_ready", marker_name)

    def _prepared_dataset_cache_dir(self) -> str:
        return f"{self._dataset_cache_ready_path()}.datasets"

    def _remove_prepared_dataset_cache(self, cache_dir: str):
        if not os.path.exists(cache_dir):
            return

        stale_cache_dir = f"{cache_dir}.stale-{os.getpid()}-{time.time_ns()}"
        try:
            os.replace(cache_dir, stale_cache_dir)
        except FileNotFoundError:
            return
        except OSError:
            shutil.rmtree(cache_dir, ignore_errors=True)
            return
        shutil.rmtree(stale_cache_dir, ignore_errors=True)

    def _save_prepared_dataset_cache(self, cache_dir: str):
        tmp_cache_dir = f"{cache_dir}.tmp-{os.getpid()}-{time.time_ns()}"
        if os.path.exists(tmp_cache_dir):
            shutil.rmtree(tmp_cache_dir, ignore_errors=True)
        os.makedirs(tmp_cache_dir, exist_ok=True)

        self.train_data.save_prepared_cache(os.path.join(tmp_cache_dir, "train"))
        self.val_data.save_prepared_cache(os.path.join(tmp_cache_dir, "val"))
        self.test_data.save_prepared_cache(os.path.join(tmp_cache_dir, "test"))

        self._remove_prepared_dataset_cache(cache_dir)
        os.replace(tmp_cache_dir, cache_dir)

    def _load_prepared_dataset_cache(self, cache_dir: str):
        self.model_logger.info("Loading prepared dataset cache from %s", cache_dir)
        self.train_data = MetaKnowledgeDataset.from_prepared_cache(
            args=self.hparams,
            tokenizer=self.tokenizer,
            data_type="train",
            is_training=True,
            cache_path=os.path.join(cache_dir, "train"),
        )
        self.val_data = MetaKnowledgeDataset.from_prepared_cache(
            args=self.hparams,
            tokenizer=self.tokenizer,
            data_type="val",
            is_training=False,
            cache_path=os.path.join(cache_dir, "val"),
        )
        self.test_data = MetaKnowledgeDataset.from_prepared_cache(
            args=self.hparams,
            tokenizer=self.tokenizer,
            data_type="test",
            is_training=False,
            cache_path=os.path.join(cache_dir, "test"),
        )
        self.model_logger.info("Prepared dataset cache loaded")

    def _prepared_dataset_cache_exists(self, cache_dir: str) -> bool:
        return all(
            MetaKnowledgeDataset.prepared_cache_exists(os.path.join(cache_dir, split))
            for split in ("train", "val", "test")
        )

    def _load_dataset_impl(self):
        if getattr(self.hparams, "test_only_dataset", False):
            self.model_logger.info(
                "Loading test dataset only (test_only_dataset=True); "
                "aliasing train/val to test")
            self.test_data = MetaKnowledgeDataset(
                args=self.hparams,
                tokenizer=self.tokenizer,
                data_type="test",
                is_training=False)
            self.train_data = self.test_data
            self.val_data = self.test_data
            self.model_logger.info("Test dataset loaded; train/val aliased to test")
            return
        do_train = bool(getattr(self.hparams, "do_train", True))
        if do_train:
            self.model_logger.info("Loading train, dev, and test dataset")
            self.train_data = MetaKnowledgeDataset(
                args=self.hparams,
                tokenizer=self.tokenizer,
                data_type="train",
                is_training=True)
        else:
            self.model_logger.info("Loading dev and test dataset (do_train=False)")
        self.val_data = MetaKnowledgeDataset(
            args=self.hparams,
            tokenizer=self.tokenizer,
            data_type="val",
            is_training=False)
        self.test_data = MetaKnowledgeDataset(
            args=self.hparams,
            tokenizer=self.tokenizer,
            data_type="test",
            is_training=False)
        if not do_train:
            # An eval-only job never trains, and a subset may ship no train
            # split at all. Alias train to val, as
            # test_only_dataset does, so every train-data consumer
            # (len(), dataloader, scheduler setup) still has a dataset.
            self.train_data = self.val_data
        self.model_logger.info("Train, Dev, and Test datasets loaded")

    def load_dataset(self):
        if getattr(self.hparams, "test_only_dataset", False):
            self._load_dataset_impl()
            return

        world_size = self._runtime_world_size()
        if world_size <= 1:
            self._load_dataset_impl()
            return

        rank = self._runtime_global_rank()
        ready_path = self._dataset_cache_ready_path()
        cache_dir = self._prepared_dataset_cache_dir()
        ready_dir = os.path.dirname(ready_path)

        if rank == 0:
            import fcntl

            os.makedirs(ready_dir, exist_ok=True)
            lock_path = f"{ready_path}.prepare.lock"
            with open(lock_path, "w", encoding="utf-8") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)

                if os.path.exists(ready_path) and self._prepared_dataset_cache_exists(cache_dir):
                    self._load_prepared_dataset_cache(cache_dir)
                    return

                if os.path.exists(ready_path):
                    try:
                        os.remove(ready_path)
                    except FileNotFoundError:
                        pass
                self._remove_prepared_dataset_cache(cache_dir)

                self._load_dataset_impl()
                self._save_prepared_dataset_cache(cache_dir)
                tmp_path = f"{ready_path}.tmp-{os.getpid()}-{time.time_ns()}"
                with open(tmp_path, "w", encoding="utf-8") as fout:
                    fout.write(f"ready_at={time.time()}\n")
                    fout.write(f"rank={rank}\n")
                    fout.write(f"cache_dir={cache_dir}\n")
                os.replace(tmp_path, ready_path)
            return

        self.model_logger.info(
            "Rank %s waiting for rank 0 dataset cache warmup marker: %s",
            rank,
            ready_path,
        )
        start_time = time.time()
        while not (os.path.exists(ready_path) and self._prepared_dataset_cache_exists(cache_dir)):
            if time.time() - start_time > DATASET_LOAD_TIMEOUT_SECONDS:
                raise TimeoutError(
                    "Timed out waiting for prepared dataset cache: "
                    f"marker={ready_path}, cache_dir={cache_dir}"
                )
            time.sleep(10)
        self._load_prepared_dataset_cache(cache_dir)

    def train_dataloader(self):
        dataloader = create_dataloader(self.hparams, self.train_data, is_training=True)
        self.model_logger.info("Train dataloader length: %d" % len(dataloader))
        self._clamp_val_check_interval(len(dataloader))
        return dataloader

    def _clamp_val_check_interval(self, batches_per_epoch: int) -> None:
        """An int `val_check_interval` above the batches in one epoch makes
        Lightning raise. Check at the end of the epoch instead."""
        trainer = getattr(self, "trainer", None)
        interval = getattr(trainer, "val_check_interval", None)
        # Lightning shards the loader per rank when it injects its distributed
        # sampler, so one epoch is that many fewer batches than the loader length.
        connector = getattr(trainer, "_accelerator_connector", None)
        world_size = int(getattr(trainer, "world_size", 1) or 1)
        if getattr(connector, "use_distributed_sampler", False) and world_size > 1:
            batches_per_epoch = -(-batches_per_epoch // world_size)
        if (
            isinstance(interval, int)
            and not isinstance(interval, bool)
            and batches_per_epoch > 0
            and interval > batches_per_epoch
        ):
            self.model_logger.warning(
                "val_check_interval=%d exceeds the %d batches in one epoch; "
                "validating every %d batches", interval, batches_per_epoch, batches_per_epoch,
            )
            trainer.val_check_interval = batches_per_epoch

    def val_dataloader(self):
        dataloader = create_dataloader(self.hparams, self.val_data, is_training=False)
        self.model_logger.info("Validation dataloader length: %d" % len(dataloader))
        return dataloader

    def test_dataloader(self):
        dataloader = create_dataloader(self.hparams, self.test_data, is_training=False)
        self.model_logger.info("Test dataloader length: %d" % len(dataloader))
        return dataloader

    def on_train_epoch_end(self):
        self.global_epoch_counter += 1

    def get_lr_scheduler(self):
        """Sets up the optimizer learning rate scheduler"""
        trainer = getattr(self, "trainer", None)
        if trainer is not None and getattr(trainer, "num_devices", None) is not None:
            num_devices = max(1, int(trainer.num_devices))
            num_nodes = max(1, int(trainer.num_nodes))
        elif torch.cuda.is_available():
            num_devices = max(1, int(self.hparams.n_gpu))
            num_nodes = max(1, int(getattr(self.hparams, "n_nodes", 1)))
        else:
            num_devices = 1
            num_nodes = 1

        global_devices = max(1, num_devices * num_nodes)

        effective_batch_size = (
            self.hparams.train_batch_size
            * self.hparams.gradient_accumulation_steps
            * global_devices
        )

        total_steps = 0
        if trainer is not None:
            estimated_steps = getattr(trainer, "estimated_stepping_batches", None)
            if estimated_steps is not None:
                total_steps = int(estimated_steps)

        if total_steps <= 0:
            if hasattr(self, "train_data"):
                dataset_size = len(self.train_data)
            else:
                dataset_size = len(self.train_dataloader().dataset)
            total_steps = _scheduler_total_steps(
                dataset_size=dataset_size,
                num_train_epochs=self.hparams.num_train_epochs,
                train_batch_size=self.hparams.train_batch_size,
                grad_accum=self.hparams.gradient_accumulation_steps,
                num_devices=num_devices,
                num_nodes=num_nodes,
            )

        self.hparams.warmup_steps = min(
            int(total_steps * self.hparams.warmup_proportion),
            8000,
        )

        self.model_logger.info(
            "total_steps computed for scheduler: %s, warmup step: %s, "
            "global_devices: %s, effective_batch_size: %s"
            % (
                total_steps,
                str(self.hparams.warmup_steps),
                global_devices,
                effective_batch_size,
            )
        )

        scheduler = get_cosine_schedule_with_warmup(
            self.opt,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def on_before_optimizer_step(self, optimizer):
        # Canonical outer-gradient norm: every optimizer-owned parameter with a
        # gradient, including the learned inner-LR table and optional weighting
        # model.  This is measured before clipping.
        raw_optimizer = getattr(optimizer, "optimizer", optimizer)
        seen = set()
        squared = None
        for group in raw_optimizer.param_groups:
            for parameter in group["params"]:
                if id(parameter) in seen or parameter.grad is None:
                    continue
                seen.add(id(parameter))
                term = parameter.grad.detach().float().pow(2).sum()
                squared = term if squared is None else squared + term
        if squared is None:
            outer_grad_norm = torch.tensor(0.0, device=self.device)
        else:
            outer_grad_norm = squared.sqrt()
        self._last_outer_grad_norm = outer_grad_norm.detach()
        self.log("outer_grad_norm", self._last_outer_grad_norm)

        norms = grad_norm(self.model, norm_type=2)
        if 'grad_2.0_norm_total' in norms:
            self.log('grad_norm', norms['grad_2.0_norm_total'])
        if 'grad_2.0_norm_weight_model.fc2.weight' in norms:
            self.log('grad_norm_fc2', norms['grad_2.0_norm_weight_model.fc2.weight'])
        if 'grad_2.0_norm_weight_model.fc1.weight' in norms:
            self.log('grad_norm_fc1', norms['grad_2.0_norm_weight_model.fc1.weight'])

    def validation_epoch_callback(self, outputs):
        valid_losses = [
            x["outer_loss"].detach().cpu()
            for x in outputs
            if not x.get("skip_loss", False)
        ]
        if len(valid_losses) > 0:
            val_loss = torch.stack(valid_losses).mean()
        else:
            val_loss = torch.tensor(0.0)
        print_out_flatten = []
        for out in outputs:
            print_out_flatten += out["records"]
        return val_loss, print_out_flatten

    def _eval_epoch_end_logic(self, eval_step_outputs, prefix="val"):
        """Shared logic for validation and test epoch end.

        Args:
            eval_step_outputs: List of outputs from validation/test steps
            prefix: Prefix for logging metrics (e.g., "val" or "test")
        """
        pipeline_scores_outputs = bool(
            getattr(self.hparams, "skip_builtin_validation_metrics", False)
            and getattr(self.hparams, "eval_resume", False)
        )
        if pipeline_scores_outputs:
            outputs = eval_step_outputs
            if prefix == "val":
                self.last_validation_step_outputs = list(outputs)
            elif prefix == "test":
                self.last_test_step_outputs = list(outputs)

            valid_losses = [
                x["outer_loss"].detach().cpu()
                for x in outputs
                if not x.get("skip_loss", False)
            ]
            loss = torch.stack(valid_losses).mean() if valid_losses else torch.tensor(0.0)
            self.log(
                f"{prefix}_loss",
                loss.to(self.device),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=False,
            )
            self.model_logger.info(
                "Skipping built-in %s metrics; the evaluation pipeline scores the generations.",
                prefix,
            )
            eval_step_outputs.clear()
            return

        if self.trainer.num_devices > 1:
            outputs = [None for _ in range(dist.get_world_size())]
            try:
                dist.all_gather_object(outputs, eval_step_outputs)
                outputs = [item for sublist in outputs for item in sublist]
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                rank = dist.get_rank() if dist.is_initialized() else 0
                self.model_logger.warning(
                    "Rank %s - all_gather_object failed during %s epoch end: %s. "
                    "Falling back to local outputs.",
                    rank,
                    prefix,
                    str(exc).splitlines()[0],
                )
                outputs = eval_step_outputs
                torch.cuda.empty_cache()
        else:
            outputs = eval_step_outputs

        if len(outputs) == 0:
            self.log(f"{prefix}_acc", 0.50, on_epoch=True, prog_bar=False)
            return

        if prefix == "val":
            self.last_validation_step_outputs = list(outputs)
        elif prefix == "test":
            self.last_test_step_outputs = list(outputs)

        loss, generations = self.validation_epoch_callback(outputs)

        self.log(
            f"{prefix}_loss", loss.to(self.device),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True)

        metrics = {}
        if bool(getattr(self.hparams, "skip_builtin_validation_metrics", False)):
            self.model_logger.info(
                "Skipping built-in %s metrics; the evaluation pipeline scores the generations.",
                prefix,
            )
            eval_step_outputs.clear()
            del generations
            return

        metrics = self.validation_eval(generations)
        for key in metrics:
            metrics[key] = round(metrics[key], 4)

        for key, val in metrics.items():
            self.log(
                f"{prefix}_{key}",
                torch.tensor(val).to(self.device),
                on_epoch=True,
                prog_bar=False,
                sync_dist=True)

        eval_step_outputs.clear()
        del generations

    def on_validation_epoch_end(self):
        """Called at the end of the validation epoch"""
        self._eval_epoch_end_logic(self.validation_step_outputs, prefix="val")

    def test_step(self, batch, batch_idx) -> Dict:
        test_out = self.validation_step(batch, batch_idx)
        self.test_step_outputs.append(test_out)
        return test_out

    def on_test_epoch_end(self):
        """Called at the end of the test epoch"""
        self._eval_epoch_end_logic(self.test_step_outputs, prefix="test")

    def validation_eval(self, results):
        metrics, generations = evaluate(results)
        if not (torch.distributed.is_initialized() and torch.distributed.get_rank() != 0):
            os.makedirs(self.hparams.run_dir, exist_ok=True)
            os.makedirs(self.hparams.gen_dir, exist_ok=True)
            os.makedirs(self.hparams.eval_dir, exist_ok=True)

            config_name = os.path.join(self.hparams.run_dir, "config.yaml")
            if not os.path.exists(config_name):
                with open(config_name, "w") as f:
                    yaml.dump(self.hparams, f)
            step = self.global_training_step
            answer_pairs = [generation.answer_pair for generation in generations]
            with open(os.path.join(self.hparams.gen_dir, f"outputs_{step}.jsonl"), "w") as f:
                f.writelines(json.dumps(row) + "\n" for row in results)
            if answer_pairs:
                with open(os.path.join(self.hparams.gen_dir, f"answer_pairs_{step}.jsonl"), "w") as f:
                    f.writelines(json.dumps(row) + "\n" for row in answer_pairs)
            with open(os.path.join(self.hparams.eval_dir, f"eval_{step}.jsonl"), "w") as f:
                json.dump(metrics, f, indent=2)
        return metrics
