"""Data reader for the open-parallel-think math release.

Training reads the ``open_parallel_think_math`` release (and, optionally, a
chunked OpenThoughts release). Evaluation reads the benchmark problems from
three quality tiers (``high`` / ``middle`` / ``low``), each a separate dataset
repository that holds the same eleven benchmark sources.

Every dataset repository is a config key (``train_repo``, ``openthoughts_repo``,
``eval_repo_prefix``); a key may be a Hugging Face repo id or a local directory.
A local directory holds one ``<name>.jsonl`` / ``<name>.json`` / ``<name>.parquet``
file per config name (``train`` for the training release, the source name for
an evaluation tier, the OpenThoughts config name for OpenThoughts).

Several string literals below (``"open_parallel_think_cot"``, ``"short"``,
``"openthoughts"``, ``"math_ai_bench_cot"`` ...) are hash input to
``_derive_seed``, not labels. Changing one silently reseeds the sampling and
draws a different subset of the data, so leave them as they are.
"""
import hashlib
import logging
import os
import random

from datasets import concatenate_datasets, load_dataset

from perk.datasets.answer_cleanup import remove_boxed_answer_sentences

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_REPO = "<org>/open-parallel-think-math"
DEFAULT_OPENTHOUGHTS_REPO = "<org>/openthoughts-114k-chunked"
DEFAULT_EVAL_REPO_PREFIX = "<org>/math-eval-sources"

# Training mixture defaults; `train.yaml` overrides them (`epoch_sample_fractions`,
# `openthoughts_config`, `openthoughts_fraction`).
DEFAULT_EPOCH_SAMPLE_FRACTIONS = (1.0, 0.7, 0.5)
DEFAULT_OPENTHOUGHTS_CONFIG = "4096"
DEFAULT_OPENTHOUGHTS_FRACTION = 0.1

#: The benchmark sources an evaluation tier repository holds.
EVAL_SOURCES = (
    "aime25",
    "aime26",
    "apex_2025",
    "arxivmath",
    "cmimc_2025",
    "gpqa_diamond",
    "hmmt_feb_2026",
    "hmmt_nov_2025",
    "imobench",
    "olympiadbench",
    "theoremqa",
)
#: Quality tiers of the evaluation repositories, in read order.
EVAL_TIERS = ("high", "middle", "low")

_LOCAL_BUILDERS = ((".jsonl", "json"), (".json", "json"), (".parquet", "parquet"))


def _check_fraction(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be a number in (0, 1], got {value!r}")
    return float(value)


def _load_split(repo, name, revision=None):
    """Load the ``train`` split of ``repo`` (hub id or local directory).

    ``name`` is the dataset configuration (``None`` for a repo without one).
    """
    local = os.path.expanduser(str(repo))
    if os.path.isdir(local):
        stem = name or "train"
        for extension, builder in _LOCAL_BUILDERS:
            candidate = os.path.join(local, stem + extension)
            if os.path.isfile(candidate):
                return load_dataset(builder, data_files=candidate, split="train")
        raise FileNotFoundError(
            f"No {stem}.jsonl / {stem}.json / {stem}.parquet in local dataset directory {local}"
        )
    if name is None:
        return load_dataset(repo, split="train", revision=revision)
    return load_dataset(repo, name, split="train", revision=revision)


class OpenParallelThinkDataReader:
    """Loads training rows and evaluation rows as ``{guid, context, queries, task}``.

    ``context`` is the list of reasoning attempts; ``queries`` holds one
    ``{question, answer, gold_answer, support}`` dict.
    """

    #: Sources whose generated trajectories lose the sentence holding the
    #: boxed answer when ``strip_generated_boxed_answers`` is on.
    GENERATED_SOLUTION_CLEANUP_SOURCES = {
        "aime24",
        "amc23",
        "math_500",
        "minerva_math",
        "gpqa",
    }
    SELECTED_ORIGINAL_SOLUTION_COLUMN = "selected_original_solution"
    ORIGINAL_SOLUTION_CANDIDATE_COLUMNS = (
        "original_solutions",
        "original_solution",
    )

    @classmethod
    def format_query(cls, problem: dict, *, prefix: str) -> tuple[str, str]:
        """Build the ``(prompt, response)`` pair for the question of one ``problem``."""
        prompt = f"{prefix}\n\nQuestion: {problem['question']}\nAnswer:"
        response = problem.get("_target_response", f"{problem['answer']}")
        return prompt, response

    # ------------------------------------------------------------------
    # Row readers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_thinking(idx, instance, args):
        """One OpenThoughts row: the question with its chunked reasoning."""
        guid = instance["id"]
        problems = [{
            "question": instance["question"],
            "answer": instance["answer"],
            "gold_answer": instance["answer"],
            "support": ""
        }]

        base_context = remove_boxed_answer_sentences(list(instance["thinking"]))
        if args.conditioned:
            context = [
                f"{instance['question']} To answer this question, I need to closely study the following contents\n\n{doc}"
                for doc in base_context
            ]
        else:
            context = base_context
        seed = getattr(args, "seed", 42)
        shuffle_seed = OpenParallelThinkDataReader._derive_seed(
            seed, "openthoughts", "shuffle", guid
        )
        random.Random(shuffle_seed).shuffle(context)

        return {"guid": guid, "queries": problems, "context": context, "task": "openthoughts"}

    @staticmethod
    def _shuffled_order(n, shuffle_seed):
        order = list(range(n))
        random.Random(shuffle_seed).shuffle(order)
        return order

    @staticmethod
    def _read_cot(idx, instance, args):
        """One row of the open-parallel-think math release."""
        guid = instance["guid"]
        problem = dict(instance["queries"][0])
        base_context = list(instance["context"])

        problem["gold_answer"] = problem.get("gold_answer", problem.get("answer", ""))
        problem["answer"] = instance["original_solution"]

        if "source" in instance:
            task = instance["source"]
        else:
            task = guid.split("_")[0]

        if args.conditioned:
            context = [
                f"{problem['question']} To answer this question, I need to closely study the following contents\n\n{doc}"
                for doc in base_context
            ]
        else:
            context = base_context
        seed = getattr(args, "seed", 42)
        shuffle_seed = OpenParallelThinkDataReader._derive_seed(
            seed, "cot", "shuffle", guid
        )
        order = OpenParallelThinkDataReader._shuffled_order(len(context), shuffle_seed)
        context = [context[i] for i in order]

        return {
            "guid": guid,
            "queries": [problem],
            "context": context,
            "task": f"parallel_cot_{task}",
        }

    @staticmethod
    def _read_cot_with_prefix_and_epoch(idx, instance, args, task_prefix, epoch_idx):
        parsed = OpenParallelThinkDataReader._read_cot(idx, instance, args)
        base_task = parsed["task"]
        if base_task.startswith("parallel_cot_"):
            base_task = base_task[len("parallel_cot_"):]
        parsed["task"] = f"{task_prefix}{base_task}_epoch{epoch_idx}"
        return parsed

    @staticmethod
    def _read_math_ai_bench_cot(idx, instance, args):
        """One evaluation row: a benchmark problem with its sampled trajectories."""
        source = instance.get("source")
        sample_idx = instance.get("index")
        if instance.get("guid") is not None:
            guid = instance["guid"]
        elif source is not None and sample_idx is not None:
            guid = f"{source}_{sample_idx}"
        elif sample_idx is not None:
            guid = f"math_ai_bench_{sample_idx}"
        else:
            guid = "math_ai_bench_unknown"
        question = instance["problem"]
        original_solution = OpenParallelThinkDataReader._selected_original_solution(
            instance
        )
        problems = [{
            "question": question,
            "gold_answer": instance["answer"],
            "answer": original_solution,
            "support": ""
        }]

        base_context = list(instance.get("generated_solutions") or [])
        if (
            OpenParallelThinkDataReader._truthy_flag(
                getattr(args, "strip_generated_boxed_answers", True)
            )
            and source in OpenParallelThinkDataReader.GENERATED_SOLUTION_CLEANUP_SOURCES
        ):
            base_context = remove_boxed_answer_sentences(base_context)
        if args.conditioned:
            context = [
                f"{question} To answer this question, I need to closely study the following contents\n\n{doc}"
                for doc in base_context
            ]
        else:
            context = base_context
        seed = getattr(args, "seed", 42)
        shuffle_seed = OpenParallelThinkDataReader._derive_seed(
            seed, "math_ai_bench_cot", "shuffle", guid
        )
        order = OpenParallelThinkDataReader._shuffled_order(len(context), shuffle_seed)
        context = [context[i] for i in order]

        return {
            "guid": guid,
            "queries": problems,
            "context": context,
            "task": source if source is not None else "math_ai_bench",
        }

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    @classmethod
    def _derive_seed(cls, base_seed, *parts):
        seed_payload = "|".join([str(base_seed)] + [str(part) for part in parts])
        digest = hashlib.sha256(seed_payload.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    @classmethod
    def _materialize_transformed_dataset(cls, data_rows, read_fn, args):
        if hasattr(data_rows, "map") and hasattr(data_rows, "column_names"):
            return data_rows.map(
                lambda instance: read_fn(0, instance, args),
                remove_columns=data_rows.column_names,
                # Avoid filesystem cache writes that can fail on shared mounts.
                load_from_cache_file=False,
                keep_in_memory=True
            )
        return [read_fn(0, instance, args) for instance in data_rows]

    @classmethod
    def _sample_fraction(cls, data_rows, fraction, seed):
        if fraction >= 1.0:
            return data_rows
        if fraction <= 0.0:
            return data_rows.select([]) if hasattr(data_rows, "select") else []
        total = len(data_rows)
        if total == 0:
            return data_rows.select([]) if hasattr(data_rows, "select") else []
        sample_size = max(1, int(total * fraction))
        if sample_size >= total:
            return data_rows
        rng = random.Random(seed)
        indices = rng.sample(range(total), sample_size)
        if hasattr(data_rows, "select"):
            return data_rows.select(indices)
        return [data_rows[i] for i in indices]

    @staticmethod
    def _truthy_flag(value):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    # ------------------------------------------------------------------
    # Evaluation-row normalization
    # ------------------------------------------------------------------

    @classmethod
    def _normalize_source_features(cls, data_rows):
        """Give every source the same ``original_solution`` column type.

        Some sources carry ``original_solution`` as a list of candidate
        solutions plus an ``original_solution_thinking`` column, others as a
        single string. ``concatenate_datasets`` refuses a set that disagrees,
        so the string form is widened to a one-element list. This is
        record-neutral: ``_original_solution_candidates`` accepts either shape
        and yields the same candidates. ``original_solution_thinking`` is read
        nowhere; it only breaks the alignment, so it is dropped.
        """
        column_names = getattr(data_rows, "column_names", None)
        if column_names is None or not hasattr(data_rows, "map"):
            return data_rows

        if "original_solution_thinking" in column_names:
            data_rows = data_rows.remove_columns("original_solution_thinking")

        if "original_solution" not in data_rows.column_names:
            return data_rows
        if len(data_rows) == 0:
            return data_rows
        if isinstance(data_rows[0].get("original_solution"), (list, tuple)):
            return data_rows

        def _widen(instance):
            value = instance.get("original_solution")
            return {"_original_solution_list": [value] if isinstance(value, str) else []}

        # Replace rather than overwrite: writing a list into a column Arrow
        # already types as a string would fail.
        return data_rows.map(
            _widen,
            remove_columns=["original_solution"],
            load_from_cache_file=False,
            keep_in_memory=True,
        ).rename_column("_original_solution_list", "original_solution")

    @classmethod
    def _original_solution_candidates(cls, instance):
        candidates = []
        seen = set()
        for key in cls.ORIGINAL_SOLUTION_CANDIDATE_COLUMNS:
            value = instance.get(key)
            if isinstance(value, str):
                raw_candidates = [value]
            elif isinstance(value, (list, tuple)):
                raw_candidates = value
            else:
                continue
            for candidate in raw_candidates:
                if not isinstance(candidate, str) or candidate.strip() == "":
                    continue
                if candidate in seen:
                    continue
                candidates.append(candidate)
                seen.add(candidate)
        return candidates

    @classmethod
    def _selected_original_solution(cls, instance):
        selected = instance.get(cls.SELECTED_ORIGINAL_SOLUTION_COLUMN)
        if isinstance(selected, str) and selected.strip() != "":
            return selected
        candidates = cls._original_solution_candidates(instance)
        if candidates:
            return candidates[0]
        return ""

    @classmethod
    def _has_nonempty_original_solution(cls, instance):
        return cls._selected_original_solution(instance).strip() != ""

    @classmethod
    def _filter_nonempty_original_solution(cls, data_rows):
        if hasattr(data_rows, "column_names") and not any(
            column_name in data_rows.column_names
            for column_name in (
                cls.SELECTED_ORIGINAL_SOLUTION_COLUMN,
                *cls.ORIGINAL_SOLUTION_CANDIDATE_COLUMNS,
            )
        ):
            return data_rows.select([]) if hasattr(data_rows, "select") else []
        if hasattr(data_rows, "select") and hasattr(data_rows, "__iter__"):
            indices = [
                idx
                for idx, instance in enumerate(data_rows)
                if cls._has_nonempty_original_solution(instance)
            ]
            return data_rows.select(indices)
        return [
            instance
            for instance in data_rows
            if cls._has_nonempty_original_solution(instance)
        ]

    @classmethod
    def _trim_math_ai_bench_context_by_tokens(cls, data_rows, max_tokens=16384):
        """Drop the longest trajectories of a row until it fits ``max_tokens``."""
        if not hasattr(data_rows, "map"):
            return data_rows
        column_names = getattr(data_rows, "column_names", [])
        metadata_key = None
        if "context_metadata" in column_names:
            metadata_key = "context_metadata"
        elif "context _metadata" in column_names:
            metadata_key = "context _metadata"
        if metadata_key is None or "generated_solutions" not in column_names:
            return data_rows

        def _trim_longest_context_first(instance):
            context = instance.get("generated_solutions")
            if not isinstance(context, list):
                return instance
            context_metadata = instance.get(metadata_key)
            if not isinstance(context_metadata, dict):
                return instance
            cot_num_token = context_metadata.get("cot_num_token")
            if not isinstance(cot_num_token, list) or len(cot_num_token) != len(context):
                return instance
            try:
                token_counts = [int(token_count) for token_count in cot_num_token]
            except (TypeError, ValueError):
                return instance

            total_num_token = context_metadata.get("total_num_token")
            if total_num_token is None:
                total_num_token = sum(token_counts)
            try:
                total_num_token = int(total_num_token)
            except (TypeError, ValueError):
                total_num_token = sum(token_counts)
            if total_num_token <= max_tokens:
                return instance

            sorted_indices = sorted(
                range(len(token_counts)),
                key=lambda idx: token_counts[idx],
                reverse=True
            )
            drop_indices = set()
            running_total = total_num_token
            for idx in sorted_indices:
                if running_total <= max_tokens:
                    break
                drop_indices.add(idx)
                running_total -= token_counts[idx]

            keep_indices = [idx for idx in range(len(context)) if idx not in drop_indices]
            trimmed_context = [context[idx] for idx in keep_indices]
            trimmed_token_counts = [token_counts[idx] for idx in keep_indices]

            updated_metadata = dict(context_metadata)
            updated_metadata["cot_num_token"] = trimmed_token_counts
            updated_metadata["total_num_token"] = running_total
            return {
                "generated_solutions": trimmed_context,
                metadata_key: updated_metadata
            }

        return data_rows.map(_trim_longest_context_first)

    @classmethod
    def _sample_generated_solutions_per_row(cls, data_rows, args, source):
        """Keep ``num_thinking`` randomly chosen trajectories of every row."""
        num_thinking = getattr(args, "num_thinking", None)
        if num_thinking is None:
            return data_rows
        try:
            num_thinking = int(num_thinking)
        except (TypeError, ValueError):
            return data_rows
        if num_thinking < 0:
            return data_rows
        if not hasattr(data_rows, "map"):
            return data_rows

        column_names = getattr(data_rows, "column_names", [])
        if "generated_solutions" not in column_names:
            return data_rows

        metadata_key = None
        if "context_metadata" in column_names:
            metadata_key = "context_metadata"
        elif "context _metadata" in column_names:
            metadata_key = "context _metadata"

        base_seed = getattr(args, "seed", 42)

        def _sample_generated_solutions(instance):
            generated_solutions = instance.get("generated_solutions")
            if not isinstance(generated_solutions, list):
                return instance
            if len(generated_solutions) <= num_thinking:
                return instance

            guid = instance.get("guid")
            sample_idx = instance.get("index")
            row_seed = cls._derive_seed(
                base_seed,
                "math_ai_bench_cot",
                source,
                "generated_solutions",
                guid,
                sample_idx
            )
            selected_indices = sorted(
                random.Random(row_seed).sample(range(len(generated_solutions)), num_thinking)
            )
            sampled_generated_solutions = [generated_solutions[idx] for idx in selected_indices]

            updated_instance = {
                "generated_solutions": sampled_generated_solutions
            }

            if metadata_key is not None:
                context_metadata = instance.get(metadata_key)
                if isinstance(context_metadata, dict):
                    updated_metadata = dict(context_metadata)
                    cot_num_token = context_metadata.get("cot_num_token")
                    if isinstance(cot_num_token, list) and len(cot_num_token) == len(generated_solutions):
                        sampled_cot_num_token = [cot_num_token[idx] for idx in selected_indices]
                        updated_metadata["cot_num_token"] = sampled_cot_num_token
                        try:
                            updated_metadata["total_num_token"] = sum(
                                int(token_count) for token_count in sampled_cot_num_token
                            )
                        except (TypeError, ValueError):
                            pass
                    updated_instance[metadata_key] = updated_metadata

            return updated_instance

        return data_rows.map(_sample_generated_solutions)

    # ------------------------------------------------------------------
    # Dataset loaders
    # ------------------------------------------------------------------

    @classmethod
    def _eval_repos(cls, args):
        prefix = str(getattr(args, "eval_repo_prefix", None) or DEFAULT_EVAL_REPO_PREFIX)
        return [f"{prefix}-{tier}" for tier in EVAL_TIERS]

    @classmethod
    def _load_eval_sources(cls, args, split):
        """Evaluation rows: ``val`` samples a few rows per source and tier; any
        other split reads the one source named by ``dataset_subset``, keeping
        the rows whose ``model`` column equals ``source_model``."""
        seed = getattr(args, "seed", 42)
        revision = getattr(args, "eval_revision", None)

        if split == "val":
            sources = list(EVAL_SOURCES)
        else:
            source = str(args.dataset_subset)
            if source not in EVAL_SOURCES:
                raise KeyError(
                    f"Unknown evaluation source {source!r}; expected one of {list(EVAL_SOURCES)}"
                )
            sources = [source]

        val_per_source_per_bucket = 20

        source_datasets = []
        for dataset_name in cls._eval_repos(args):
            for source in sources:
                data_rows = _load_split(dataset_name, source, revision)
                data_rows = cls._normalize_source_features(data_rows)
                data_rows = cls._filter_nonempty_original_solution(data_rows)
                data_rows = cls._trim_math_ai_bench_context_by_tokens(
                    data_rows,
                    max_tokens=16384
                )
                data_rows = cls._sample_generated_solutions_per_row(
                    data_rows,
                    args,
                    source
                )
                if split == "val":
                    val_seed = cls._derive_seed(
                        seed, "math_ai_bench_sources_cot", "val", dataset_name, source
                    )
                    total = len(data_rows)
                    if total > val_per_source_per_bucket:
                        indices = random.Random(val_seed).sample(
                            range(total), val_per_source_per_bucket
                        )
                        data_rows = data_rows.select(sorted(indices))
                else:
                    source_model = getattr(args, "source_model", None)
                    if source_model is not None and "model" in getattr(data_rows, "column_names", []):
                        selected_indices = [
                            idx for idx, model_name in enumerate(data_rows["model"])
                            if model_name == source_model
                        ]
                        data_rows = data_rows.select(selected_indices)
                source_datasets.append(data_rows)
        dataset = concatenate_datasets(source_datasets)
        return cls._materialize_transformed_dataset(
            dataset,
            lambda idx, instance, local_args: cls._read_math_ai_bench_cot(
                idx, instance, local_args
            ),
            args
        )

    @classmethod
    def _load_open_parallel_think(cls, args):
        """Sample the open-parallel-think math release once per training epoch.

        The release holds the short and long trajectory sets together, so a
        single ``_sample_fraction`` per epoch draws from both in the proportion
        they are published in. The epochs are concatenated.

        The two string literals passed to ``_derive_seed`` are hash input, not
        labels (see the module docstring). ``"parallel_cot_"`` is stamped into
        every record's ``task``.
        """
        seed = getattr(args, "seed", 42)
        repo = str(getattr(args, "train_repo", None) or DEFAULT_TRAIN_REPO)
        data_rows = _load_split(repo, None, getattr(args, "train_revision", None))

        epoch_datasets = []
        epoch_fractions = getattr(args, "epoch_sample_fractions", None)
        if epoch_fractions is None:
            epoch_fractions = DEFAULT_EPOCH_SAMPLE_FRACTIONS
        elif len(epoch_fractions) == 0:
            raise ValueError(
                "epoch_sample_fractions is empty; give one fraction per epoch, e.g. [1.0, 0.7, 0.5]"
            )
        for epoch_idx, sample_fraction in enumerate(epoch_fractions):
            sample_fraction = _check_fraction(sample_fraction, f"epoch_sample_fractions[{epoch_idx}]")
            epoch_seed = cls._derive_seed(
                seed, "open_parallel_think_cot", "short", epoch_idx
            )
            sampled_rows = cls._sample_fraction(data_rows, sample_fraction, epoch_seed)
            epoch_datasets.append(
                cls._materialize_transformed_dataset(
                    sampled_rows,
                    lambda idx, instance, local_args, _epoch_idx=epoch_idx: (
                        cls._read_cot_with_prefix_and_epoch(
                            idx, instance, local_args, "parallel_cot_", _epoch_idx
                        )
                    ),
                    args
                )
            )
        return concatenate_datasets(epoch_datasets)

    @classmethod
    def _load_openthoughts(cls, args):
        seed = getattr(args, "seed", 42)
        config = str(getattr(args, "openthoughts_config", None) or DEFAULT_OPENTHOUGHTS_CONFIG)
        fraction = _check_fraction(
            getattr(args, "openthoughts_fraction", DEFAULT_OPENTHOUGHTS_FRACTION),
            "openthoughts_fraction",
        )
        repo = str(getattr(args, "openthoughts_repo", None) or DEFAULT_OPENTHOUGHTS_REPO)
        data_rows = _load_split(repo, config, getattr(args, "openthoughts_revision", None))
        data_rows = cls._sample_fraction(
            data_rows,
            fraction,
            cls._derive_seed(seed, "openthoughts", "4096", "sample")
        )
        return cls._materialize_transformed_dataset(
            data_rows,
            cls._read_thinking,
            args
        )

    @classmethod
    def jsonl_file_reader(cls, args, split="train"):
        """Rows of ``split``: ``train``, ``val`` (a few rows per source) or ``test``."""
        if split == "train":
            dataset = cls._load_open_parallel_think(args)
            if cls._truthy_flag(getattr(args, "use_openthoughts", False)):
                dataset = concatenate_datasets([dataset, cls._load_openthoughts(args)])
            return dataset
        return cls._load_eval_sources(args, split)
