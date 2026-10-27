"""Dataset that turns reader rows into tokenized context and query tensors."""

import os
import json
import pickle
import torch
import logging
import random

from pprint import pprint
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
)

from opt_math.data_reader import OpenParallelThinkDataReader

# Prefix of every task name and of the W&B run name.
DATASET_NAME = "open_parallel_think"

# Inserted between a query prompt and its completion when no chat template is
# used, so the prompt's last token cannot merge with the completion's first
# token and tokenize differently from the generation prompt.
PROMPT_COMPLETION_SEPARATOR = "\n"


def uses_chat_template(tokenizer) -> bool:
    """Only Qwen3 models use the chat template; all others use
    ``prompt + PROMPT_COMPLETION_SEPARATOR + completion``."""
    return "Qwen3" in (getattr(tokenizer, "name_or_path", None) or "")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger("perk.dataset")


class MetaKnowledgeDataset(Dataset):
    def __init__(self, args, tokenizer, data_type, is_training):
        self.args = args
        self.tokenizer = tokenizer

        if args.dataset_subset != "none":
            self.task = f"{DATASET_NAME}/{args.dataset_subset}"
        else:
            self.task = DATASET_NAME

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.data_type = data_type

        self.is_training = is_training
        self.tokenizer = tokenizer

        self.data_rows = self.read_data_from_file()

        if not self.is_training and args.max_eval_data > 0:
            num_samples = min(args.max_eval_data, len(self.data_rows))
            if num_samples < len(self.data_rows):
                if hasattr(self.data_rows, "select"):
                    indices = random.sample(range(len(self.data_rows)), num_samples)
                    self.data_rows = self.data_rows.select(indices)
                else:
                    self.data_rows = random.sample(self.data_rows, k=num_samples)

    @classmethod
    def from_data_rows(cls, args, tokenizer, data_type, is_training, data_rows):
        dataset = cls.__new__(cls)
        dataset.args = args
        dataset.tokenizer = tokenizer

        if args.dataset_subset != "none":
            dataset.task = f"{DATASET_NAME}/{args.dataset_subset}"
        else:
            dataset.task = DATASET_NAME

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset.pad_token = tokenizer.pad_token
        dataset.data_type = data_type
        dataset.is_training = is_training
        dataset.data_rows = data_rows
        return dataset

    def save_prepared_cache(self, cache_path):
        os.makedirs(cache_path, exist_ok=True)
        metadata_path = os.path.join(cache_path, "metadata.json")
        rows_path = os.path.join(cache_path, "rows")

        if hasattr(self.data_rows, "save_to_disk") and len(self.data_rows) > 0:
            self.data_rows.save_to_disk(rows_path)
            metadata = {"format": "hf_dataset"}
        else:
            rows_path = os.path.join(cache_path, "rows.pkl")
            with open(rows_path, "wb") as fout:
                if hasattr(self.data_rows, "save_to_disk"):
                    pickle.dump(list(self.data_rows), fout, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    pickle.dump(self.data_rows, fout, protocol=pickle.HIGHEST_PROTOCOL)
            metadata = {"format": "pickle"}

        with open(metadata_path, "w", encoding="utf-8") as fout:
            json.dump(metadata, fout)

    @staticmethod
    def prepared_cache_exists(cache_path):
        metadata_path = os.path.join(cache_path, "metadata.json")
        if not os.path.isfile(metadata_path):
            return False

        try:
            with open(metadata_path, "r", encoding="utf-8") as fin:
                metadata = json.load(fin)
        except (OSError, json.JSONDecodeError):
            return False

        if metadata.get("format") == "hf_dataset":
            return os.path.isdir(os.path.join(cache_path, "rows"))
        if metadata.get("format") == "pickle":
            return os.path.isfile(os.path.join(cache_path, "rows.pkl"))
        return False

    @classmethod
    def from_prepared_cache(cls, args, tokenizer, data_type, is_training, cache_path):
        metadata_path = os.path.join(cache_path, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as fin:
            metadata = json.load(fin)

        if metadata["format"] == "hf_dataset":
            from datasets import load_from_disk
            data_rows = load_from_disk(os.path.join(cache_path, "rows"), keep_in_memory=True)
        elif metadata["format"] == "pickle":
            with open(os.path.join(cache_path, "rows.pkl"), "rb") as fin:
                data_rows = pickle.load(fin)
        else:
            raise ValueError(f"Unsupported prepared dataset cache format: {metadata['format']}")

        return cls.from_data_rows(
            args=args,
            tokenizer=tokenizer,
            data_type=data_type,
            is_training=is_training,
            data_rows=data_rows,
        )

    def __len__(self):
        return len(self.data_rows)

    def __getitem__(self, index):
        batch = self.data_rows[index]
        return self.causal_lm_collator(batch)

    def read_data_from_file(self):
        # Training reads the train split; validation and test read the test sources.
        return OpenParallelThinkDataReader.jsonl_file_reader(self.args, self.data_type)

    def compute_seq_length(self, prompt, response):
        input_text = f"{prompt}{response}{self.tokenizer.eos_token}"
        tokenized_input = self.tokenizer.encode(input_text, add_special_tokens=True)
        return len(tokenized_input) + 32

    def encode_with_prompt_completion_format(
        self, prompt, completion, max_seq_length, use_chat_template=False, separator=""
    ):
        """
        Here we assume each example has 'prompt' and 'completion' fields.
        We concatenate prompt and completion and tokenize them together because otherwise prompt will be too long.
        and it doesn't make sense to follow directly with the completion.
        ``separator`` goes between prompt and completion when no chat template is used.
        """
        if use_chat_template:
            prompt_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            example_text = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
                ],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False
            )
        else:
            prompt_text = f"{prompt}{separator}"
            example_text = f"{prompt_text}{completion}{self.tokenizer.eos_token}"

        tokenized_prompt = self.tokenizer(
            prompt_text,
            return_tensors='pt',
            max_length=max_seq_length,
            truncation=True
        )

        tokenized_example = self.tokenizer(
            example_text,
            return_tensors='pt',
            max_length=max_seq_length,
            truncation=True,
            padding='max_length',
        )

        input_ids = tokenized_example.input_ids
        attention_mask = tokenized_example.attention_mask
        labels = input_ids.clone()

        prompt_len = int(tokenized_prompt.input_ids.shape[1])
        if torch.any((attention_mask[0, 1:] - attention_mask[0, :-1]) < 0):
            raise ValueError("Right/mixed padding detected; expected left padding only.")

        prompt_start_idx = int((attention_mask[0] == 0).sum().item())
        prompt_end_idx = min(prompt_start_idx + prompt_len, labels.shape[1])

        labels[:, prompt_start_idx:prompt_end_idx] = -100
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids.flatten(),
            'labels': labels.flatten(),
            'attention_mask': attention_mask.flatten(),
        }

    def _normalize_queries(self, qa_data):
        queries = qa_data.get("queries")
        if not queries:
            if "question" in qa_data or "answer" in qa_data:
                queries = [{
                    "question": qa_data.get("question", ""),
                    "answer": qa_data.get("answer", ""),
                    "gold_answer": qa_data.get("gold_answer", qa_data.get("answer", "")),
                    "support": qa_data.get("support", "")
                }]
            else:
                return []

        if isinstance(queries, dict):
            list_lengths = [len(value) for value in queries.values() if isinstance(value, list)]
            if list_lengths:
                count = min(list_lengths)
                expanded = []
                for i in range(count):
                    expanded.append({
                        key: (value[i] if isinstance(value, list) else value)
                        for key, value in queries.items()
                    })
                queries = expanded
            else:
                queries = [queries]
        elif isinstance(queries, str):
            queries = [queries]

        normalized = []
        for idx, problem in enumerate(queries):
            if isinstance(problem, dict):
                question = problem.get("question") or problem.get("query") or ""
                answer = problem.get("answer") or problem.get("output") or ""
                gold_answer = problem.get("gold_answer", answer)
                support = problem.get("support", "")
                normalized_problem = {
                    "question": question,
                    "answer": answer,
                    "gold_answer": gold_answer,
                    "support": support
                }
                normalized.append(normalized_problem)
                continue
            if isinstance(problem, (list, tuple)):
                normalized.append({
                    "question": problem[0] if len(problem) > 0 else "",
                    "answer": problem[1] if len(problem) > 1 else "",
                    "gold_answer": problem[1] if len(problem) > 1 else "",
                    "support": problem[2] if len(problem) > 2 else ""
                })
                continue
            if isinstance(qa_data.get("answers"), list) and idx < len(qa_data["answers"]):
                answer = qa_data["answers"][idx]
            else:
                answer = qa_data.get("answer", "")
            normalized.append({
                "question": str(problem),
                "answer": answer,
                "gold_answer": answer,
                "support": ""
            })
        return normalized

    def _normalize_task(self, qa_data):
        task = qa_data.get("task", "")
        if isinstance(task, dict):
            if "task" in task:
                task = task["task"]
            else:
                return ""
        if isinstance(task, (list, tuple)):
            if not task:
                return ""
            if all(isinstance(item, str) and len(item) == 1 for item in task):
                return "".join(task)
            if len(task) == 1:
                return str(task[0])
            return str(task[0])
        if hasattr(task, "item"):
            try:
                task = task.item()
            except Exception:
                pass
        return task if isinstance(task, str) else str(task)

    def causal_lm_collator(self, qa_data):
        """
        **batch collator for this custom class**
        :param qa_data: the dict of a reasoning problem with context, question, and answer
        """
        train_input_ids_batch = []
        train_attention_mask_batch = []
        train_labels_batch = []
        train_inputs, train_outputs = [], []
        max_seq_length = 0
        sequences = []
        context = qa_data["context"]

        record_ids = [i for i in range(len(context))]
        for idx, fact in zip(record_ids, context):
            sequences.append((f"record_{idx}:", fact))
            max_seq_length = max(
                max_seq_length,
                self.compute_seq_length(f"record_{idx}:", fact)
            )

        max_seq_length = min(max_seq_length, self.tokenizer.model_max_length)
        for prompt, response in sequences:
            encoded = self.encode_with_prompt_completion_format(
                prompt,
                response,
                max_seq_length=max_seq_length
            )
            train_input_ids_batch.append(encoded['input_ids'])
            train_attention_mask_batch.append(encoded['attention_mask'])
            train_labels_batch.append(encoded['labels'])
            train_inputs.append(prompt)
            train_outputs.append(response)

        dev_input_ids_batch = []
        dev_attention_mask_batch = []
        dev_labels_batch = []
        dev_inputs, dev_outputs = [], []
        sequences = []
        max_seq_length = 0

        prefix = f"record_0 to record_{len(context) - 1} are the context you memorized."
        queries = self._normalize_queries(qa_data)
        gold_answers = []
        for problem in queries:
            prompt, response = OpenParallelThinkDataReader.format_query(problem, prefix=prefix)
            gold_answers.append(str(problem.get("gold_answer", problem["answer"])))
            sequences.append((prompt, response))
            max_seq_length = max(
                max_seq_length,
                self.compute_seq_length(
                    prompt, response
                )
            )

        max_seq_length = min(max_seq_length, self.tokenizer.model_max_length)

        for prompt, response in sequences:
            encoded = self.encode_with_prompt_completion_format(
                prompt,
                response,
                max_seq_length=max_seq_length,
                use_chat_template=uses_chat_template(self.tokenizer),
                separator=PROMPT_COMPLETION_SEPARATOR,
            )
            dev_input_ids_batch.append(encoded['input_ids'])
            dev_attention_mask_batch.append(encoded['attention_mask'])
            dev_labels_batch.append(encoded['labels'])
            dev_inputs.append(prompt)
            dev_outputs.append(str(response))

        try:
            task = self._normalize_task(qa_data)
            feature = {
                "input_ids": torch.stack(dev_input_ids_batch, dim=0),
                "attention_mask": torch.stack(dev_attention_mask_batch, dim=0),
                "labels": torch.stack(dev_labels_batch, dim=0),
                "train_input_ids": torch.stack(train_input_ids_batch, dim=0),
                "train_attention_mask": torch.stack(train_attention_mask_batch, dim=0),
                "train_labels": torch.stack(train_labels_batch, dim=0),
                "print_out": {
                    "guid": qa_data["guid"],
                    "prompt": dev_inputs,
                    "response": dev_outputs,
                    "gold_answer": gold_answers,
                    "task": task,
                }
            }

        except RuntimeError as e:
            pprint(qa_data)
            raise (e)
        return feature


def create_dataloader(args, dataset, is_training):
    batch_size = args.train_batch_size if is_training else args.eval_batch_size

    if is_training:
        if bool(getattr(args, "train_sequential_sampler", False)):
            sampler = SequentialSampler(dataset)
        else:
            sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=args.num_workers,
        collate_fn=unroll,
    )


def unroll(batch):
    if len(batch) != 1:
        raise ValueError(
            "unroll supports batch_size=1 only; refusing to discard tasks"
        )
    return batch[0]
