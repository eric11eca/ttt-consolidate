# Consolidating Reasoning with Test-Time Learning

When a model solves a hard problem, it often writes many reasoning attempts.
This project teaches the model to learn from those attempts at test time.
The model reads a set of reasoning attempts for a problem, briefly fine-tunes
a small set of adapter weights on them, and then answers the problem with the
updated weights.

Training is meta-learning: the model learns how to learn from reasoning
attempts, so that a few quick updates at test time are enough.

This repository has the code to train and evaluate the math model from the
paper, based on Qwen3-4B-Instruct. For more details, please refer to our
[paper](https://openreview.net/forum?id=dwDo02oHXs).

> **Note**: This is a partial codebase. More details will be added, and the
> code will be refactored.

## Setup

```bash
git clone https://github.com/eric11eca/ttt-consolidate.git
cd ttt-consolidate
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Usage

Launch the code with `torchrun`:

```bash
torchrun --nproc_per_node=<num_gpus> train.py
```

Settings are in `configs/`.

> **Note**: FlashAttention does not support double backward. Only enable it
> (`attn_implementation=flash_attention_2`) if performance does not drop
> significantly. Otherwise use `attn_implementation=sdpa`.

> **Note**: Packing (`packing=true`) only works with FlashAttention.

> **Note**: To save memory, turn on inner-loop accumulation with
> `inner_grad_accum=<n>` (see below).

## How the inner loop works

Training has two loops. The inner loop fine-tunes the LoRA weights on the
reasoning attempts for a problem. The outer loop measures how well the
fine-tuned weights answer the problem, and trains the starting weights so that
the inner loop works well. For this, the outer loss must be differentiated
through every inner-loop update. The code in `perk/modules/inner_loop/` does
this as follows:

- **Weights as plain tensors.** The model is turned into a function of its
  weights (`functorch.make_functional_with_buffers`). The weights are a list
  of tensors that the code passes in, not fixed module state.
- **Gradients that keep their graph.** Each inner step computes the gradient
  of the inner loss with `torch.autograd.grad(..., create_graph=True)`. The
  gradient is itself a differentiable function of the weights.
- **Out-of-place updates.** The inner optimizer is a functional AdamW
  (`perk/modules/optimizers.py`). It returns new tensors (`p + update`) and
  never changes weights in place. So the adapted weights stay connected to
  the starting weights in the autograd graph.
- **Learned learning rates.** The inner learning rate is a trainable table
  with one value per weight tensor and per inner step. The outer loss also
  trains this table.
- **Outer backward.** The outer loss uses the adapted weights. Its backward
  pass goes back through all inner updates. This includes second-order terms
  (gradients of gradients).
- **Truncated unrolling.** Inner steps before `unroll_start` are not
  differentiated through: their gradients are detached. With the defaults
  (`n_inner_iter=4`, `unroll_start=2`), the last 2 of the 4 steps are
  differentiated.
- **Checkpointing.** The inner forward pass, the gradient computation and the
  update are each wrapped in `torch.utils.checkpoint`. Their activations are
  not stored. They are computed again during the outer backward pass. This
  costs extra compute and saves memory.
- **Double backward.** Second-order gradients need an attention kernel that
  supports double backward. The inner forward pass uses the PyTorch SDPA math
  kernel for this reason.

### Inner-loop accumulation

The memory for one inner step grows with the length of the reasoning attempts
(the context). Inner-loop accumulation splits the context into groups and
processes one group at a time:

1. The context is split into `n` groups with about the same number of tokens.
2. For each group, the code computes the loss, divides it by `n`, and computes
   its gradient.
3. The gradients of all groups are added together.
4. One AdamW update is made with the summed gradient.

Only one group is in memory at a time. With checkpointing, each group is also
computed again one at a time during the outer backward pass. So peak memory
depends on the group size, not on the full context length. The cost is more,
smaller passes, which is slower.

`inner_grad_accum` sets the minimum number of groups. Contexts longer than
8192 tokens are split further automatically (for example, into 4 groups up to
16384 tokens and 8 groups up to 32768 tokens).

## Citation

```
@inproceedings{
haowu2026consolidating,
title={Consolidating Reasoning with Test-Time Learning},
author={Haotian Wu and Zeming Chen and Hao Zhao and Antoine Bosselut},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=dwDo02oHXs}
}
```
