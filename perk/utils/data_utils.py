"""Sequence packing and batch splitting for the inner-loop (context) tokens."""

import torch

from copy import deepcopy
from typing import List, Dict, Literal

def sequence_packing_flash(train_features):
    """Pack sequences for FlashAttention

    Args:
        train_features: List of batch dictionaries with keys:
        - input_ids: (batch, seq_len)
        - attention_mask: (batch, seq_len) - 1D padding mask
        - labels: (batch, seq_len)

    Returns:
        List of packed batch dictionaries with keys:
        - input_ids: (1, packed_len) - packed without padding
        - attention_mask: (1, 1, packed_len, packed_len) - 2D block-diagonal mask
        - position_ids: (1, packed_len) - position IDs resetting at boundaries
        - labels: (1, packed_len) - with -100 at sequence starts
        - packed_seq_lengths: (num_sequences,) - length of each packed sequence
        - document_ids: (packed_len,) - document ID for each token
    """
    train_features_packed = []
    for train_batch in train_features:
        attn_mask = train_batch.pop("attention_mask")
        train_batch["input_ids"] = train_batch["input_ids"][attn_mask.bool()].unsqueeze(0)
        train_batch["position_ids"] = attn_mask.cumsum(1)[attn_mask.bool()].unsqueeze(0) - 1
        train_batch["labels"] = train_batch["labels"][attn_mask.bool()].unsqueeze(0)
        train_batch["labels"][train_batch["position_ids"] == 0] = -100

        # Calculate cumulative sequence lengths for queries and keys to prevent graph breaks during further computations.
        flattened_position_ids = train_batch["position_ids"].flatten()
        indices_q = torch.arange(
            flattened_position_ids.size(0), device=flattened_position_ids.device, dtype=torch.int32
        )
        train_batch["cu_seq_lens_q"] = torch.cat(
            (
                indices_q[flattened_position_ids == 0],
                torch.tensor(
                    flattened_position_ids.size(),
                    device=flattened_position_ids.device,
                    dtype=torch.int32
                ),
            )
        )
        train_batch["cu_seq_lens_k"] = train_batch["cu_seq_lens_q"]

        # Determine maximum sequence lengths to prevent graph breaks during further computations.
        train_batch["max_length_k"] = torch.tensor([flattened_position_ids.max().item() + 1])
        train_batch["max_length_q"] = train_batch["max_length_k"]
        train_features_packed.append(train_batch)

    return train_features_packed


def _get_document_ids_from_seq_lengths(
    seq_lengths: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert sequence lengths to document IDs.

    For example, seq_lengths=[3, 2, 4] produces document_ids=[0, 0, 0, 1, 1, 2, 2, 2, 2]

    Args:
        seq_lengths: Tensor of shape (num_sequences,) with length of each sequence
        device: Device to create the tensor on

    Returns:
        document_ids: Tensor of shape (total_length,) where each element indicates
                     which document/sequence that token belongs to
    """
    return torch.repeat_interleave(
        torch.arange(len(seq_lengths), device=device, dtype=torch.int32),
        seq_lengths.to(device)
    )


def _create_block_causal_mask_block_diag(
    seq_lengths: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create block-diagonal causal mask using torch.block_diag.

    This is the most efficient method when the number of sequences is moderate.
    Creates individual causal (lower triangular) masks for each sequence,
    then combines them into a block-diagonal structure.

    Following Unsloth/TorchTune's approach of using torch.block_diag for
    efficient block-diagonal mask construction.

    Args:
        seq_lengths: Tensor of shape (num_sequences,) with length of each sequence
        device: Device to create the mask on
        dtype: Data type for the mask (float for additive mask)

    Returns:
        Attention mask of shape (1, 1, total_len, total_len) in additive format
        where 0.0 = attend, -inf = mask out
    """
    total_length = seq_lengths.sum().item()
    causal_blocks = []
    for length in seq_lengths.tolist():
        if length > 0:
            block = torch.ones(length, length, dtype=torch.bool, device=device).tril()
            causal_blocks.append(block)

    if not causal_blocks:
        return torch.zeros(1, 1, 0, 0, dtype=dtype, device=device)

    combined_mask = torch.block_diag(*causal_blocks)

    attention_mask = torch.zeros(total_length, total_length, dtype=dtype, device=device)
    attention_mask.masked_fill_(~combined_mask, float("-inf"))

    return attention_mask.unsqueeze(0).unsqueeze(0)


def _create_block_causal_mask_document_ids(
    document_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create block-diagonal causal mask using document ID comparison.

    This approach is more memory-efficient for very long sequences as it
    avoids creating intermediate block tensors. It directly computes the
    mask by checking if tokens belong to the same document and satisfy
    the causal constraint.

    Following Unsloth's document_id pattern for uncontaminated packing.

    Args:
        document_ids: Tensor of shape (total_length,) where each element indicates
                     which document/sequence that token belongs to
        device: Device to create the mask on
        dtype: Data type for the mask (float for additive mask)

    Returns:
        Attention mask of shape (1, 1, total_len, total_len) in additive format
        where 0.0 = attend, -inf = mask out
    """
    T = document_ids.shape[0]

    # same_document[i, j] = True if token i and j are in the same document
    # This prevents cross-sequence contamination
    same_document = document_ids.unsqueeze(1) == document_ids.unsqueeze(0)  # [T, T]

    # causal[i, j] = True if i >= j (token at row i can attend to token at column j)
    # This creates a lower triangular mask (standard causal attention)
    indices = torch.arange(T, device=device)
    causal = indices.unsqueeze(1) >= indices.unsqueeze(0)  # [T, T]

    # Combined mask: both conditions must be satisfied
    # Token i can attend to token j iff:
    # 1. They are in the same document (no cross-contamination)
    # 2. j <= i (causal constraint)
    valid_attention = same_document & causal

    attention_mask = torch.zeros(T, T, dtype=dtype, device=device)
    attention_mask.masked_fill_(~valid_attention, float("-inf"))
    return attention_mask.unsqueeze(0).unsqueeze(0)


MaskStrategy = Literal["block_diag", "document_ids", "auto"]

def sequence_packing_standard(
    train_features: List[Dict[str, torch.Tensor]],
    mask_strategy: MaskStrategy = "auto",
) -> List[Dict[str, torch.Tensor]]:
    """
    Pack sequences for SDPA/eager attention following Unsloth's uncontaminated packing.

    Creates a block-diagonal causal attention mask that prevents attention across
    sequence boundaries while maintaining causality within each sequence. This
    implementation follows Unsloth's approach for padding-free uncontaminated packing.

    Key features:
    - Supports multiple mask creation strategies for different use cases
    - Computes sequence lengths directly from attention_mask (more efficient)
    - Stores packed_seq_lengths metadata for downstream use
    - Compatible with SDPA Math mode (supports higher-order gradients)

    The key difference from flash attention packing:
    - Flash attention uses cu_seq_lens (cumulative sequence lengths) with flash_attn_varlen_func
    - Standard/SDPA attention uses a full 2D attention mask

    Args:
        train_features: List of batch dictionaries with keys:
            - input_ids: (batch, seq_len)
            - attention_mask: (batch, seq_len) - 1D padding mask
            - labels: (batch, seq_len)
        mask_strategy: Strategy for creating the attention mask:
            - "block_diag": Use torch.block_diag (efficient for moderate sequence counts)
            - "document_ids": Use document ID comparison (memory-efficient for long sequences)
            - "auto": "document_ids" for more than 16 sequences, else "block_diag"

    Returns:
        List of packed batch dictionaries with keys:
            - input_ids: (1, packed_len) - packed without padding
            - attention_mask: (1, 1, packed_len, packed_len) - 2D block-diagonal mask
            - position_ids: (1, packed_len) - position IDs resetting at boundaries
            - labels: (1, packed_len) - with -100 at sequence starts
            - packed_seq_lengths: (num_sequences,) - length of each packed sequence
            - document_ids: (packed_len,) - document ID for each token

    Example:
        # Standard SDPA packing (works with higher-order gradients):
        packed = sequence_packing_standard(features, mask_strategy="block_diag")

        # For very long sequences (memory-efficient):
        packed = sequence_packing_standard(features, mask_strategy="document_ids")
    """
    train_features_packed = []

    for train_batch in train_features:
        attn_mask = train_batch.pop("attention_mask")
        device = attn_mask.device
        batch_size = attn_mask.shape[0]

        seq_lengths = attn_mask.sum(dim=1).to(torch.int32)  # [batch_size]
        total_length = seq_lengths.sum().item()

        # Pack tokens by removing padding
        valid_mask = attn_mask.bool()
        packed_input_ids = train_batch["input_ids"][valid_mask].unsqueeze(0)
        packed_labels = train_batch["labels"][valid_mask].unsqueeze(0)
        document_ids = _get_document_ids_from_seq_lengths(seq_lengths, device)

        # Create position_ids that reset at each sequence boundary
        position_ids_list = [
            torch.arange(length, device=device, dtype=torch.long)
            for length in seq_lengths.tolist()
        ]
        position_ids = torch.cat(position_ids_list).unsqueeze(0)  # (1, total_length)

        # Mask labels at sequence starts (position_id == 0)
        # This prevents loss computation on the first token of each sequence
        # Following Unsloth's mask_packed_sequence_boundaries approach
        packed_labels = packed_labels.clone()
        packed_labels[position_ids == 0] = -100

        effective_strategy = mask_strategy
        if mask_strategy == "auto":
            effective_strategy = "document_ids" if batch_size > 16 else "block_diag"

        if effective_strategy == "document_ids":
            attention_mask_2d = _create_block_causal_mask_document_ids(
                document_ids,
                device=device,
                dtype=torch.float32,
            )
        else:
            attention_mask_2d = _create_block_causal_mask_block_diag(
                seq_lengths,
                device=device,
                dtype=torch.float32,
            )

        train_batch["input_ids"] = packed_input_ids
        train_batch["labels"] = packed_labels
        train_batch["position_ids"] = position_ids
        train_batch["attention_mask"] = attention_mask_2d

        # Store metadata for potential downstream use
        train_batch["packed_seq_lengths"] = seq_lengths
        train_batch["document_ids"] = document_ids
        train_features_packed.append(train_batch)

    return train_features_packed

def get_accumulation_steps_1b_model(num_packed_tokens):
    """Number of groups the inner-loop context is split into, by its packed
    length in tokens, to bound activation memory. `inner_grad_accum` can only raise it."""
    if num_packed_tokens <= 8192:
        return 1
    if num_packed_tokens <= 9216:
        return 2
    if num_packed_tokens <= 16384:
        return 4
    if num_packed_tokens <= 32768:
        return 8
    if num_packed_tokens <= 65536:
        return 16
    if num_packed_tokens <= 131072:
        return 32
    return 64

def _segments_to_train_feature(segments, attention_mask_dtype):
    max_length = max(input_ids.numel() for input_ids, _ in segments)
    device = segments[0][0].device

    input_ids_batch = torch.zeros(
        (len(segments), max_length),
        dtype=segments[0][0].dtype,
        device=device,
    )
    attention_mask_batch = torch.zeros(
        (len(segments), max_length),
        dtype=attention_mask_dtype,
        device=device,
    )
    labels_batch = torch.full(
        (len(segments), max_length),
        -100,
        dtype=segments[0][1].dtype,
        device=device,
    )

    for idx, (input_ids, labels) in enumerate(segments):
        length = input_ids.numel()
        input_ids_batch[idx, :length] = input_ids
        attention_mask_batch[idx, :length] = 1
        labels_batch[idx, :length] = labels

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attention_mask_batch,
        "labels": labels_batch,
    }


def split_train_features_by_token_budget(
    input_ids,
    attention_mask,
    labels,
    num_splits,
):
    total_tokens = int(attention_mask.bool().sum().item())
    if num_splits <= 1 or total_tokens == 0:
        return [{
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }]

    # We can only create non-empty packed sequences. Split oversized rows when
    # there are fewer rows than requested accumulation steps.
    remaining_groups = min(num_splits, total_tokens)
    remaining_tokens = total_tokens
    train_features = []
    current_segments = []
    current_tokens = 0
    current_target = (remaining_tokens + remaining_groups - 1) // remaining_groups

    for row_idx in range(input_ids.shape[0]):
        valid_positions = attention_mask[row_idx].bool().nonzero(as_tuple=False).flatten()
        offset = 0

        while offset < valid_positions.numel():
            space = current_target - current_tokens
            if space <= 0:
                train_features.append(
                    _segments_to_train_feature(current_segments, attention_mask.dtype)
                )
                remaining_groups -= 1
                current_segments = []
                current_tokens = 0
                current_target = (
                    (remaining_tokens + remaining_groups - 1) // remaining_groups
                    if remaining_groups > 0 else remaining_tokens
                )
                continue

            take = min(space, valid_positions.numel() - offset)
            positions = valid_positions[offset:offset + take]
            current_segments.append((
                input_ids[row_idx, positions],
                labels[row_idx, positions],
            ))
            current_tokens += take
            remaining_tokens -= take
            offset += take

            if current_tokens >= current_target and remaining_groups > 1:
                train_features.append(
                    _segments_to_train_feature(current_segments, attention_mask.dtype)
                )
                remaining_groups -= 1
                current_segments = []
                current_tokens = 0
                current_target = (
                    (remaining_tokens + remaining_groups - 1) // remaining_groups
                    if remaining_groups > 0 else remaining_tokens
                )

    if current_segments:
        train_features.append(
            _segments_to_train_feature(current_segments, attention_mask.dtype)
        )

    return train_features


PackingMode = Literal["flash", "standard"]

def get_features(
    batch: Dict,
    packing: PackingMode = "flash",
    accumulation_steps: int = 1,
    mask_strategy: MaskStrategy = "auto",
):
    """Get features from batch with optional sequence packing.

    This function extracts training and evaluation features from a batch,
    with optional padding-free sequence packing following Unsloth's approach.

    Args:
        batch: The target batch containing train and dev data.
        packing: Packing mode for training features:
            - False/None: No packing (keep padding)
            - "flash": Pack for FlashAttention (uses cu_seq_lens)
            - "standard": Pack for SDPA/eager attention (uses 2D block-diagonal mask)
            - True: same as "flash"
        accumulation_steps: Number of groups to split the training tensors into.
            When accumulation_steps > 1, the training features will be divided
            into that many chunks for gradient accumulation.
        mask_strategy: Strategy for creating attention mask (only for packing="standard"):
            - "block_diag": Use torch.block_diag (efficient for moderate sequence counts)
            - "document_ids": Use document ID comparison (memory-efficient for long sequences)
            - "auto": "document_ids" for more than 16 sequences, else "block_diag"

    Returns:
        tuple: (train_features, dev_features, print_out)
            - train_features: List of feature dicts for training
            - dev_features: Feature dict for evaluation
            - print_out: Metadata for printing/logging

    Example:
        # For FlashAttention models:
        train_features, dev_features, _ = get_features(batch, packing="flash")

        # For SDPA with higher-order gradient support:
        train_features, dev_features, _ = get_features(batch, packing="standard")

        # For memory-efficient packing with very long sequences:
        train_features, dev_features, _ = get_features(
            batch, packing="standard", mask_strategy="document_ids"
        )

        # No packing:
        train_features, dev_features, _ = get_features(batch, packing=False)
    """
    print_out = batch["print_out"]
    dev_features = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"]
    }

    local_input_ids = batch["train_input_ids"]
    local_attention_mask = batch["train_attention_mask"]
    local_labels = batch["train_labels"]

    train_features = [{
        "input_ids": local_input_ids,
        "attention_mask": local_attention_mask,
        "labels": local_labels
    }]

    # Calculate packed token count to determine accumulation steps
    packed_features = sequence_packing_flash(deepcopy(train_features))
    num_packed_tokens = packed_features[0]["input_ids"].shape[1]

    num_accum_steps = get_accumulation_steps_1b_model(num_packed_tokens)
    num_accum_steps = max(num_accum_steps, accumulation_steps)

    if num_accum_steps > 1:
        train_features = split_train_features_by_token_budget(
            local_input_ids,
            local_attention_mask,
            local_labels,
            num_accum_steps,
        )

    if packing is True or packing == "flash":
        train_features = sequence_packing_flash(train_features)
    elif packing == "standard":
        train_features = sequence_packing_standard(
            train_features,
            mask_strategy=mask_strategy,
        )
    elif packing is False or packing is None:
        pass
    else:
        raise ValueError(f"unknown packing mode {packing!r}")

    return train_features, dev_features, print_out
