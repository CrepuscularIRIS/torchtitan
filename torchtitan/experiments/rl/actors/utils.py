# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
import torch.nn.functional as F

from torchtitan.models.common.attention import VarlenMetadata


# TODO We should either unify all the mask creation for RL, or move them to a
#      single file.
def create_varlen_metadata(seq_lens: list[int], device: torch.device) -> VarlenMetadata:
    """Build VarlenMetadata from sequence lengths.

    Example:
        seq_lens = [3, 5, 2]
        -> cu_seqs = [0, 3, 8, 10], max_len = 5
    """
    cu_seqs = torch.tensor(
        [0] + list(itertools.accumulate(seq_lens)), dtype=torch.int32, device=device
    )
    max_len = max(seq_lens)
    return VarlenMetadata(
        cu_seq_q=cu_seqs, cu_seq_k=cu_seqs, max_q=max_len, max_k=max_len
    )


def compute_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Compute per-token logprobs from logits.

    Returns logprobs for positions 1..N (the predicted tokens).
    Output shape is ``[batch, seq_len - 1]``.
    """
    shift_logits = logits[:, :-1, :].float()
    shift_targets = token_ids[:, 1:]
    logprobs = F.log_softmax(shift_logits, dim=-1)
    return logprobs.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)


def extract_response_logprobs(
    packed_logprobs: torch.Tensor,
    seq_lens: list[int],
    prompt_lens: list[int],
    response_lens: list[int],
) -> list[torch.Tensor]:
    """Extract per-sample response logprobs from packed logprobs."""
    seq_start = 0
    result = []
    for i in range(len(seq_lens)):
        # Logprobs are shifted: position j holds logprob of token j+1,
        # so response start (seq_start + prompt_len) maps to index
        # (seq_start + prompt_len - 1) in the logprobs tensor.
        s = seq_start + prompt_lens[i] - 1
        e = s + response_lens[i]
        result.append(packed_logprobs[0, s:e])
        seq_start += seq_lens[i]
    return result


def create_positions_from_seq_lens(
    seq_lens: list[int], device: torch.device
) -> torch.Tensor:
    """Build a ``[1, total_tokens]`` positions tensor that resets at each sequence boundary.

    Example:
        seq_lens = [3, 5, 2]
        -> positions = [[0, 1, 2, 0, 1, 2, 3, 4, 0, 1]]
    """
    return torch.cat([torch.arange(l, device=device) for l in seq_lens]).unsqueeze(0)


def compute_token_log_probs(
    model: torch.nn.Module,
    prompt_ids: list[int],
    gen_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute per-token log probabilities for generated tokens.
    TODO Only batch size 1 is supported for now.

    Args:
        model: The model to use for computing logits
        prompt_ids: Prompt token IDs
        gen_ids: Generated token IDs
        device: Device to run computation on

    Returns:
        Per-token log probabilities for the generated tokens
    """
    token_ids = torch.tensor(prompt_ids + gen_ids, dtype=torch.long, device=device)
    prompt_len = len(prompt_ids)
    gen_len = len(gen_ids)
    seq_lens = [len(prompt_ids) + len(gen_ids)]
    attention_masks = create_varlen_metadata(seq_lens, device)

    full_tensor = token_ids.unsqueeze(0)

    # NOTE: We should move towards batching to improve efficiency here
    # See https://github.com/pytorch/torchtitan/issues/2674
    # Explicit positions avoid dynamic rope_cache[0:seqlen] slice in RoPE,
    # which breaks torch.compile with symbolic shapes.
    seq_len = full_tensor.shape[1]
    positions = torch.arange(seq_len, device=device).unsqueeze(0)

    logits = model(full_tensor, attention_masks=attention_masks, positions=positions)

    # Convert to float32 for numerical stability
    logits_f32 = logits[:, :-1, :].to(torch.float32)
    log_probs = F.log_softmax(logits_f32, dim=-1)
    target_tokens = full_tensor[:, 1:]

    # Extract log probs for generated tokens only
    gen_start_idx = prompt_len - 1
    gen_end_idx = gen_start_idx + gen_len

    gen_token_logprobs = log_probs[0, gen_start_idx:gen_end_idx, :]
    gen_token_ids_tensor = target_tokens[0, gen_start_idx:gen_end_idx]
    token_lps = gen_token_logprobs.gather(
        1, gen_token_ids_tensor.unsqueeze(-1)
    ).squeeze(-1)

    return token_lps


def verify_logprob_identity(
    vllm_token_log_probs: list[list[float]],
    batch_token_log_probs: list[torch.Tensor],
) -> dict:
    """
    Check if vLLM log probs and computed log probs are bit-wise identical,
    and compute the log ratio (train/generator) between them.

    Args:
        vllm_token_log_probs: Per-token log probs from vLLM (generator)
        batch_token_log_probs: Per-token log probs computed by the trainer model

    Returns:
        Verification result dict with identity status, delta info, and log ratio stats
    """
    result = {
        "logprob_bitwise_identical": True,
        "num_samples_checked": len(vllm_token_log_probs),
        "total_tokens_checked": 0,
        "num_tokens_different": 0,
        "logprob_max_delta": 0.0,
        "avg_delta": 0.0,
        "logprob_diff_mean": 0.0,
        "logprob_diff_max": 0.0,
    }

    all_deltas = []
    all_log_ratios = []

    for vllm_lps, titan_lps in zip(vllm_token_log_probs, batch_token_log_probs):
        # Convert vLLM log probs to tensor
        vllm_tensor = torch.tensor(vllm_lps, dtype=torch.float32)
        # Convert titan log probs to float32 for comparison
        titan_tensor = titan_lps.detach().cpu().float()

        num_tokens = len(vllm_lps)
        result["total_tokens_checked"] += num_tokens

        # Check bitwise identity
        bitwise_match = torch.equal(vllm_tensor, titan_tensor)

        if not bitwise_match:
            result["logprob_bitwise_identical"] = False
            num_different = (vllm_tensor != titan_tensor).sum().item()
            result["num_tokens_different"] += num_different
            deltas = (vllm_tensor - titan_tensor).abs()
            all_deltas.append(deltas)

        # Log ratio: log(pi_train / pi_generator) = logprob_train - logprob_generator
        # Should be 0 when weights are identical (ratio = 1)
        all_log_ratios.append(titan_tensor - vllm_tensor)

    # Compute aggregate delta stats
    if all_deltas:
        combined_deltas = torch.cat(all_deltas)
        result["logprob_max_delta"] = combined_deltas.max().item()
        result["avg_delta"] = combined_deltas.mean().item()

    # Compute log ratio stats
    if all_log_ratios:
        combined_log_ratios = torch.cat(all_log_ratios)
        result["logprob_diff_mean"] = combined_log_ratios.mean().item()
        result["logprob_diff_max"] = combined_log_ratios.abs().max().item()

    return result
