# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Trainer for paged stash experiments with 3-level overflow defense.

Extends GraphTrainer with paged stash buffer management and Megatron-style
overflow handling:

1. **Host spillover**: When CUDA stash buffers are full, the Triton copy kernel
   falls back to a pinned host buffer.  The pop kernel reads ``spilled_to_host``
   to select the source.  Warning only, no retry needed.

2. **Cross-rank detection**: After each fwd/bwd, stash overflow, host spill, and
   HybridEP over-budget flags are ``all_reduce``-d so all ranks agree.

3. **Retry/rerun**: On stash overflow (both CUDA + host exhausted) or dispatch
   over-budget, gradients are zeroed, CUDA graphs reset, buffers optionally grown,
   and the step is re-run.  Max 2 attempts.  Optimizer step only executes after
   successful fwd/bwd.

The CUDA graph lifecycle is: warmup → capture → replay.  On overflow during
replay, the graph is reset (via ``CUDAGraphWrapper.reset()``) and re-captured
on the next step.
"""

from dataclasses import dataclass, field
from typing import Iterator

import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.cuda_graphable_moe.configs import (
    PagedStashActivationCheckpointConfig,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.cudagraph import (
    _cg_manager,
    CUDAGraphWrapper,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.tools.logging import logger


def _reset_all_cuda_graphs() -> None:
    """Reset all CUDAGraphWrappers for re-capture without full teardown.

    After overflow-triggered buffer growth, captured CUDA graphs hold stale
    buffer pointers. This resets each wrapper so the next call does a fresh
    warmup → capture cycle with the new buffer addresses.
    """
    for wrapper in _cg_manager._cudagraph_wrappers:
        wrapper._cudagraph = None
        wrapper._has_warmup = False
        wrapper._args = None
        wrapper._output = None


def _check_overflow_all_ranks(
    paged_stash_overflow: torch.Tensor | None,
    overbudget_flag: torch.Tensor | None,
    host_spill_flag: torch.Tensor | None,
) -> tuple[int, int, int]:
    """Check for overflow across all ranks via a single all_reduce.

    Packs three GPU-resident flags into a single ``all_reduce(SUM)`` so every
    rank agrees on the outcome (critical to avoid deadlocks).

    Returns (stash_overflow_ranks, overbudget_ranks, host_spill_ranks).
    """
    device = "cuda"
    if paged_stash_overflow is not None:
        device = paged_stash_overflow.device

    overflow_val = torch.zeros(1, dtype=torch.int32, device=device)
    if paged_stash_overflow is not None:
        overflow_val += (paged_stash_overflow != 0).to(torch.int32)

    overbudget_val = torch.zeros(1, dtype=torch.int32, device=device)
    if overbudget_flag is not None:
        overbudget_val += overbudget_flag.to(torch.int32).view(1)

    host_spill_val = torch.zeros(1, dtype=torch.int32, device=device)
    if host_spill_flag is not None:
        host_spill_val += (host_spill_flag != 0).to(torch.int32)

    flags = torch.stack(
        [overflow_val.view(-1)[0], overbudget_val.view(-1)[0], host_spill_val.view(-1)[0]],
        dim=0,
    )

    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.SUM)

    return flags[0].item(), flags[1].item(), flags[2].item()


class PagedStashTrainer(GraphTrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(GraphTrainer.Config):
        activation_checkpoint: PagedStashActivationCheckpointConfig = field(
            default_factory=PagedStashActivationCheckpointConfig
        )
        compile: GraphTrainerCompileConfig = field(
            default_factory=GraphTrainerCompileConfig
        )

    def close(self) -> None:
        # Unregister paged stash buffers from the module-level registry before
        # GraphTrainer.close().  This removes the strong references to buffer
        # tensors held by the registry.
        from torchtitan.experiments.cuda_graphable_moe.paged_stash_ops import (
            unregister_paged_stash_buffer,
        )

        for model_part in self.model_parts:
            buffers = getattr(model_part, "_paged_stash_buffers", None)
            if buffers:
                for buf in buffers:
                    unregister_paged_stash_buffer(id(buf))
        super().close()

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        """Training step with overflow detection and retry.

        Replicates the base ``Trainer.train_step`` logic but inserts overflow
        checking between backward and optimizer step.  On overflow, zeros grads,
        optionally grows buffers, resets CUDA graphs, and re-runs fwd/bwd with
        the same microbatch data.  Max ``1 + max_retries`` attempts.
        """
        ac_config = self.config.activation_checkpoint
        max_retries = getattr(ac_config, "paged_stash_max_retries", 1)
        overflow_detection = getattr(
            ac_config, "paged_stash_overflow_detection", True
        )
        grow_on_overflow = getattr(ac_config, "paged_stash_grow_on_overflow", True)

        self._reset_buffers()
        self.optimizers.zero_grad()

        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]
        parallel_dims = self.parallel_dims

        # 1. Collect microbatches (consume iterator once — needed for replay)
        microbatches = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        for _microbatch in range(self.gradient_accumulation_steps):
            input_dict, labels = next(data_iterator)
            local_valid_tokens += (labels != IGNORE_INDEX).sum()
            microbatches.append((input_dict, labels))

        local_valid_tokens = local_valid_tokens.to(self.device)
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            global_valid_tokens = dist_utils.dist_sum(local_valid_tokens, batch_mesh)
        else:
            global_valid_tokens = local_valid_tokens.float()

        # 2. Forward/backward with retry (max 1 + max_retries attempts)
        num_attempts = 1 + (max_retries if overflow_detection else 0)
        for attempt in range(num_attempts):
            if attempt > 0:
                self._reset_buffers()
                self.optimizers.zero_grad()

            accumulated_losses = []
            for input_dict, labels in microbatches:
                for k, v in input_dict.items():
                    if isinstance(v, torch.Tensor):
                        input_dict[k] = v.to(self.device)
                labels = labels.to(self.device)

                loss = self.forward_backward_step(
                    input_dict=input_dict,
                    labels=labels,
                    global_valid_tokens=global_valid_tokens,
                )
                accumulated_losses.append(loss.detach())

            if not overflow_detection:
                break

            # Check overflow across all ranks
            overflow_ranks, overbudget_ranks, spill_ranks = (
                self._check_overflow_all_ranks()
            )

            if overflow_ranks == 0 and overbudget_ranks == 0:
                if spill_ranks > 0:
                    logger.warning(
                        "Paged stash: spilled activations to pinned host on "
                        "%d rank(s) (CUDA stash full). Consider increasing "
                        "paged_stash_buffer_size_factor.",
                        spill_ranks,
                    )
                break

            # Overflow or over-budget — prepare for retry
            if overflow_ranks > 0:
                logger.warning(
                    "Paged stash: stash buffer overflow on %d rank(s).",
                    overflow_ranks,
                )
            if overbudget_ranks > 0:
                logger.warning(
                    "Paged stash: HybridEP dispatch over-budget on %d rank(s).",
                    overbudget_ranks,
                )

            if attempt < num_attempts - 1:
                logger.warning(
                    "Paged stash: retrying step (attempt %d/%d).",
                    attempt + 2,
                    num_attempts,
                )
                self._prepare_for_retry(
                    grow=grow_on_overflow and overflow_ranks > 0,
                )
            else:
                logger.error(
                    "Paged stash: overflow on final attempt. Gradients for "
                    "this step may be corrupt. Consider increasing "
                    "paged_stash_buffer_size_factor or "
                    "paged_stash_host_buffer_size_factor.",
                )

        # 3. Optimizer step (only after successful fwd/bwd)
        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        loss = torch.sum(torch.stack(accumulated_losses))

        self._log_buffer_usage()

        # 4. Log metrics (mirrors base Trainer.train_step)
        if not self.metrics_processor.should_log(self.step):
            return

        if parallel_dims.dp_cp_enabled:
            loss = loss.detach()
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            local_avg_loss = loss * global_valid_tokens / local_valid_tokens
            global_avg_loss, global_max_loss, global_ntokens_seen = (
                dist_utils.dist_sum(loss, loss_mesh),
                dist_utils.dist_max(local_avg_loss, loss_mesh),
                dist_utils.dist_sum(
                    torch.tensor(
                        self.ntokens_seen, dtype=torch.int64, device=self.device
                    ),
                    loss_mesh,
                ),
            )
        else:
            global_avg_loss = global_max_loss = float(loss.detach().item())
            global_ntokens_seen = self.ntokens_seen

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            "lr": lr,
        }
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
        )

    def _reset_buffers(self) -> None:
        """Reset paged stash buffers and overflow/host_spill flags before each step."""
        for model_part in self.model_parts:
            buffers = getattr(model_part, "_paged_stash_buffers", None)
            if buffers:
                for buf in buffers:
                    buf.reset()
            overflow = getattr(model_part, "_paged_stash_overflow", None)
            if overflow is not None:
                overflow.zero_()
            host_spill = getattr(model_part, "_paged_stash_host_spill", None)
            if host_spill is not None:
                host_spill.zero_()

    def _check_overflow_all_ranks(self) -> tuple[int, int, int]:
        """Check stash overflow, dispatch over-budget, and host spill across all ranks.

        Uses a single ``all_reduce(SUM)`` of 3 flags so every rank agrees on
        the outcome.  This is critical: even if only one rank overflows, ALL
        ranks must take the same code path or training deadlocks.

        Returns (stash_overflow_ranks, overbudget_ranks, host_spill_ranks).
        """
        # Stash overflow
        paged_overflow = None
        host_spill = None
        for model_part in self.model_parts:
            ovf = getattr(model_part, "_paged_stash_overflow", None)
            if ovf is not None:
                paged_overflow = ovf
            hs = getattr(model_part, "_paged_stash_host_spill", None)
            if hs is not None:
                host_spill = hs

        # HybridEP over-budget
        overbudget = None
        try:
            from torchtitan.distributed.deepep.hybridep import (
                check_hybridep_over_budget,
            )

            overbudget = check_hybridep_over_budget()
        except ImportError:
            pass

        return _check_overflow_all_ranks(paged_overflow, overbudget, host_spill)

    def _prepare_for_retry(self, *, grow: bool = False) -> None:
        """Prepare for rerunning the step.

        Zeros gradients, optionally grows buffers, resets free lists (but leaves
        overflow flag cleared by _reset_buffers on next attempt), and resets
        CUDA graphs for re-capture.
        """
        # Zero all gradients
        for model_part in self.model_parts:
            for p in model_part.parameters():
                if p.grad is not None:
                    p.grad.zero_()

        # Optionally grow buffers so the retry has more space
        if grow:
            for model_part in self.model_parts:
                buffers = getattr(model_part, "_paged_stash_buffers", None)
                if buffers:
                    for buf in buffers:
                        buf.grow(factor=2.0)

        # Reset CUDA graphs for re-capture
        _reset_all_cuda_graphs()

    def _log_buffer_usage(self) -> None:
        """Log paged stash buffer usage after each step."""
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            for model_part in self.model_parts:
                buffers = getattr(model_part, "_paged_stash_buffers", None)
                if buffers:
                    for buf in buffers:
                        logger.debug(
                            "train_step: hidden_size=%d, dtype=%s, "
                            "cuda_pages_consumed=%d/%d, host_pages=%d",
                            buf.hidden_size,
                            buf.dtype,
                            buf.free_list_head[0].item(),
                            buf.num_cuda_pages,
                            buf.num_host_pages,
                        )
