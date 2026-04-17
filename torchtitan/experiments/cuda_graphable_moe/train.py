# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Trainer for paged stash experiments.

Extends GraphTrainer with Megatron-style buffer observation:

1. Step 1 (CUDAGraph warmup): Stash ops run eagerly. The PagedStashObserver
   (status='capture') records actual/avg token counts. After the step,
   allocate_stash_buffers resizes buffers from observed peak. register_buffer
   propagates new tensor pointers to the GraphModule for capture.

2. Step 2 (CUDAGraph capture): Captures with right-sized buffers.

3. Step 3+ (replay): cudagraph.replay().
"""

from dataclasses import dataclass, field
from typing import Iterator

import torch

from torchtitan.experiments.cuda_graphable_moe.configs import (
    PagedStashActivationCheckpointConfig,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.tools.logging import logger


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
        # Clear PagedStashBuffer._registered_modules before GraphTrainer.close().
        # These hold strong references to the fwd/bwd GraphModules, which hold
        # CUDAGraphWrapper objects that keep NCCL communicator handles alive.
        # Without clearing these, GraphTrainer.close()'s joint_graph_module=None
        # and gc.collect() can't free the CUDA graphs, and
        # destroy_process_group() hangs waiting for NCCL to release.
        for model_part in self.model_parts:
            buffers = getattr(model_part, "_paged_stash_buffers", None)
            if buffers:
                for buf in buffers:
                    buf._registered_modules.clear()
        super().close()

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        # Reset paged stash buffers before each training step
        self._reset_buffers()

        # Enable observation on step 1 (CUDAGraph warmup — eager, .item() safe)
        if self.step == 1:
            from .paged_stash_ops import _observer

            _observer.paged_stash_reset()  # begin → capture

        super().train_step(data_iterator)

        self._log_buffer_usage()

        # After step 1 (warmup): observe and resize buffers.
        # CUDAGraphWrapper ran eagerly — buffers are warm, free_list_head has
        # actual page consumption. The observer has per-key peak concurrent usage.
        # Resize via register_buffer propagates new tensor pointers to the
        # GraphModule before CUDAGraph capture on step 2.
        if self.step == 1:
            from .paged_stash_ops import _observer

            _observer.paged_stash_reset()  # capture → captured

            for model_part in self.model_parts:
                buffers = getattr(model_part, "_paged_stash_buffers", None)
                if buffers:
                    stash_buffer_size_factor_cuda = getattr(
                        model_part, "_paged_stash_buffer_size_factor", 1.1
                    )
                    _observer.allocate_stash_buffers(
                        buffers, stash_buffer_size_factor_cuda,
                    )

        self._check_overflow()

    def _reset_buffers(self) -> None:
        """Reset paged stash buffers and overflow flag before each step."""
        for model_part in self.model_parts:
            buffers = getattr(model_part, "_paged_stash_buffers", None)
            if buffers:
                for buf in buffers:
                    buf.reset()
            overflow = getattr(model_part, "_paged_stash_overflow", None)
            if overflow is not None:
                overflow.zero_()

    def _log_buffer_usage(self) -> None:
        """Log paged stash buffer usage after each step."""
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            for model_part in self.model_parts:
                buffers = getattr(model_part, "_paged_stash_buffers", None)
                if buffers:
                    for buf in buffers:
                        logger.debug(
                            "train_step: hidden_size=%d, dtype=%s, "
                            "pages_consumed=%d, total_pages=%d",
                            buf.hidden_size, buf.dtype,
                            buf.free_list_head.item(), buf.num_pages,
                        )

    def _check_overflow(self) -> None:
        """Check for paged stash overflow after a step."""
        for model_part in self.model_parts:
            overflow = getattr(model_part, "_paged_stash_overflow", None)
            if overflow is not None and overflow.item() != 0:
                raise RuntimeError(
                    "PagedStashBuffer overflow detected! Increase "
                    "paged_stash_buffer_size_factor in activation_checkpoint config."
                )
