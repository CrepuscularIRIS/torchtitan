# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Paged stash buffer, Triton kernels, and custom ops for MoE activation storage.

Instead of recomputing activations (standard AC), stores them in a pre-allocated
paged buffer managed by Triton kernels. This avoids recompute cost while reducing
memory fragmentation for MoE expert layers with dynamic token counts.

Custom ops are registered via torch.library so they can be inserted into FX
graphs by the graph-based paged SAC pass.

page_record format: [num_tokens, page_id_0, page_id_1, ...]
This encodes num_tokens as the first element, allowing it to travel through
the fwd→bwd boundary without needing extra saved tensors or symbolic expressions.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from torchtitan.tools.logging import logger

GLOBAL_BLOCK_SIZE = 1024


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def _paged_stash_copy_kernel(
    src_ptr, dst_ptr, num_tokens_ptr, free_list_ptr,
    free_list_head_ptr, free_list_tail_ptr, free_list_capacity_ptr,
    page_record_ptr, overflow_ptr, new_free_list_head_ptr,
    PAGE_SIZE: tl.constexpr, HIDDEN_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)
    num_tokens = tl.load(num_tokens_ptr)
    free_list_head = tl.load(free_list_head_ptr)
    free_list_tail = tl.load(free_list_tail_ptr)
    free_list_capacity = tl.load(free_list_capacity_ptr)
    avail_pages = free_list_tail - free_list_head
    required_pages = tl.cdiv(num_tokens, PAGE_SIZE)
    overflow_detected = avail_pages < required_pages
    if pid == 0 and overflow_detected:
        tl.store(overflow_ptr, 1)
    if overflow_detected:
        return
    token_idx = pid
    while token_idx < num_tokens:
        page_slot = token_idx // PAGE_SIZE
        token_in_page = token_idx % PAGE_SIZE
        free_list_idx = (free_list_head + page_slot) % free_list_capacity
        page_id = tl.load(free_list_ptr + free_list_idx)
        if token_in_page == 0:
            tl.store(page_record_ptr + page_slot, page_id)
        dst_token_idx = page_id * PAGE_SIZE + token_in_page
        elements_per_thread = HIDDEN_SIZE // BLOCK_SIZE
        need_mask = (HIDDEN_SIZE % BLOCK_SIZE) != 0
        num_iters = elements_per_thread + (1 if need_mask else 0)
        token_idx_i64 = token_idx.to(tl.int64)
        dst_token_idx_i64 = dst_token_idx.to(tl.int64)
        src_base = src_ptr + token_idx_i64 * HIDDEN_SIZE
        dst_base = dst_ptr + dst_token_idx_i64 * HIDDEN_SIZE
        if need_mask:
            for iter in range(num_iters):
                hidden_offsets = tl.arange(0, BLOCK_SIZE) + iter * BLOCK_SIZE
                hidden_mask = hidden_offsets < HIDDEN_SIZE
                data = tl.load(src_base + hidden_offsets, mask=hidden_mask, other=0)
                tl.store(dst_base + hidden_offsets, data, mask=hidden_mask)
        else:
            for iter in range(elements_per_thread):
                hidden_offsets = tl.arange(0, BLOCK_SIZE) + iter * BLOCK_SIZE
                data = tl.load(src_base + hidden_offsets)
                tl.store(dst_base + hidden_offsets, data)
        token_idx += num_blocks
    if pid == 0:
        new_head = free_list_head + required_pages
        tl.store(new_free_list_head_ptr, new_head)


@triton.jit
def _paged_stash_pop_kernel(
    src_ptr, dst_ptr, num_tokens_ptr, page_record_ptr,
    free_list_ptr, free_list_head_ptr, free_list_tail_ptr,
    free_list_capacity_ptr, new_free_list_tail_ptr,
    PAGE_SIZE: tl.constexpr, HIDDEN_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_blocks = tl.num_programs(axis=0)
    num_tokens = tl.load(num_tokens_ptr)
    free_list_tail = tl.load(free_list_tail_ptr)
    free_list_capacity = tl.load(free_list_capacity_ptr)
    token_idx = pid
    while token_idx < num_tokens:
        page_slot = token_idx // PAGE_SIZE
        token_in_page = token_idx % PAGE_SIZE
        page_id = tl.load(page_record_ptr + page_slot)
        src_token_idx = page_id * PAGE_SIZE + token_in_page
        elements_per_thread = HIDDEN_SIZE // BLOCK_SIZE
        need_mask = (HIDDEN_SIZE % BLOCK_SIZE) != 0
        num_iters = elements_per_thread + (1 if need_mask else 0)
        src_token_idx_i64 = src_token_idx.to(tl.int64)
        token_idx_i64 = token_idx.to(tl.int64)
        src_base = src_ptr + src_token_idx_i64 * HIDDEN_SIZE
        dst_base = dst_ptr + token_idx_i64 * HIDDEN_SIZE
        if need_mask:
            for iter in range(num_iters):
                hidden_offsets = tl.arange(0, BLOCK_SIZE) + iter * BLOCK_SIZE
                hidden_mask = hidden_offsets < HIDDEN_SIZE
                data = tl.load(src_base + hidden_offsets, mask=hidden_mask, other=0)
                tl.store(dst_base + hidden_offsets, data, mask=hidden_mask)
        else:
            for iter in range(elements_per_thread):
                hidden_offsets = tl.arange(0, BLOCK_SIZE) + iter * BLOCK_SIZE
                data = tl.load(src_base + hidden_offsets)
                tl.store(dst_base + hidden_offsets, data)
        if token_in_page == 0:
            write_idx = (free_list_tail + page_slot) % free_list_capacity
            tl.store(free_list_ptr + write_idx, page_id)
        token_idx += num_blocks
    if pid == 0:
        required_pages = tl.cdiv(num_tokens, PAGE_SIZE)
        new_tail = free_list_tail + required_pages
        tl.store(new_free_list_tail_ptr, new_tail)


# ---------------------------------------------------------------------------
# PagedStashBuffer — pre-allocated paged memory pool
# ---------------------------------------------------------------------------


class PagedStashBuffer:
    """Pre-allocated paged memory pool for stashing activations.

    Uses a flat 2D buffer [total_tokens, hidden_size] with a circular free list
    managed by unwrapped head/tail pointers.

    Args:
        num_tokens: Upper bound on tokens to store.
        hidden_size: Size of the hidden dimension.
        page_size: Number of tokens per page.
        device: Device for the buffer ('cuda' or 'cpu').
        overflow: Shared int64 tensor, set to 1 on OOM.
        dtype: Data type for the buffer.
    """

    def __init__(
        self,
        num_tokens: int,
        hidden_size: int,
        page_size: int,
        device: str | torch.device,
        overflow: torch.Tensor,
        dtype: torch.dtype,
    ):
        self.hidden_size = hidden_size
        self.page_size = page_size
        self.num_pages = (num_tokens + page_size - 1) // page_size
        self.total_tokens = self.num_pages * page_size
        self.dtype = dtype
        self.device = device
        self.overflow = overflow  # shared across buffers

        # Track (module, prefix) pairs for updating attrs after resize
        self._registered_modules: list[tuple] = []

        if str(device) == "cpu":
            self.buffer = torch.empty(
                (self.total_tokens, hidden_size),
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            )
        else:
            self.buffer = torch.empty(
                (self.total_tokens, hidden_size),
                dtype=dtype,
                device=device,
            )

        # Circular free list with unwrapped head/tail pointers
        self.free_list = torch.arange(
            self.num_pages, dtype=torch.int64, device=device
        )
        self.free_list_head = torch.zeros(1, dtype=torch.int64, device=device)
        self.free_list_tail = self.num_pages * torch.ones(
            1, dtype=torch.int64, device=device
        )
        self.free_list_capacity = self.num_pages * torch.ones(
            1, dtype=torch.int64, device=device
        )

    def __repr__(self):
        return (
            f"PagedStashBuffer(num_pages={self.num_pages}, page_size={self.page_size}, "
            f"hidden_size={self.hidden_size}, device={self.device}, dtype={self.dtype})"
        )

    def reset(self):
        """Reset free list to full capacity. Called per training step."""
        self.free_list.copy_(
            torch.arange(self.num_pages, dtype=torch.int64, device=self.device)
        )
        self.free_list_head.zero_()
        self.free_list_tail.fill_(self.num_pages)

    def resize(self, new_num_tokens: int) -> None:
        """Resize buffer to new capacity.

        Must be called between CUDAGraph warmup (iter 1) and capture (iter 2)
        so that ``get_attr`` nodes in the FX graph resolve to the new tensors.
        """
        old_pages = self.num_pages
        old_tokens = self.total_tokens
        self.num_pages = (new_num_tokens + self.page_size - 1) // self.page_size
        self.total_tokens = self.num_pages * self.page_size

        if str(self.device) == "cpu":
            self.buffer = torch.empty(
                (self.total_tokens, self.hidden_size),
                dtype=self.dtype,
                device="cpu",
                pin_memory=True,
            )
        else:
            self.buffer = torch.empty(
                (self.total_tokens, self.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )

        self.free_list = torch.arange(
            self.num_pages, dtype=torch.int64, device=self.device
        )
        self.free_list_head = torch.zeros(
            1, dtype=torch.int64, device=self.device
        )
        self.free_list_tail = self.num_pages * torch.ones(
            1, dtype=torch.int64, device=self.device
        )
        self.free_list_capacity = self.num_pages * torch.ones(
            1, dtype=torch.int64, device=self.device
        )

        # Update registered module attributes so get_attr resolves to new tensors
        for module, prefix in self._registered_modules:
            module.register_buffer(f"{prefix}_buffer", self.buffer)
            module.register_buffer(f"{prefix}_free_list", self.free_list)
            module.register_buffer(f"{prefix}_free_list_head", self.free_list_head)
            module.register_buffer(f"{prefix}_free_list_tail", self.free_list_tail)
            module.register_buffer(
                f"{prefix}_free_list_capacity", self.free_list_capacity
            )

        logger.info(
            "Resized PagedStashBuffer(hidden=%d, dtype=%s): "
            "%d -> %d pages (%d -> %d tokens)",
            self.hidden_size,
            self.dtype,
            old_pages,
            self.num_pages,
            old_tokens,
            self.total_tokens,
        )


# ---------------------------------------------------------------------------
# create_paged_buffers — factory
# ---------------------------------------------------------------------------


def create_paged_buffers(model, ac_config, *, max_tokens):
    """Create paged stash buffers for MoE expert activations.

    Scans the model for GroupedExperts modules, counts stash ops per
    (dtype, hidden_size) key, and creates one PagedStashBuffer per key
    sized to the number of dynamic tensors that will be stashed.

    Per GroupedExperts module, ``_run_experts_grouped_mm`` produces 5
    dynamic-shaped activations that the paged stash pass force-saves:

      - x.bf16()           [tokens, dim]         (key: dtype, dim)       × 1
      - gmm1 output        [tokens, hidden_dim]  (key: dtype, hidden_dim)
      - silu output        [tokens, hidden_dim]  (key: dtype, hidden_dim)
      - gmm2 output        [tokens, hidden_dim]  (key: dtype, hidden_dim)
      - h = silu * gmm2    [tokens, hidden_dim]  (key: dtype, hidden_dim) × 4

    So 4 ops contribute to (dtype, hidden_dim) and 1 op to (dtype, dim) per module.

    Args:
        model: The transformer model to scan for GroupedExperts.
        ac_config: Activation checkpoint config with paged stash settings.
        max_tokens: Upper bound on tokens routed to experts per step
            (batch_size * seq_len * top_k).

    Returns:
        Tuple of (buffers, overflow) where buffers is a dict mapping
        (dtype, hidden_size) to PagedStashBuffer and overflow is the shared
        overflow flag tensor. Returns (None, None) if no GroupedExperts found.
    """
    from collections import defaultdict

    from torchtitan.models.common.moe.moe import GroupedExperts

    device = getattr(ac_config, "paged_stash_buffer_device", "cuda")
    page_size = getattr(ac_config, "paged_stash_page_size", 64)
    buffer_size_factor = getattr(ac_config, "paged_stash_buffer_size_factor", 1.1)

    # Count stash ops per (dtype, hidden_size) key across all GroupedExperts modules.
    ops_per_key: dict[tuple[torch.dtype, int], int] = defaultdict(int)
    num_expert_modules = 0
    for _fqn, mod in model.named_modules():
        if isinstance(mod, GroupedExperts):
            num_expert_modules += 1
            # w1 shape: [num_experts, hidden_dim, dim]
            # 4 dynamic activations have hidden_dim as last dim:
            #   gmm1 out, silu out, gmm2 out, h = silu * gmm2
            ops_per_key[(mod.w1.dtype, mod.w1.shape[-2])] += 4
            # 1 dynamic activation has dim as last dim: x.bf16()
            ops_per_key[(mod.w1.dtype, mod.w1.shape[-1])] += 1

    if not ops_per_key:
        logger.warning("No GroupedExperts found; no paged stash buffers created.")
        return None, None

    # Create buffers sized to actual ops per key.
    overflow = torch.zeros(1, dtype=torch.int64, device=device)
    buffers = {}
    _rank0 = torch.distributed.is_initialized() and torch.distributed.get_rank() == 0
    if _rank0:
        logger.debug("create_paged_buffers: max_tokens=%d, buffer_size_factor=%.2f", max_tokens, buffer_size_factor)
    for (dtype, hidden_size), num_ops in ops_per_key.items():
        scaled_max = int(max_tokens * buffer_size_factor * num_ops)
        buffers[dtype, hidden_size] = PagedStashBuffer(
            scaled_max, hidden_size, page_size, device, overflow, dtype
        )
        if _rank0:
            logger.debug("create_paged_buffers: key=(dtype=%s, hidden_size=%d), ops_per_key=%d, scaled_max_tokens=%d, num_pages=%d", dtype, hidden_size, num_ops, scaled_max, buffers[dtype, hidden_size].num_pages)

    logger.info(
        "Created %d paged stash buffers (max_tokens=%d, num_expert_modules=%d, "
        "ops_per_key=%s, page_size=%d, device=%s)",
        len(buffers),
        max_tokens,
        num_expert_modules,
        dict(ops_per_key),
        page_size,
        device,
    )

    return buffers, overflow


# ---------------------------------------------------------------------------
# PagedStashObserver: buffer sizing via observation
# (mirrors Megatron's PagedStashManager capture-phase logic)
# ---------------------------------------------------------------------------


class PagedStashObserver:
    """Observe actual token counts during the observation iteration for buffer sizing.

    Mirrors Megatron's ``PagedStashManager`` capture-phase logic: tracks actual and
    avg token counts per ``(dtype, hidden_size)`` key using increment (on_copy) /
    decrement (on_pop) counters.  The high-water mark determines buffer allocation.

    The ``status`` state machine mirrors Megatron's ``paged_stash_reset``:
    - ``'begin'``: initial state, observation not yet started
    - ``'capture'``: observation active, ``.item()`` calls record token counts
    - ``'captured'``: observation complete, buffers allocated
    """

    def __init__(self):
        self.status = "begin"

        # Mirrors Megatron's temp_tokens_across_vp_stages (running count)
        self.temp_tokens_across_vp_stages: dict[tuple, int] = {}
        # Mirrors Megatron's max_tokens_across_vp_stages (high-water mark, actual)
        self.max_tokens_across_vp_stages: dict[tuple, int] = {}
        # Mirrors Megatron's temp_avg_tokens_across_vp_stages (running count, avg)
        self.temp_avg_tokens_across_vp_stages: dict[tuple, int] = {}
        # Mirrors Megatron's max_avg_tokens_across_vp_stages (high-water mark, avg)
        self.max_avg_tokens_across_vp_stages: dict[tuple, int] = {}

    def on_copy(
        self,
        hidden_size: int,
        dtype: torch.dtype,
        num_tokens_tensor: Tensor,
        avg_num_tokens: int | None,
    ) -> None:
        """Called from ``paged_stash.copy``.  Mirrors Megatron's ``on_save_for_backward``."""
        if self.status != "capture":
            return
        key = (dtype, hidden_size)
        actual_num_tokens = num_tokens_tensor.to(torch.int64).item()

        if key not in self.temp_tokens_across_vp_stages:
            self.temp_tokens_across_vp_stages[key] = 0
            self.max_tokens_across_vp_stages[key] = 0
            self.temp_avg_tokens_across_vp_stages[key] = 0
            self.max_avg_tokens_across_vp_stages[key] = 0

        self.temp_tokens_across_vp_stages[key] += actual_num_tokens
        self.max_tokens_across_vp_stages[key] = max(
            self.max_tokens_across_vp_stages[key],
            self.temp_tokens_across_vp_stages[key],
        )

        if avg_num_tokens is not None and avg_num_tokens > 0:
            self.temp_avg_tokens_across_vp_stages[key] += avg_num_tokens
            self.max_avg_tokens_across_vp_stages[key] = max(
                self.max_avg_tokens_across_vp_stages[key],
                self.temp_avg_tokens_across_vp_stages[key],
            )

    def on_pop(
        self,
        hidden_size: int,
        dtype: torch.dtype,
        num_tokens_tensor: Tensor,
        avg_num_tokens: int | None,
    ) -> None:
        """Called from ``paged_stash.pop``.  Mirrors Megatron's ``on_get_saved_tensor``."""
        if self.status != "capture":
            return
        key = (dtype, hidden_size)
        actual_num_tokens = num_tokens_tensor.to(torch.int64).item()

        if key in self.temp_tokens_across_vp_stages:
            self.temp_tokens_across_vp_stages[key] -= actual_num_tokens
        if (
            avg_num_tokens is not None
            and avg_num_tokens > 0
            and key in self.temp_avg_tokens_across_vp_stages
        ):
            self.temp_avg_tokens_across_vp_stages[key] -= avg_num_tokens

    def allocate_stash_buffers(
        self,
        buffers: list,
        stash_buffer_size_factor_cuda: float,
    ) -> None:
        """Resize buffers from observed peak.  Mirrors Megatron's ``allocate_stash_buffers``.

        Sign convention (mirrors Megatron):
        - positive ``stash_buffer_size_factor_cuda``: use avg-based peak (default)
        - negative: use actual-based peak (conservative)
        """
        cuda_factor = stash_buffer_size_factor_cuda

        if cuda_factor >= 0:
            max_tokens_dict = self.max_avg_tokens_across_vp_stages
            cuda_scale = cuda_factor
        else:
            max_tokens_dict = self.max_tokens_across_vp_stages
            cuda_scale = -cuda_factor

        # Fallback: if avg dict empty, use actual
        if not max_tokens_dict:
            max_tokens_dict = self.max_tokens_across_vp_stages

        for buf in buffers:
            key = (buf.dtype, buf.hidden_size)
            if key in max_tokens_dict:
                num_tokens = int(max_tokens_dict[key] * cuda_scale)
                buf.resize(num_tokens)

    def paged_stash_reset(self) -> None:
        """Transition state machine.  Mirrors Megatron's ``paged_stash_reset``."""
        if self.status == "begin":
            self.status = "capture"
        elif self.status == "capture":
            self.status = "captured"


_observer = PagedStashObserver()


def _block_size(hidden_size: int) -> int:
    """Compute the block size for Triton kernels, capped at hidden_size and rounded to power of 2."""
    import triton

    return min(GLOBAL_BLOCK_SIZE, triton.next_power_of_2(hidden_size))


# ---------------------------------------------------------------------------
# paged_stash::copy — pack a tensor into the paged buffer
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "paged_stash::copy",
    mutates_args=("free_list_head", "overflow"),
)
def paged_stash_copy(
    tensor: Tensor,
    buffer: Tensor,
    free_list: Tensor,
    free_list_head: Tensor,
    free_list_tail: Tensor,
    free_list_capacity: Tensor,
    overflow: Tensor,
    page_size: int,
    hidden_size: int,
    num_tokens_tensor: Tensor,
    avg_num_tokens: int = 0,
) -> tuple[Tensor, Tensor]:
    """Pack tensor into paged buffer using actual token count.

    Args:
        num_tokens_tensor: GPU scalar (int32/int64) with the actual number of
            tokens to stash, from ``offsets[-1]`` (= ``tokens_per_expert.sum()``).
            Only this many rows are copied; padding rows are skipped.

    Returns (page_record, new_head).
    page_record format: [num_tokens, page_id_0, ..., page_id_{N-1}]
    where page_record is sized to the worst-case (tensor shape) for CUDA graph
    static shapes, but only ceil(actual_tokens / page_size) page slots are used.
    """
    flat = tensor.reshape(-1, hidden_size).contiguous()
    max_num_tokens = flat.shape[0]  # oversized capacity (for page_record sizing + grid)
    max_num_pages = (max_num_tokens + page_size - 1) // page_size

    # page_record sized to worst-case for CUDA graph static shapes
    page_record = torch.empty(
        max_num_pages + 1, dtype=torch.int64, device=flat.device
    )
    # Store actual token count in page_record[0] for pop to read
    num_tokens_i64 = num_tokens_tensor.reshape(1).to(torch.int64)
    page_record[0:1].copy_(num_tokens_i64)
    page_ids = page_record[1:]  # view into page IDs portion

    new_free_list_head = free_list_head.clone()

    # Grid sized to max for CUDA graph static launch config;
    # kernel loop bounds on actual num_tokens via num_tokens_i64 pointer
    num_blocks = max(min(max_num_tokens, 2048), 1)
    grid = (num_blocks,)
    _paged_stash_copy_kernel[grid](
        flat,
        buffer,
        num_tokens_i64,
        free_list,
        free_list_head,
        free_list_tail,
        free_list_capacity,
        page_ids,
        overflow,
        new_free_list_head,
        PAGE_SIZE=page_size,
        HIDDEN_SIZE=hidden_size,
        BLOCK_SIZE=_block_size(hidden_size),
    )
    # Update the head pointer in-place
    free_list_head.copy_(new_free_list_head)

    # Observation recording (mirrors Megatron's on_save_for_backward)
    _observer.on_copy(
        hidden_size,
        tensor.dtype,
        num_tokens_tensor,
        avg_num_tokens if avg_num_tokens > 0 else None,
    )

    return page_record, new_free_list_head


@paged_stash_copy.register_fake
def paged_stash_copy_fake(
    tensor: Tensor,
    buffer: Tensor,
    free_list: Tensor,
    free_list_head: Tensor,
    free_list_tail: Tensor,
    free_list_capacity: Tensor,
    overflow: Tensor,
    page_size: int,
    hidden_size: int,
    num_tokens_tensor: Tensor,
    avg_num_tokens: int = 0,
) -> tuple[Tensor, Tensor]:
    flat = tensor.reshape(-1, hidden_size)
    # page_record sized to worst-case (tensor shape) for CUDA graph static shapes
    max_num_tokens = flat.shape[0]
    max_num_pages = (max_num_tokens + page_size - 1) // page_size
    page_record = tensor.new_empty(max_num_pages + 1, dtype=torch.int64)
    new_head = tensor.new_empty(1, dtype=torch.int64)
    return page_record, new_head


# ---------------------------------------------------------------------------
# paged_stash::pop — restore a tensor from the paged buffer
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "paged_stash::pop",
    mutates_args=("free_list_tail",),
)
def paged_stash_pop(
    page_record: Tensor,
    buffer: Tensor,
    free_list: Tensor,
    free_list_head: Tensor,
    free_list_tail: Tensor,
    free_list_capacity: Tensor,
    page_size: int,
    hidden_size: int,
    dtype: torch.dtype,
    avg_num_tokens: int = 0,
) -> Tensor:
    """Pop tensor from paged buffer. Returns reconstructed 2D tensor.

    page_record format: [num_tokens, page_id_0, ..., page_id_{N-1}]
    """
    assert page_record.dtype == torch.int64, (
        f"paged_stash.pop: expected page_record dtype=int64, got {page_record.dtype}"
    )
    # num_tokens is encoded in page_record[0], and num_pages = len(page_record) - 1
    # Derive num_tokens from the page_record shape to avoid D2H sync during CUDA graph capture
    num_pages = page_record.shape[0] - 1
    num_tokens = num_pages * page_size  # upper bound; actual count is in page_record[0]
    page_ids = page_record[1:]

    flat_out = torch.empty(
        (num_tokens, hidden_size), dtype=dtype, device=buffer.device
    )
    # Use page_record[0:1] directly as the num_tokens tensor (already on GPU)
    num_tokens_tensor = page_record[0:1]
    new_free_list_tail = free_list_tail.clone()

    num_blocks = max(min(num_tokens, 2048), 1)
    grid = (num_blocks,)
    _paged_stash_pop_kernel[grid](
        buffer,
        flat_out,
        num_tokens_tensor,
        page_ids,
        free_list,
        free_list_head,
        free_list_tail,
        free_list_capacity,
        new_free_list_tail,
        PAGE_SIZE=page_size,
        HIDDEN_SIZE=hidden_size,
        BLOCK_SIZE=_block_size(hidden_size),
    )
    # Update the tail pointer in-place
    free_list_tail.copy_(new_free_list_tail)

    # Observation recording (mirrors Megatron's on_get_saved_tensor)
    _observer.on_pop(
        hidden_size,
        dtype,
        num_tokens_tensor,
        avg_num_tokens if avg_num_tokens > 0 else None,
    )

    return flat_out


@paged_stash_pop.register_fake
def paged_stash_pop_fake(
    page_record: Tensor,
    buffer: Tensor,
    free_list: Tensor,
    free_list_head: Tensor,
    free_list_tail: Tensor,
    free_list_capacity: Tensor,
    page_size: int,
    hidden_size: int,
    dtype: torch.dtype,
    avg_num_tokens: int = 0,
) -> Tensor:
    # num_tokens is data-dependent (stored in page_record[0]).
    # Use create_unbacked_symint for the dynamic first dimension.
    ctx = torch.library.get_ctx()
    num_tokens = ctx.create_unbacked_symint()
    return buffer.new_empty((num_tokens, hidden_size), dtype=dtype)
