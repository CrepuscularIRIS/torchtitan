# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph-based paged SAC passes.

Joint-graph pass (before min-cut partitioning):

- apply_paged_stash_pass: Identifies paged-stash-eligible activations in the
  joint fwd+bwd graph using the ``paged_stash`` annotation from ``annotate_fn``,
  inserts ``paged_stash.copy`` + ``ao.wait_tensor`` after the forward producer,
  and ``paged_stash.pop`` + ``ao.wait_tensor`` before the backward consumers.
  Backward consumers are redirected to read from the pop output instead of the
  original activation.  After this pass, min-cut sees only the compact
  ``page_record`` (int64 handle) crossing the fwd→bwd boundary — the large
  activation has no backward users and is freed after forward.

  This follows the same architecture as PR #2879's ``cpu_offload_pass``:
  - Joint-graph pass applied before partitioning
  - ``seq_nr`` metadata for fwd/bwd classification
  - ``ao.wait_tensor`` for stream synchronization
  - ``replace_input_with`` for backward consumer redirection
  - Stream management is imperative inside the op implementations
    (ao's ``_get_or_create_transfer_stream`` / ``_register_wait``)

Pre-partition utility passes:

- apply_paged_sac_pass: Counts and logs annotated nodes (diagnostic only).
- apply_sac_grouped_mm_pass: SAC with ``_grouped_mm`` added to the save list.

The region annotation is applied via ``torch.fx.traceback.annotate_fn`` on the
target function (e.g., ``_run_experts_grouped_mm``), which sets
``node.meta["custom"]["paged_stash"]`` on every FX node traced inside that
function.
"""

import operator

import torch
import torch.fx as fx
from torch.fx.experimental.proxy_tensor import is_sym_node
from torch.utils.checkpoint import CheckpointPolicy

# Side-effect import: registers ao::wait_tensor custom op
import torch._functorch._activation_offloading.offload_ops as offload_ops  # noqa: F401

from torchtitan.distributed.activation_checkpoint import _get_save_ops
from torchtitan.experiments.graph_trainer.passes import apply_sac_pass
from torchtitan.tools.logging import logger

# Import to ensure paged_stash::copy/pop custom ops are registered
from . import paged_stash_ops  # noqa: F401
from .paged_stash_ops import PagedStashBuffer

# Extend the default SAC save ops with _grouped_mm for fair baseline comparison.
_SAC_SAVE_OPS_WITH_GROUPED_MM = _get_save_ops() | {
    torch.ops.aten._grouped_mm.default,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_paged_stash_annotation(node: fx.Node) -> bool:
    """Check if a node carries the ``paged_stash`` annotation.

    ``annotate_fn({"paged_stash": True})`` stores the annotation in
    ``node.meta["custom"]["paged_stash"]``.
    """
    custom = node.meta.get("custom")
    if isinstance(custom, dict):
        return custom.get("paged_stash", False)
    return False


def _has_dynamic_first_dim(node: fx.Node) -> bool:
    """Check if a node's first dimension is dynamic (SymInt)."""
    val = node.meta.get("val")
    if val is None or not hasattr(val, "shape") or len(val.shape) < 1:
        return False
    return isinstance(val.shape[0], torch.SymInt)


# ---------------------------------------------------------------------------
# Pre-partition utility passes
# ---------------------------------------------------------------------------


def apply_sac_grouped_mm_pass(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Apply SAC with _grouped_mm added to the save list.

    Same as ``apply_sac_pass`` but also marks ``_grouped_mm`` outputs as
    MUST_SAVE.  This provides a fair baseline for comparing regular tensor
    saves (which fragment the allocator due to dynamic MoE token counts)
    against paged stash saves.

    Use ``--compile.joint_passes apply_sac_grouped_mm`` for the baseline.
    Compare against ``--compile.joint_passes apply_sac apply_paged_sac``
    for the paged stash variant.
    """
    return apply_sac_pass(gm, op_list_to_save=_SAC_SAVE_OPS_WITH_GROUPED_MM)


def apply_paged_sac_pass(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Count and log paged-stash-annotated nodes (diagnostic only).

    Scans the joint graph for nodes carrying the ``paged_stash`` annotation.
    This pass only counts and logs — it does NOT modify the graph.  The
    actual surgery happens in ``apply_paged_stash_pass``.

    Use ``--compile.joint_passes apply_sac apply_paged_sac`` to run both.
    """
    paged_stash_count = 0
    _rank0 = torch.distributed.is_initialized() and torch.distributed.get_rank() == 0

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if _has_paged_stash_annotation(node):
            paged_stash_count += 1
            if _rank0:
                val = node.meta.get("val")
                shape_str = str(tuple(val.shape)) if val is not None and hasattr(val, "shape") else "N/A"
                logger.debug(
                    "[PAGED_STASH] ANNOTATED: name=%s, target=%s, shape=%s, symint=%s",
                    node.name, node.target, shape_str, _has_dynamic_first_dim(node),
                )

    logger.info(
        "Applied paged SAC annotation pass (%d annotated nodes found)",
        paged_stash_count,
    )
    return gm


# ---------------------------------------------------------------------------
# Helper: find actual token count for _grouped_mm outputs
# ---------------------------------------------------------------------------


def _find_num_tokens_node(
    gm: fx.GraphModule,
    fwd_node: fx.Node,
    insert_before: fx.Node,
    *,
    _cache: dict[int, fx.Node] | None = None,
) -> fx.Node | None:
    """Find a graph node representing the actual token count for a saved tensor.

    For ``_grouped_mm`` outputs, the offsets tensor (from ``cumsum``) is
    available via ``kwargs["offs"]``.  ``offsets[-1]`` equals
    ``tokens_per_expert.sum()`` — the actual (non-padded) token count.

    For derived tensors (transposes, casts, etc.), walks backward through
    the producer chain to find the originating ``_grouped_mm`` node.

    Returns an int64 scalar node, or None if not found.
    """
    if _cache is None:
        _cache = {}

    if (
        fwd_node.op == "call_function"
        and fwd_node.target == torch.ops.aten._grouped_mm.default
    ):
        offsets_node = fwd_node.kwargs.get("offs")
        if offsets_node is None and len(fwd_node.args) > 2:
            offsets_node = fwd_node.args[2]
        if offsets_node is None or not isinstance(offsets_node, fx.Node):
            return None

        cache_key = id(offsets_node)
        if cache_key in _cache:
            return _cache[cache_key]

        with gm.graph.inserting_before(insert_before):
            total_node = gm.graph.call_function(
                torch.ops.aten.select.int,
                args=(offsets_node, 0, -1),
            )
            offsets_val = offsets_node.meta.get("val")
            if offsets_val is not None:
                total_node.meta["val"] = offsets_val.select(0, -1)
            total_i64 = gm.graph.call_function(
                torch.ops.aten.to.dtype,
                args=(total_node, torch.int64),
            )
            if "val" in total_node.meta:
                total_i64.meta["val"] = total_node.meta["val"].to(torch.int64)
        _cache[cache_key] = total_i64
        return total_i64

    for arg in fwd_node.all_input_nodes:
        result = _find_num_tokens_node(
            gm, arg, insert_before, _cache=_cache
        )
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# Joint-graph pass: fwd/bwd classification
# ---------------------------------------------------------------------------


def _classify_forward_backward(gm: fx.GraphModule):
    """Classify nodes as forward or backward using ``seq_nr`` metadata.

    Follows PR #2879's ``_classify_forward_backward`` pattern: for each
    unique ``seq_nr``, the first encountered node is forward; subsequent
    nodes with the same ``seq_nr`` are backward.
    """
    seq_nr_to_first: dict[int, fx.Node] = {}
    forward_nodes: set[fx.Node] = set()
    backward_nodes: set[fx.Node] = set()
    for node in gm.graph.nodes:
        seq_nr = node.meta.get("seq_nr")
        if seq_nr is None:
            continue
        if seq_nr not in seq_nr_to_first:
            seq_nr_to_first[seq_nr] = node
            forward_nodes.add(node)
        else:
            backward_nodes.add(node)
    return forward_nodes, backward_nodes


# ---------------------------------------------------------------------------
# Joint-graph pass: eligibility check
# ---------------------------------------------------------------------------


def _is_paged_stash_eligible(
    node: fx.Node,
    forward_nodes: set[fx.Node],
    backward_nodes: set[fx.Node],
    paged_buffers: dict[tuple[torch.dtype, int], PagedStashBuffer],
) -> tuple[list[fx.Node], PagedStashBuffer] | None:
    """Check if a forward node is eligible for paged stash.

    Returns ``(bwd_consumers, buffer)`` if eligible, or ``None``.

    A node is eligible when all of:
    1. It is a forward ``call_function`` node.
    2. It carries the ``paged_stash`` annotation (from ``annotate_fn``).
    3. It has real (non-sym) backward consumers.
    4. It has a dynamic SymInt first dimension.
    5. Its ``(dtype, shape[-1])`` matches a pre-allocated paged buffer.
    """
    if node.op != "call_function":
        return None
    if node not in forward_nodes:
        return None
    if not _has_paged_stash_annotation(node):
        return None

    bwd_consumers = [u for u in node.users if u in backward_nodes]
    if not bwd_consumers:
        return None
    if all(is_sym_node(u) for u in bwd_consumers):
        return None
    if not _has_dynamic_first_dim(node):
        return None

    val = node.meta.get("val")
    if val is None or not hasattr(val, "shape") or len(val.shape) < 1:
        return None
    key = (val.dtype, val.shape[-1])
    buf = paged_buffers.get(key)
    if buf is None:
        return None

    return bwd_consumers, buf


# ---------------------------------------------------------------------------
# Joint-graph pass: main entry point
# ---------------------------------------------------------------------------


def apply_paged_stash_pass(
    gm: torch.fx.GraphModule,
    paged_buffers: dict[tuple[torch.dtype, int], PagedStashBuffer],
) -> torch.fx.GraphModule:
    """Insert paged_stash.copy/pop + ao.wait_tensor into the joint graph.

    This is the main paged stash pass, following the same architecture as
    PR #2879's ``cpu_offload_pass``:

    1. Classify forward vs backward nodes using ``seq_nr``.
    2. For each eligible forward node (annotated, dynamic, has bwd consumers):
       a. Insert ``paged_stash.copy`` + ``ao.wait_tensor`` after the fwd node.
       b. Insert ``paged_stash.pop`` + ``ao.wait_tensor`` before bwd consumers.
       c. Redirect backward consumers to read from the pop output.
    3. After this pass, min-cut sees ``page_record`` (small int64) crossing
       the boundary, not the large activation.

    Args:
        gm: The joint forward-backward graph module.
        paged_buffers: Dict mapping ``(dtype, hidden_size)`` to
            ``PagedStashBuffer``.

    Returns:
        The modified graph module.
    """
    from .paged_stash_ops import register_paged_stash_buffer

    forward_nodes, backward_nodes = _classify_forward_backward(gm)

    _rank0 = torch.distributed.is_initialized() and torch.distributed.get_rank() == 0

    # Ensure all buffers are registered in the module-level registry.
    # register_paged_stash_buffer is idempotent for the same buffer object.
    buffer_ids: dict[int, int] = {}
    for buf in paged_buffers.values():
        buf_key = id(buf)
        if buf_key not in buffer_ids:
            buffer_ids[buf_key] = register_paged_stash_buffer(buf)

    # Get FakeTensorMode from the joint graph for creating proper metadata
    # on inserted nodes (the partitioner requires FakeTensor, not meta tensors).
    fake_mode = None
    for node in gm.graph.nodes:
        if node.op == "placeholder" and "val" in node.meta:
            v = node.meta["val"]
            if hasattr(v, "fake_mode"):
                fake_mode = v.fake_mode
                break

    def _make_fake(shape, dtype=torch.int64):
        """Create a FakeTensor with the correct mode for the partitioner."""
        if fake_mode is not None:
            with fake_mode:
                return torch.empty(shape, dtype=dtype, device="cuda")
        return torch.empty(shape, dtype=dtype, device="meta")

    # Collect eligible nodes in topological order
    eligible: list[tuple[fx.Node, list[fx.Node], PagedStashBuffer]] = []
    for node in gm.graph.nodes:
        result = _is_paged_stash_eligible(
            node, forward_nodes, backward_nodes, paged_buffers
        )
        if result is not None:
            bwd_consumers, buf = result
            eligible.append((node, bwd_consumers, buf))
            # SAC must have already annotated these nodes. They should be
            # PREFER_RECOMPUTE (expert activations are not in SAC's save
            # list), meaning SAC would recompute them — paged stash saves
            # them via paging instead. If SAC marked them MUST_SAVE, the
            # paged stash is redundant; if SAC hasn't run, something is
            # wrong with pass ordering.
            sac_policy = node.meta.get("recompute")
            assert sac_policy is not None, (
                f"Paged stash eligible node {node.name} has no SAC annotation. "
                f"apply_sac_pass must run before apply_paged_stash_pass."
            )
            if _rank0 and sac_policy != CheckpointPolicy.PREFER_RECOMPUTE:
                logger.warning(
                    "Paged stash node %s has SAC policy %s (expected "
                    "PREFER_RECOMPUTE). Paged stash will override.",
                    node.name, sac_policy,
                )

    if not eligible:
        logger.info("apply_paged_stash_pass: no eligible nodes found")
        return gm

    node_to_index = {n: i for i, n in enumerate(gm.graph.nodes)}
    num_tokens_cache: dict[int, fx.Node] = {}

    for fwd_node, bwd_consumers, buf in eligible:
        buffer_id = buffer_ids[id(buf)]
        val = fwd_node.meta["val"]

        # --- Forward: insert copy + wait_tensor after fwd_node ---

        first_bwd_consumer = min(bwd_consumers, key=lambda n: node_to_index[n])

        # Find actual token count (offsets[-1] from _grouped_mm kwargs).
        # Helper nodes (select, to.dtype) are inserted right after fwd_node.
        num_tokens_node = _find_num_tokens_node(
            gm, fwd_node, fwd_node.next, _cache=num_tokens_cache
        )
        if _rank0:
            found_str = "offsets[-1]" if num_tokens_node is not None else "tensor shape fallback"
            logger.debug("  num_tokens for %s: %s", fwd_node.name, found_str)

        # Walk past any helper nodes that _find_num_tokens_node just created
        # so that our copy node is inserted after them (topological order).
        insert_pt = fwd_node
        cursor = fwd_node.next
        while cursor is not None and cursor.op == "call_function" and cursor.target in (
            torch.ops.aten.select.int, torch.ops.aten.to.dtype,
        ):
            insert_pt = cursor
            cursor = cursor.next

        if num_tokens_node is not None:
            actual_num_tokens = num_tokens_node
        else:
            with gm.graph.inserting_after(fwd_node):
                num_tokens_fallback = gm.graph.call_function(
                    torch.ops.aten.full.default,
                    args=([1], val.shape[0]),
                    kwargs={"dtype": torch.int64, "device": val.device},
                )
                num_tokens_fallback.meta["val"] = _make_fake(1, torch.int64)
            insert_pt = num_tokens_fallback
            actual_num_tokens = num_tokens_fallback

        # Compute fake metadata for the inserted nodes
        flat_shape = val.reshape(-1, buf.hidden_size)
        max_num_tokens = flat_shape.shape[0]
        max_num_pages = (max_num_tokens + buf.page_size - 1) // buf.page_size

        page_record_fake = _make_fake(max_num_pages + 2, torch.int64)
        new_head_fake = _make_fake(2, torch.int64)
        pop_fake = _make_fake((max_num_tokens, buf.hidden_size), val.dtype)

        with gm.graph.inserting_after(insert_pt):
            copy_node = gm.graph.call_function(
                torch.ops.paged_stash.copy,
                args=(fwd_node, buf.page_size, buf.hidden_size,
                      actual_num_tokens, buffer_id),
            )
            copy_node.meta["val"] = (page_record_fake, new_head_fake)
            # MUST_SAVE so the partitioner doesn't treat it as impure
            copy_node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            copy_node.meta["ac_graph_id"] = 0

        with gm.graph.inserting_after(copy_node):
            page_record_node = gm.graph.call_function(
                operator.getitem, args=(copy_node, 0),
            )
            page_record_node.meta["val"] = page_record_fake
            page_record_node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            page_record_node.meta["ac_graph_id"] = 0

        with gm.graph.inserting_after(page_record_node):
            # keepalive=fwd_node extends the activation's lifetime past the
            # async Triton copy on the transfer stream (same pattern as
            # ao.wait_tensor(offload_result, gpu_tensor) in cpu_offload_pass)
            wait_copy_node = gm.graph.call_function(
                torch.ops.ao.wait_tensor.default,
                args=(page_record_node, fwd_node),
            )
            wait_copy_node.meta["val"] = page_record_fake
            # ao.wait_tensor has has_side_effect, which the partitioner treats
            # as impure. MUST_SAVE bypasses the impure-op assertion.
            wait_copy_node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            wait_copy_node.meta["ac_graph_id"] = 0

        # --- Backward: insert pop + wait_tensor before first bwd consumer ---

        with gm.graph.inserting_before(first_bwd_consumer):
            pop_node = gm.graph.call_function(
                torch.ops.paged_stash.pop,
                args=(wait_copy_node, buf.page_size, buf.hidden_size,
                      val.dtype, buffer_id),
            )
            pop_node.meta["val"] = pop_fake

        with gm.graph.inserting_after(pop_node):
            wait_pop_node = gm.graph.call_function(
                torch.ops.ao.wait_tensor.default,
                args=(pop_node,),
            )
            wait_pop_node.meta["val"] = pop_fake

        if len(val.shape) > 2:
            with gm.graph.inserting_after(wait_pop_node):
                restore_node = gm.graph.call_function(
                    torch.ops.aten.reshape.default,
                    args=(wait_pop_node, list(val.shape)),
                )
                restore_node.meta["val"] = val
        else:
            restore_node = wait_pop_node

        # Redirect backward consumers from fwd_node to the restored tensor
        for bwd_user in bwd_consumers:
            bwd_user.replace_input_with(fwd_node, restore_node)

    # TODO: Add scheduling optimizations (defer copy waits, prefetch pops)
    # for compute-copy overlap. Both this pass and PR #2879's cpu_offload_pass
    # place ao.wait_tensor adjacent to the copy/offload — the compute stream
    # blocks immediately. Deferring the wait to a later point in the forward
    # graph (post-partition) would allow the Triton copy kernel to overlap
    # with subsequent compute. See paged_stashing_guide.md for details.

    gm.graph.lint()
    gm.recompile()

    logger.info(
        "Inserted paged stash ops: %d copy + wait in fwd, %d pop + wait in bwd",
        len(eligible), len(eligible),
    )
    return gm
