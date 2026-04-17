# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph-based paged SAC passes.

Pre-partition pass (joint graph, before min-cut):
- apply_paged_sac_pass: Force-saves all dynamic-shaped activations from annotated
  regions (marked with ``annotate_fn({"paged_stash": True})``).

Post-partition pass (after min-cut splits fwd/bwd):
- choose_paged_stash_sets: Identifies saved tensors eligible for paged stash using
  the ``paged_stash`` annotation + dynamic SymInt first dimension + buffer key match.
- stash_chosen_sets: Inserts paged_stash.copy/pop ops into the separated fwd/bwd graphs.
- enable_paged_stash: Orchestrates choose + stash (analogous to enable_activation_offloading).
- make_paged_stash_partition_fn: Wraps min_cut_rematerialization_partition to run
  enable_paged_stash after the split.

The region annotation is applied via ``torch.fx.traceback.annotate_fn`` on the target
function (e.g., ``_run_experts_grouped_mm``), which sets ``node.meta["custom"]["paged_stash"]``
on every FX node traced inside that function. This is the graph-level equivalent of
Megatron's ``saved_tensors_hooks`` interception inside the expert region.

The post-partition design follows PyTorch's activation offloading pattern
(torch/_functorch/_activation_offloading/activation_offloading.py):
- can_paged_stash ↔ can_offload
- choose_paged_stash_sets ↔ choose_offload_sets
- stash_chosen_sets ↔ offload_chosen_sets
"""

import operator

import torch
import torch.fx as fx
from torch._functorch.partitioners import (
    classify_nodes,
    is_sym_node,
    min_cut_rematerialization_partition,
)
from torch.utils._ordered_set import OrderedSet
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.passes import (
    apply_sac_pass,
    DEFAULT_SAC_SAVE_OPS,
)
from torchtitan.tools.logging import logger

# Import to ensure custom ops are registered
from . import paged_stash_ops  # noqa: F401
from .paged_stash_ops import PagedStashBuffer

# Extend the default SAC save ops with _grouped_mm for fair baseline comparison.
_SAC_SAVE_OPS_WITH_GROUPED_MM = DEFAULT_SAC_SAVE_OPS | {
    torch.ops.aten._grouped_mm.default,
}


# ---------------------------------------------------------------------------
# Helper: check for paged_stash annotation from annotate_fn
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
    """Check if a node's first dimension is dynamic (SymInt).

    MoE expert activations have a dynamic token dimension (``shape[0]``
    is a ``SymInt``) because the number of tokens dispatched to each
    expert varies per-batch.  Static-shaped tensors (weights, buffers)
    have concrete ``int`` dimensions.

    When HybridEP's ``_dispatch_fake`` returns symbolic shapes via
    ``ctx.new_dynamic_size()``, this check identifies the dynamic
    activations that need paged stashing.  When shapes are concrete
    (current behavior), this returns ``False`` and the annotation-based
    selection is the sole gate.
    """
    val = node.meta.get("val")
    if val is None or not hasattr(val, "shape") or len(val.shape) < 1:
        return False
    return isinstance(val.shape[0], torch.SymInt)


def _has_any_dynamic_dim(node: fx.Node) -> bool:
    """Check if a node has any dynamic (SymInt) dimension.

    A tensor with at least one ``SymInt`` in its shape is a dynamic
    activation derived from the dispatch output — not a weight or
    parameter (which always have concrete shapes).  This is more
    general than ``_has_dynamic_first_dim`` and correctly identifies
    dynamic activations regardless of their rank (2D, 3D, etc.).
    """
    val = node.meta.get("val")
    if val is None or not hasattr(val, "shape") or len(val.shape) < 1:
        return False
    return any(isinstance(d, torch.SymInt) for d in val.shape)


# ---------------------------------------------------------------------------
# Pre-partition passes (joint graph)
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
    """Identify paged-stash-annotated activations in the joint graph.

    Scans the joint graph for nodes that carry the ``paged_stash`` annotation
    (set by ``annotate_fn({"paged_stash": True})`` on the target function).
    This pass only **counts and logs** annotated nodes — it does NOT set
    ``MUST_SAVE`` or modify any recompute tags.  SAC's decisions from
    ``apply_sac_pass`` are left completely intact.

    The actual ``MUST_SAVE`` decisions for paged stashing happen later in
    ``_apply_paged_stash_must_save`` (called from ``partition_fn``), which has
    access to ``num_fwd_outputs`` and can perform backward-usage analysis to
    determine which annotated nodes truly need saving.

    Use ``--compile.joint_passes apply_sac apply_paged_sac`` to run both.
    """
    paged_stash_count = 0

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if _has_paged_stash_annotation(node):
            paged_stash_count += 1
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                val = node.meta.get("val")
                shape_str = str(tuple(val.shape)) if val is not None and hasattr(val, "shape") else "N/A"
                has_symint = _has_dynamic_first_dim(node)
                logger.debug("[PAGED_STASH] apply_paged_sac_pass ANNOTATED: name=%s, op=%s, target=%s, shape=%s, symint=%s", node.name, node.op, node.target, shape_str, has_symint)

    logger.info(
        "Applied paged SAC annotation pass (%d annotated nodes found)",
        paged_stash_count,
    )
    return gm


# ---------------------------------------------------------------------------
# Pre-partition: fine-grained MUST_SAVE for dynamic annotated nodes
# ---------------------------------------------------------------------------


def _apply_paged_stash_must_save(
    joint_module: fx.GraphModule,
    num_fwd_outputs: int,
    static_lifetime_input_indices: list[int],
) -> int:
    """Set ``MUST_SAVE`` on annotated nodes that are dynamically shaped and
    needed by the backward computation.

    Called from ``partition_fn`` before ``min_cut_rematerialization_partition``,
    where ``num_fwd_outputs`` is available for backward-usage analysis.

    For each annotated 2D node in the joint graph:

    a. Compute backward usages (users not in the forward subgraph).
    b. Skip if no backward usages (backward doesn't need this tensor).
    c. Skip if all backward usages are sym nodes (backward only needs shape
       info — SAC saves sym nodes separately, no tensor save needed).
    d. If has real tensor backward usages AND has a dynamic SymInt first
       dimension → ``MUST_SAVE`` (paged stash candidate).
    e. If has real tensor backward usages AND static shape → leave SAC's
       decision (MUST_SAVE for expensive ops, PREFER_RECOMPUTE for cheap ops).

    Returns the number of nodes set to ``MUST_SAVE``.
    """
    node_info = classify_nodes(
        joint_module, static_lifetime_input_indices, num_fwd_outputs
    )
    forward_node_names = {n.name for n in node_info.required_fw_nodes}

    _rank0 = torch.distributed.is_initialized() and torch.distributed.get_rank() == 0
    must_save_count = 0
    for node in joint_module.graph.nodes:
        if node.op != "call_function":
            continue
        if not _has_paged_stash_annotation(node):
            continue

        backward_usages = [
            u for u in node.users if u.name not in forward_node_names
        ]

        # (b) No backward usages — backward doesn't need this tensor
        if len(backward_usages) == 0:
            continue

        # (c) All backward usages are sym nodes — only shape info needed
        if all(is_sym_node(u) for u in backward_usages):
            continue

        # (d) Real backward usages + dynamic shape → MUST_SAVE
        if _has_dynamic_first_dim(node):
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            if "ac_graph_id" not in node.meta:
                node.meta["ac_graph_id"] = 0
            must_save_count += 1
            if _rank0:
                val = node.meta.get("val")
                shape_str = (
                    str(tuple(val.shape))
                    if val is not None and hasattr(val, "shape")
                    else "N/A"
                )
                logger.debug(
                    "MUST_SAVE %s (shape=%s, bwd_usages=%d, symint=True)",
                    node.name, shape_str, len(backward_usages),
                )
        else:
            # (e) Real backward usages + static shape → leave SAC's decision
            if _rank0:
                val = node.meta.get("val")
                shape_str = (
                    str(tuple(val.shape))
                    if val is not None and hasattr(val, "shape")
                    else "N/A"
                )
                sac_tag = node.meta.get("recompute", "unset")
                logger.debug(
                    "DEFER %s (shape=%s, bwd_usages=%d, symint=False, sac_tag=%s)",
                    node.name, shape_str, len(backward_usages), sac_tag,
                )

    logger.info(
        "Paged stash MUST_SAVE: %d annotated dynamic nodes with real backward "
        "usages set to MUST_SAVE",
        must_save_count,
    )
    return must_save_count


# ---------------------------------------------------------------------------
# Post-partition: eligibility checks
# ---------------------------------------------------------------------------


def can_paged_stash(
    node: fx.Node,
    fwd_outputs: OrderedSet[fx.Node],
    model_outputs: OrderedSet[fx.Node],
    static_lifetime_input_nodes: OrderedSet[fx.Node],
    paged_buffers: dict[tuple[torch.dtype, int], PagedStashBuffer],
) -> PagedStashBuffer | None:
    """Check if a node is eligible for paged stash.

    Returns the matching ``PagedStashBuffer`` if the node should be
    paged-stashed, or ``None`` if it should be skipped.

    A node is eligible when all of:

    1. Standard gates (from ``can_offload`` in activation offloading):
       in fwd_outputs, not a model output, not a static lifetime input,
       not a getitem.
    2. Carries the ``paged_stash`` annotation (from ``annotate_fn``) OR
       has a dynamic SymInt first dimension.
    3. Buffer key ``(dtype, shape[-1])`` matches a pre-allocated paged buffer.

    We skip the ``is_view`` and ``is_contiguous`` checks from ``can_offload``
    because ``paged_stash.copy`` copies by value (Triton kernel), not by
    reference — views and non-contiguous tensors are handled correctly.
    """
    # 1. Standard gates (from can_offload)
    if node not in fwd_outputs:
        return None
    if node in model_outputs:
        return None
    if node in static_lifetime_input_nodes:
        return None
    if node.target == operator.getitem:
        return None

    # 2. Must carry the paged_stash annotation OR have any dynamic dim
    if not _has_paged_stash_annotation(node) and not _has_any_dynamic_dim(node):
        return None

    # 3. Must be a dynamic activation (has at least one SymInt dim) or a 2D
    #    tensor.  This filters out static weights/parameters (all concrete
    #    dims) that happen to be in the annotated region.  When shapes are
    #    concrete (no SymInt), falls back to the 2D check for compatibility.
    val = node.meta.get("val")
    if val is None or not hasattr(val, "shape") or len(val.shape) < 1:
        return None
    if not _has_any_dynamic_dim(node) and len(val.shape) != 2:
        return None

    # 4. Must match a buffer key
    key = (val.dtype, val.shape[-1])
    return paged_buffers.get(key)


# ---------------------------------------------------------------------------
# Post-partition: choose which saved tensors to paged-stash
# ---------------------------------------------------------------------------


def choose_paged_stash_sets(
    fwd_module: fx.GraphModule,
    num_fwd_outputs: int,
    static_lifetime_input_nodes: OrderedSet[fx.Node],
    paged_buffers: dict[tuple[torch.dtype, int], PagedStashBuffer],
) -> list[tuple[int, fx.Node, PagedStashBuffer]]:
    """Identify saved tensors eligible for paged stash.

    Iterates saved tensors in the forward graph output (everything after
    ``num_fwd_outputs``) and applies ``can_paged_stash`` to each.

    Follows the same pattern as ``choose_offload_sets`` in
    ``torch/_functorch/_activation_offloading/activation_offloading.py``.

    Returns:
        List of ``(saved_idx, node, buffer)`` for eligible nodes, where
        ``saved_idx`` is the index into the saved tensor portion of fwd
        outputs (i.e. ``fwd_outs[num_fwd_outputs + saved_idx]``).
    """
    fwd_output = next(n for n in fwd_module.graph.nodes if n.op == "output")
    fwd_outs = fwd_output.args[0]

    fwd_outputs_set = OrderedSet(fwd_outs)
    model_outputs = OrderedSet(fwd_outs[:num_fwd_outputs])
    saved_tensors = list(fwd_outs[num_fwd_outputs:])

    entries = []
    skipped_reasons: dict[str, int] = {}
    _rank0 = torch.distributed.is_initialized() and torch.distributed.get_rank() == 0
    for i, node in enumerate(saved_tensors):
        if not isinstance(node, fx.Node):
            continue
        buf = can_paged_stash(
            node,
            fwd_outputs_set,
            model_outputs,
            static_lifetime_input_nodes,
            paged_buffers,
        )
        if buf is not None:
            entries.append((i, node, buf))
            if _rank0:
                val = node.meta.get("val")
                shape_str = str(tuple(val.shape)) if val is not None and hasattr(val, "shape") else "N/A"
                dtype_str = str(val.dtype) if val is not None and hasattr(val, "dtype") else "N/A"
                key = (val.dtype, val.shape[-1]) if val is not None and hasattr(val, "shape") and len(val.shape) >= 1 else "N/A"
                logger.debug("choose_paged_stash_sets ELIGIBLE: name=%s, shape=%s, dtype=%s, buffer_key=%s", node.name, shape_str, dtype_str, key)
        else:
            # Track skip reason for debugging
            val = node.meta.get("val")
            if node.target == operator.getitem:
                reason = "getitem"
            elif not _has_paged_stash_annotation(node) and not _has_any_dynamic_dim(node):
                reason = "no_annotation_or_symint"
            elif val is None or not hasattr(val, "shape") or len(val.shape) < 1:
                reason = "no_tensor_meta"
            elif not _has_any_dynamic_dim(node) and len(val.shape) != 2:
                reason = "static_non_2d"
            else:
                reason = "no_buffer_key"
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
            if _rank0:
                shape_str = str(tuple(val.shape)) if val is not None and hasattr(val, "shape") else "N/A"
                dtype_str = str(val.dtype) if val is not None and hasattr(val, "dtype") else "N/A"
                logger.debug("choose_paged_stash_sets SKIPPED: name=%s, shape=%s, dtype=%s, reason=%s", node.name, shape_str, dtype_str, reason)

    logger.info(
        "choose_paged_stash_sets: %d/%d saved tensors eligible "
        "(of %d total fwd outputs, %d model outputs, skipped: %s)",
        len(entries),
        len(saved_tensors),
        len(fwd_outs),
        num_fwd_outputs,
        dict(skipped_reasons) if skipped_reasons else "none",
    )
    return entries


# ---------------------------------------------------------------------------
# Post-partition: graph surgery helpers
# ---------------------------------------------------------------------------


def _register_buffer_attrs(
    module: fx.GraphModule,
    buf: PagedStashBuffer,
    prefix: str,
) -> dict[str, str | int]:
    """Register PagedStashBuffer tensors as module buffers for graph access.

    Uses ``register_buffer`` so that ``graph.get_attr`` can resolve the names.
    Returns a dict with attribute names and scalar values for the buffer.
    """
    attrs = {}
    for attr_name, tensor in [
        ("buffer", buf.buffer),
        ("free_list", buf.free_list),
        ("free_list_head", buf.free_list_head),
        ("free_list_capacity", buf.free_list_capacity),
        ("overflow", buf.overflow),
        ("free_list_tail", buf.free_list_tail),
    ]:
        full_name = f"{prefix}_{attr_name}"
        attrs[attr_name] = full_name
        module.register_buffer(full_name, tensor)
    attrs["page_size"] = buf.page_size
    attrs["hidden_size"] = buf.hidden_size
    # Track registration for resize updates
    buf._registered_modules.append((module, prefix))
    return attrs


# ---------------------------------------------------------------------------
# Post-partition: find actual token count for _grouped_mm saved tensors
# ---------------------------------------------------------------------------


def _find_num_tokens_node(
    fwd_module: fx.GraphModule,
    fwd_node: fx.Node,
    fwd_output: fx.Node,
    *,
    _cache: dict[int, fx.Node] | None = None,
) -> fx.Node | None:
    """Find a graph node representing the actual token count for a saved tensor.

    For ``_grouped_mm`` outputs, the offsets tensor (from ``cumsum``) is
    available via ``kwargs["offs"]``.  ``offsets[-1]`` equals
    ``tokens_per_expert.sum()`` — the actual (non-padded) token count.

    For derived tensors (transposes, casts, etc.), walks backward to find
    the producing ``_grouped_mm`` node.

    Returns an int64 scalar node, or None if not found.
    """
    if _cache is None:
        _cache = {}

    # Check if this node IS a _grouped_mm
    if (
        fwd_node.op == "call_function"
        and fwd_node.target == torch.ops.aten._grouped_mm.default
    ):
        # Try kwargs first, then positional args for offs
        offsets_node = fwd_node.kwargs.get("offs")
        if offsets_node is None and len(fwd_node.args) > 2:
            offsets_node = fwd_node.args[2]
        if offsets_node is None or not isinstance(offsets_node, fx.Node):
            return None

        # Cache: reuse the same select node for all ops sharing this offsets
        cache_key = id(offsets_node)
        if cache_key in _cache:
            return _cache[cache_key]

        with fwd_module.graph.inserting_before(fwd_output):
            total_node = fwd_module.graph.call_function(
                torch.ops.aten.select.int,
                args=(offsets_node, 0, -1),
            )
            total_i64 = fwd_module.graph.call_function(
                torch.ops.aten.to.dtype,
                args=(total_node, torch.int64),
            )
        _cache[cache_key] = total_i64
        return total_i64

    # Walk backward for derived tensors
    for arg in fwd_node.all_input_nodes:
        result = _find_num_tokens_node(
            fwd_module, arg, fwd_output, _cache=_cache
        )
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# Post-partition: insert paged_stash.copy/pop ops
# ---------------------------------------------------------------------------


def stash_chosen_sets(
    fwd_module: fx.GraphModule,
    bwd_module: fx.GraphModule,
    num_fwd_outputs: int,
    paged_entries: list[tuple[int, fx.Node, PagedStashBuffer]],
    avg_num_tokens: int = 0,
) -> None:
    """Insert paged_stash.copy/pop ops for chosen saved tensors.

    For each entry in ``paged_entries``:
    1. In fwd graph: inserts ``paged_stash.copy`` before the output node,
       replaces the saved tensor output with a compact ``page_record`` handle.
    2. In bwd graph: inserts ``paged_stash.pop`` after the corresponding
       placeholder, replaces all uses with the reconstructed tensor.

    Follows the same graph surgery pattern as ``offload_chosen_sets`` in
    ``torch/_functorch/_activation_offloading/activation_offloading.py``.
    """
    fwd_output = next(n for n in fwd_module.graph.nodes if n.op == "output")
    fwd_outs = fwd_output.args[0]

    # Build name-to-placeholder mapping for backward graph
    bwd_name_to_ph = {
        n.name: n for n in bwd_module.graph.nodes if n.op == "placeholder"
    }

    # Filter entries that have a corresponding bwd placeholder
    valid_entries = []
    for saved_idx, fwd_node, buf in paged_entries:
        if fwd_node.name not in bwd_name_to_ph:
            logger.debug(
                "stash_chosen_sets: skipping %s (no bwd placeholder)", fwd_node.name
            )
            continue
        valid_entries.append((saved_idx, fwd_node, buf))

    if not valid_entries:
        logger.info("stash_chosen_sets: no valid entries after bwd placeholder check")
        return

    # 1. Register buffer tensors on both modules
    buf_attrs_map: dict[int, tuple[dict, dict]] = {}
    for _, _, buf in valid_entries:
        buf_key = id(buf)
        if buf_key not in buf_attrs_map:
            prefix = f"_paged_buf_{buf_key}"
            fwd_attrs = _register_buffer_attrs(fwd_module, buf, prefix)
            bwd_attrs = _register_buffer_attrs(bwd_module, buf, prefix)
            buf_attrs_map[buf_key] = (fwd_attrs, bwd_attrs)

    # 2. In fwd graph: insert paged_stash.copy for each saved tensor
    fwd_outs_list = list(fwd_outs)
    num_tokens_cache: dict[int, fx.Node] = {}
    _rank0 = torch.distributed.is_initialized() and torch.distributed.get_rank() == 0
    for saved_idx, fwd_node, buf in valid_entries:
        fwd_attrs, _ = buf_attrs_map[id(buf)]

        # Find actual token count node (offsets[-1] from _grouped_mm kwargs)
        num_tokens_node = _find_num_tokens_node(
            fwd_module, fwd_node, fwd_output, _cache=num_tokens_cache
        )
        if _rank0:
            val = fwd_node.meta.get("val")
            shape_str = str(tuple(val.shape)) if val is not None and hasattr(val, "shape") else "N/A"
            found_str = "SUCCESS (offsets[-1])" if num_tokens_node is not None else "FALLBACK (tensor shape)"
            logger.debug("stash_chosen_sets: name=%s, shape=%s, num_tokens_node=%s", fwd_node.name, shape_str, found_str)
        if num_tokens_node is None:
            logger.warning(
                "stash_chosen_sets: could not find num_tokens for %s, "
                "falling back to tensor shape",
                fwd_node.name,
            )

        with fwd_module.graph.inserting_before(fwd_output):
            buf_node = fwd_module.graph.get_attr(fwd_attrs["buffer"])
            fl_node = fwd_module.graph.get_attr(fwd_attrs["free_list"])
            flh_node = fwd_module.graph.get_attr(fwd_attrs["free_list_head"])
            flt_node = fwd_module.graph.get_attr(fwd_attrs["free_list_tail"])
            flc_node = fwd_module.graph.get_attr(fwd_attrs["free_list_capacity"])
            ovf_node = fwd_module.graph.get_attr(fwd_attrs["overflow"])

            # If we couldn't find offsets, create a fallback from tensor shape
            if num_tokens_node is None:
                val = fwd_node.meta["val"]
                flat_size = 1
                for d in val.shape[:-1]:
                    flat_size *= d if isinstance(d, int) else 1
                flat_size *= 1 if len(val.shape) <= 1 else 1
                # For the fallback, use the oversized shape (existing behavior)
                num_tokens_fallback = fwd_module.graph.call_function(
                    torch.ops.aten.full.default,
                    args=([1], val.shape[0]),
                    kwargs={"dtype": torch.int64, "device": val.device},
                )
                actual_num_tokens = num_tokens_fallback
            else:
                actual_num_tokens = num_tokens_node

            copy_node = fwd_module.graph.call_function(
                torch.ops.paged_stash.copy,
                args=(
                    fwd_node,
                    buf_node,
                    fl_node,
                    flh_node,
                    flt_node,
                    flc_node,
                    ovf_node,
                    fwd_attrs["page_size"],
                    fwd_attrs["hidden_size"],
                    actual_num_tokens,
                    avg_num_tokens,
                ),
            )

            # Extract page_record (index 0 of the returned tuple)
            page_record_node = fwd_module.graph.call_function(
                operator.getitem,
                args=(copy_node, 0),
            )

        # Replace saved tensor in fwd output with page_record
        fwd_outs_list[num_fwd_outputs + saved_idx] = page_record_node

    fwd_output.args = (tuple(fwd_outs_list),)

    # 3. In bwd graph: insert paged_stash.pop for corresponding placeholders
    for saved_idx, fwd_node, buf in valid_entries:
        ph = bwd_name_to_ph[fwd_node.name]
        _, bwd_attrs = buf_attrs_map[id(buf)]
        val = fwd_node.meta["val"]

        # Compute original shape for potential reshape
        shape = list(val.shape)

        # Insert get_attr nodes sequentially after the placeholder
        with bwd_module.graph.inserting_after(ph):
            buf_node = bwd_module.graph.get_attr(bwd_attrs["buffer"])
        with bwd_module.graph.inserting_after(buf_node):
            fl_node = bwd_module.graph.get_attr(bwd_attrs["free_list"])
        with bwd_module.graph.inserting_after(fl_node):
            flh_node = bwd_module.graph.get_attr(bwd_attrs["free_list_head"])
        with bwd_module.graph.inserting_after(flh_node):
            flt_node = bwd_module.graph.get_attr(bwd_attrs["free_list_tail"])
        with bwd_module.graph.inserting_after(flt_node):
            flc_node = bwd_module.graph.get_attr(bwd_attrs["free_list_capacity"])

        with bwd_module.graph.inserting_after(flc_node):
            pop_node = bwd_module.graph.call_function(
                torch.ops.paged_stash.pop,
                args=(
                    ph,
                    buf_node,
                    fl_node,
                    flh_node,
                    flt_node,
                    flc_node,
                    bwd_attrs["page_size"],
                    bwd_attrs["hidden_size"],
                    val.dtype,
                    avg_num_tokens,
                ),
            )

        # Reshape if the original tensor had more than 2 dimensions
        if len(shape) > 2:
            with bwd_module.graph.inserting_after(pop_node):
                restore_node = bwd_module.graph.call_function(
                    torch.ops.aten.reshape.default,
                    args=(pop_node, shape),
                )
        else:
            restore_node = pop_node

        # Replace all uses of placeholder with restored tensor
        ph.replace_all_uses_with(restore_node)
        # Fix self-reference: pop_node must still reference ph as its first arg
        pop_node.args = (ph, *pop_node.args[1:])

    logger.info(
        "Inserted paged stash ops: %d copy in fwd, %d pop in bwd",
        len(valid_entries),
        len(valid_entries),
    )


# ---------------------------------------------------------------------------
# Post-partition: stream overlap for async copy/pop
# (mirrors torch/_functorch/_activation_offloading/activation_offloading.py)
# ---------------------------------------------------------------------------


def _add_forward_copy_stream_ops(graph: fx.Graph) -> None:
    """Wrap ``paged_stash.copy`` ops with stream fork/join/event for async overlap.

    Mirrors ``add_forward_offload_stream_ops`` from activation offloading.
    Pattern per copy op:

        record_event(ready_event, default_stream)
        fork(default_stream, copy_stream)
        wait_event(ready_event, copy_stream)
        record_stream(tensor, copy_stream)
        --- paged_stash.copy ---
        record_event(done_event, copy_stream)
        join(copy_stream, default_stream)
        wait_event(done_event, default_stream)    ← will be sunk to end of graph
    """
    # Lazy import to register torch.ops.streams.*
    from torch._functorch._aot_autograd import streams as _  # noqa: F401
    from torch._dynamo.variables.streams import get_current_stream, new_event, new_stream

    copy_nodes = [
        n for n in graph.nodes
        if n.op == "call_function" and n.target == torch.ops.paged_stash.copy.default
    ]
    if not copy_nodes:
        return

    # Get stream IDs (compile-time constants baked into the graph)
    # Use the device from the first copy op's tensor arg
    first_tensor = copy_nodes[0].args[0]
    device = first_tensor.meta["val"].device if "val" in first_tensor.meta else torch.device("cuda")
    current_stream_id = get_current_stream(device)
    copy_stream_id = new_stream()

    for copy_node in copy_nodes:
        ready_event_id = new_event()
        done_event_id = new_event()
        tensor_node = copy_node.args[0]  # the activation tensor being copied

        with graph.inserting_before(copy_node):
            graph.call_function(
                torch.ops.streams.record_event.default,
                args=(ready_event_id, current_stream_id),
            )
            graph.call_function(
                torch.ops.streams.fork.default,
                args=(current_stream_id, copy_stream_id),
            )
            graph.call_function(
                torch.ops.streams.wait_event.default,
                args=(ready_event_id, copy_stream_id),
            )
            graph.call_function(
                torch.ops.streams.record_stream.default,
                args=(tensor_node, copy_stream_id),
            )

        with graph.inserting_after(copy_node):
            record_done = graph.call_function(
                torch.ops.streams.record_event.default,
                args=(done_event_id, copy_stream_id),
            )
        with graph.inserting_after(record_done):
            join_node = graph.call_function(
                torch.ops.streams.join.default,
                args=(copy_stream_id, current_stream_id),
            )
        with graph.inserting_after(join_node):
            graph.call_function(
                torch.ops.streams.wait_event.default,
                args=(done_event_id, current_stream_id),
            )


def _add_backward_pop_stream_ops(graph: fx.Graph) -> None:
    """Wrap ``paged_stash.pop`` ops with stream fork/join/event for async overlap.

    Mirrors ``add_backward_reload_stream_ops`` from activation offloading.
    Pattern per pop op:

        fork(default_stream, pop_stream)
        wait_stream(pop_stream, default_stream)
        --- paged_stash.pop ---
        record_event(done_event, pop_stream)
        join(pop_stream, default_stream)
        wait_event(done_event, default_stream)    ← will be prefetched earlier
    """
    from torch._functorch._aot_autograd import streams as _  # noqa: F401
    from torch._dynamo.variables.streams import get_current_stream, new_event, new_stream

    pop_nodes = [
        n for n in graph.nodes
        if n.op == "call_function" and n.target == torch.ops.paged_stash.pop.default
    ]
    if not pop_nodes:
        return

    first_page_record = pop_nodes[0].args[0]
    device = first_page_record.meta["val"].device if "val" in first_page_record.meta else torch.device("cuda")
    current_stream_id = get_current_stream(device)
    pop_stream_id = new_stream()

    for pop_node in pop_nodes:
        done_event_id = new_event()

        with graph.inserting_before(pop_node):
            graph.call_function(
                torch.ops.streams.fork.default,
                args=(current_stream_id, pop_stream_id),
            )
            graph.call_function(
                torch.ops.streams.wait_stream.default,
                args=(pop_stream_id, current_stream_id),
            )

        with graph.inserting_after(pop_node):
            record_done = graph.call_function(
                torch.ops.streams.record_event.default,
                args=(done_event_id, pop_stream_id),
            )
        with graph.inserting_after(record_done):
            join_node = graph.call_function(
                torch.ops.streams.join.default,
                args=(pop_stream_id, current_stream_id),
            )
        with graph.inserting_after(join_node):
            graph.call_function(
                torch.ops.streams.wait_event.default,
                args=(done_event_id, current_stream_id),
            )


def _sink_forward_copy_wait(graph: fx.Graph) -> None:
    """Sink ``wait_event`` for copy completion to end of fwd graph.

    Mirrors ``activation_offload_sink_wait``. Allows the next layer's compute
    to overlap with the async copy.
    """
    copy_nodes = [
        n for n in graph.nodes
        if n.op == "call_function" and n.target == torch.ops.paged_stash.copy.default
    ]
    if not copy_nodes:
        return

    nodes_list = list(graph.nodes)
    node_to_idx = {n: i for i, n in enumerate(nodes_list)}
    output_node = next(n for n in graph.nodes if n.op == "output")

    # For each copy node, find the wait_event 3 positions after it
    # (copy → record_event → join → wait_event)
    for copy_node in copy_nodes:
        idx = node_to_idx[copy_node]
        # The getitem node is right after copy, then record_event, join, wait_event
        # Pattern: copy → getitem → record_event → join → wait_event
        wait_idx = idx + 4  # copy(+0) → getitem(+1) → record(+2) → join(+3) → wait(+4)
        if wait_idx < len(nodes_list):
            wait_node = nodes_list[wait_idx]
            if (
                wait_node.op == "call_function"
                and wait_node.target == torch.ops.streams.wait_event.default
            ):
                output_node.prepend(wait_node)


def _put_stash_ops_on_separate_stream(
    fwd_module: fx.GraphModule,
    bwd_module: fx.GraphModule,
) -> None:
    """Add stream overlap for paged stash copy/pop operations.

    Mirrors ``put_offload_nodes_on_separate_stream`` from activation offloading.
    """
    _add_forward_copy_stream_ops(fwd_module.graph)
    _sink_forward_copy_wait(fwd_module.graph)
    _add_backward_pop_stream_ops(bwd_module.graph)


# ---------------------------------------------------------------------------
# Post-partition: orchestrator
# ---------------------------------------------------------------------------


def enable_paged_stash(
    fwd_module: fx.GraphModule,
    bwd_module: fx.GraphModule,
    num_fwd_outputs: int,
    paged_buffers: dict[tuple[torch.dtype, int], PagedStashBuffer],
    static_lifetime_input_nodes: OrderedSet[fx.Node] | None = None,
    avg_num_tokens: int = 0,
    separate_stream: bool = False,
) -> None:
    """Insert paged_stash.copy/pop ops into separated fw/bw graphs.

    Orchestrates the full paged stash insertion flow, analogous to
    ``enable_activation_offloading`` in
    ``torch/_functorch/_activation_offloading/activation_offloading.py``:

    1. ``choose_paged_stash_sets`` — identify eligible saved tensors
    2. ``stash_chosen_sets`` — insert copy/pop ops
    3. Optionally wrap with stream ops for async overlap
    4. Validate and recompile both graphs

    Args:
        fwd_module: The forward graph module (after partitioning).
        bwd_module: The backward graph module (after partitioning).
        num_fwd_outputs: Number of user-visible forward outputs.
        paged_buffers: Dict mapping (dtype, hidden_size) to PagedStashBuffer.
        static_lifetime_input_nodes: Set of parameter/buffer nodes that must
            not be paged-stashed (same as in ``can_offload``).
    """
    if static_lifetime_input_nodes is None:
        static_lifetime_input_nodes = OrderedSet()

    # 1. Choose which saved tensors to paged-stash
    paged_entries = choose_paged_stash_sets(
        fwd_module, num_fwd_outputs, static_lifetime_input_nodes, paged_buffers
    )
    if not paged_entries:
        return

    # 2. Insert copy/pop ops
    stash_chosen_sets(fwd_module, bwd_module, num_fwd_outputs, paged_entries, avg_num_tokens)

    # 3. Add stream overlap for async copy/pop (optional)
    if separate_stream:
        _put_stash_ops_on_separate_stream(fwd_module, bwd_module)

    # 4. Validate and recompile
    fwd_module.graph.lint()
    bwd_module.graph.lint()
    fwd_module.recompile()
    bwd_module.recompile()


def make_paged_stash_partition_fn(
    paged_buffers: dict[tuple[torch.dtype, int], PagedStashBuffer],
    avg_num_tokens: int = 0,
    separate_stream: bool = False,
):
    """Create a partition_fn wrapper that runs enable_paged_stash after min-cut.

    Args:
        paged_buffers: Dict mapping (dtype, hidden_size) to PagedStashBuffer.

    Returns:
        A partition_fn callable for aot_compile_joint_with_descriptors.
    """

    def partition_fn(joint_module, joint_inputs, *, num_fwd_outputs, **kwargs):
        # Extract static_lifetime_input_indices before forwarding kwargs
        static_lifetime_input_indices = (
            kwargs.pop("static_lifetime_input_indices", None) or []
        )

        # Fine-grained MUST_SAVE: only annotated nodes with real backward
        # usages AND dynamic SymInt first dim.  Static-shaped nodes are left
        # to SAC's decision (MUST_SAVE for expensive ops, PREFER_RECOMPUTE
        # for cheap ops).
        _apply_paged_stash_must_save(
            joint_module, num_fwd_outputs, static_lifetime_input_indices
        )

        fw_module, bw_module = min_cut_rematerialization_partition(
            joint_module,
            joint_inputs,
            num_fwd_outputs=num_fwd_outputs,
            static_lifetime_input_indices=static_lifetime_input_indices,
            **kwargs,
        )

        # Convert joint-graph primal indices to fwd-graph node set.
        # Same logic as classify_nodes in torch/_functorch/partitioners.py.
        from torch._functorch._aot_autograd.utils import _is_primal

        primal_inputs = [n for n in joint_module.graph.nodes if _is_primal(n)]
        joint_static_nodes = OrderedSet(
            p for i, p in enumerate(primal_inputs) if i in static_lifetime_input_indices
        )
        # Map from joint-graph node names to fwd-graph nodes
        fw_name_to_node = {n.name: n for n in fw_module.graph.nodes}
        fw_static_nodes = OrderedSet(
            fw_name_to_node[n.name]
            for n in joint_static_nodes
            if n.name in fw_name_to_node
        )

        enable_paged_stash(
            fw_module, bw_module, num_fwd_outputs, paged_buffers, fw_static_nodes,
            avg_num_tokens=avg_num_tokens,
            separate_stream=separate_stream,
        )
        return fw_module, bw_module

    return partition_fn
