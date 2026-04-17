# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Parallelize DeepSeek V3 with graph_trainer AOT compilation and paged SAC.

Extends graph_trainer's parallelize_deepseekv3 with graph-based paged SAC
for MoE expert activations when ``apply_paged_sac`` is in joint_passes.

Usage:
    # Without paged stash (standard graph_trainer SAC):
    --compile.joint_passes apply_sac

    # With paged stash (apply_sac handles SAC annotations, apply_paged_sac
    # adds paged stash metadata on top):
    --compile.joint_passes apply_sac apply_paged_sac
"""

import functools
from collections.abc import Callable

import torch
import torch.nn as nn
from torch._functorch.aot_autograd import aot_compile_joint_with_descriptors
from torch._guards import tracing
from torch.fx.traceback import annotate_fn

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.graph_trainer.common_utils import (
    parallelize_inputs,
    register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.deepseek_v3.parallelize import (
    parallelize_deepseekv3 as graph_trainer_parallelize_deepseekv3,
)
from torchtitan.experiments.graph_trainer.graph_utils import (
    CompiledModule,
    export_joint,
    get_compiler_passes_from_config,
    get_joint_custom_passes_from_config,
    make_compiler_with_passes,
)
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


# ---------------------------------------------------------------------------
# Self-contained AOT compile with partition_fn support
# ---------------------------------------------------------------------------


def _joint_graph_builder_with_partition(
    model: nn.Module,
    model_args: tuple,
    model_kwargs: dict,
    fw_compiler: Callable | None = None,
    bw_compiler: Callable | None = None,
    joint_custom_passes: list[Callable] | None = None,
    dump_folder: str | None = None,
    compile_config: CompileConfig | None = None,
    partition_fn: Callable | None = None,
):
    """Build a joint forward-backward graph with partition_fn support.

    Mirrors ``graph_trainer.graph_utils.joint_graph_builder`` but passes
    ``partition_fn`` to ``aot_compile_joint_with_descriptors``, enabling
    post-partition graph transformations (e.g., inserting paged stash ops).
    """
    assert isinstance(model_args, tuple)

    (joint_with_descriptors, tracing_context) = export_joint(
        model, model_args, model_kwargs, dump_folder=dump_folder,
    )

    # Handle inductor_decomposition pass (same as joint_graph_builder)
    if compile_config is not None:
        joint_pass_names = getattr(compile_config, "joint_passes", [])
        if "inductor_decomposition" in joint_pass_names:
            from torchtitan.experiments.graph_trainer.passes import (
                inductor_decomposition_pass,
            )

            decomp_pass = functools.partial(
                inductor_decomposition_pass,
                joint_with_descriptors=joint_with_descriptors,
            )
            if joint_custom_passes is None:
                joint_custom_passes = []
            joint_custom_passes = [decomp_pass] + joint_custom_passes

    # Run custom passes on joint graph before partitioner
    if joint_custom_passes is not None:
        for joint_custom_pass in joint_custom_passes:
            joint_with_descriptors.graph_module = joint_custom_pass(
                joint_with_descriptors.graph_module
            )

    with tracing(tracing_context):
        compile_kwargs = {
            "fw_compiler": fw_compiler,
            "bw_compiler": bw_compiler,
        }
        if partition_fn is not None:
            compile_kwargs["partition_fn"] = partition_fn
        fn = aot_compile_joint_with_descriptors(
            joint_with_descriptors, **compile_kwargs
        )

    def wrapper_fn(args, kwargs):
        inputs = [
            *model.parameters(),
            *model.buffers(),
            *args,
        ]
        return fn(*inputs, **kwargs)

    return wrapper_fn


def _apply_paged_stash_compile(
    model: nn.Module,
    parallel_dims: ParallelDims,
    compile_config,
    parallelism: ParallelismConfig,
    dump_folder: str,
    partition_fn: Callable,
) -> CompiledModule:
    """AOT compile with partition_fn for paged stash copy/pop insertion.

    Self-contained version of ``apply_compile`` that supports ``partition_fn``
    without requiring changes to ``graph_trainer/compile.py``.
    """
    torch._inductor.config.reorder_for_peak_memory = False
    torch._dynamo.config.capture_scalar_outputs = True

    fsdp_reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward, parallel_dims.pp_enabled
    )

    register_blockmask_pytree_node()

    # Get joint custom passes from config
    joint_custom_passes = get_joint_custom_passes_from_config(
        parallel_dims, compile_config, fsdp_reshard_after_forward
    )

    # Get compiler passes from config
    compiler_passes = get_compiler_passes_from_config(
        model, compile_config, parallel_dims
    )

    # Create compilers with specified passes
    fw_compiler, bw_compiler = make_compiler_with_passes(
        compiler_passes, dump_folder=dump_folder
    )

    # Create joint_graph_builder with partition_fn
    model_joint_graph_builder = functools.partial(
        _joint_graph_builder_with_partition,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        joint_custom_passes=joint_custom_passes,
        dump_folder=dump_folder,
        compile_config=compile_config,
        partition_fn=partition_fn,
    )

    model = CompiledModule(
        model, parallel_dims, model_joint_graph_builder, parallelize_inputs
    )
    logger.info("Applied AOT compilation with paged stash partition_fn")
    return model


# ---------------------------------------------------------------------------
# Main parallelize function
# ---------------------------------------------------------------------------


def parallelize_deepseekv3(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Parallelize DeepSeek V3 with graph_trainer + optional paged SAC.

    When ``apply_paged_sac`` is in ``compile_config.joint_passes``, this
    function allocates paged stash buffers, creates a partition_fn that
    inserts paged_stash.copy/pop ops after min-cut partitioning, and
    compiles with that partition_fn.  Otherwise delegates entirely to
    graph_trainer's ``parallelize_deepseekv3``.
    """
    # Check if paged SAC is requested
    joint_pass_names = getattr(compile_config, "joint_passes", [])
    paged_sac_enabled = "apply_paged_sac" in joint_pass_names

    if not paged_sac_enabled:
        # No paged stash — use graph_trainer's parallelize directly
        return graph_trainer_parallelize_deepseekv3(
            model,
            parallel_dims=parallel_dims,
            training=training,
            model_converters=model_converters,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=ac_config,
            dump_folder=dump_folder,
        )

    # Use graph_trainer's parallelize for TP/EP/DP/AC/hybridep setup, but
    # temporarily disable compilation so we can handle it ourselves.
    original_enable = compile_config.enable
    compile_config.enable = False
    model = graph_trainer_parallelize_deepseekv3(
        model,
        parallel_dims=parallel_dims,
        training=training,
        model_converters=model_converters,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=ac_config,
        dump_folder=dump_folder,
    )
    compile_config.enable = original_enable

    # Annotate _run_experts_grouped_mm so every FX node traced inside it
    # carries {"paged_stash": True} in node.meta["custom"]. The pre-partition
    # pass uses this to force-save dynamic-shaped activations, and the
    # post-partition pass uses it to identify tensors for paged stashing.
    # This is the graph-level equivalent of Megatron's saved_tensors_hooks
    # interception inside the expert region.
    from torchtitan.models.common.moe import moe as _moe_module

    _moe_module._run_experts_grouped_mm = annotate_fn({"paged_stash": True})(
        _moe_module._run_experts_grouped_mm
    )

    # Set up paged stash buffers (requires parallelized model to scan
    # GroupedExperts modules for dtype/hidden_size)
    from ..paged_stash_ops import (
        create_paged_buffers,
    )
    from ..paged_stash_graph_pass import make_paged_stash_partition_fn

    num_experts = model.config.layer.moe.num_experts
    top_k = model.config.layer.moe.top_k
    base_tokens = training.local_batch_size * training.seq_len
    if (
        parallelism.expert_parallel_comm_backend == "hybridep"
        and parallelism.hybridep_non_blocking_expert_capacity_factor is not None
        and parallel_dims.ep_enabled
    ):
        ep_size = parallel_dims.ep
        num_local_experts = num_experts // ep_size
        cf = parallelism.hybridep_non_blocking_expert_capacity_factor
        max_tokens = int(
            base_tokens * ep_size * min(num_local_experts, top_k) * cf
        )
    else:
        cf = None
        max_tokens = base_tokens * top_k

    buffers, overflow = create_paged_buffers(
        model, ac_config, max_tokens=max_tokens,
    )

    if buffers is not None:
        model._paged_stash_buffers = list(buffers.values())
        model._paged_stash_overflow = overflow
        model._paged_stash_buffer_size_factor = getattr(
            ac_config, "paged_stash_buffer_size_factor", 1.1
        )
        # avg_num_tokens: pre-padding estimate (mirrors Megatron's avg_num_tokens)
        model._paged_stash_avg_num_tokens = (
            int(max_tokens // cf) if cf is not None and cf > 0 else 0
        )
        logger.info("Graph-based paged SAC enabled")

        # Compile with partition_fn that inserts paged_stash.copy/pop ops
        separate_stream = getattr(ac_config, "paged_stash_separate_stream", False)
        partition_fn = make_paged_stash_partition_fn(
            buffers, avg_num_tokens=model._paged_stash_avg_num_tokens,
            separate_stream=separate_stream,
        )
        model = _apply_paged_stash_compile(
            model, parallel_dims, compile_config, parallelism,
            dump_folder, partition_fn,
        )
    else:
        # No buffers (no GroupedExperts found) — fall back to standard compile
        model = apply_compile(
            model,
            compile_config=compile_config,
            parallelism=parallelism,
            parallel_dims=parallel_dims,
            dump_folder=dump_folder,
        )

    return model
