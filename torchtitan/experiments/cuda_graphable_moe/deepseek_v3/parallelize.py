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

import torch
import torch.nn as nn
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
    get_compiler_passes_from_config,
    get_joint_custom_passes_from_config,
    joint_graph_builder,
    make_compiler_with_passes,
)
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


# ---------------------------------------------------------------------------
# Self-contained AOT compile with paged stash joint pass ordering
# ---------------------------------------------------------------------------


def _apply_paged_stash_compile(
    model: nn.Module,
    parallel_dims: ParallelDims,
    compile_config,
    parallelism: ParallelismConfig,
    dump_folder: str,
    paged_stash_joint_pass,
) -> CompiledModule:
    """AOT compile with paged stash joint pass appended after config passes.

    Mirrors ``apply_compile`` for AOT mode, but appends the paged stash
    joint pass AFTER config-driven joint passes (apply_sac, etc.).

    This ordering is required: if ``apply_sac`` ran after the paged stash
    pass, it would see ``ao.wait_tensor`` nodes (not in its save list) and
    mark them PREFER_RECOMPUTE.  The partitioner then asserts because
    ``ao.wait_tensor`` has ``has_side_effect`` (impure) but is tagged for
    recompute.  By running paged stash last, SAC never sees these nodes,
    and the paged stash pass sets MUST_SAVE on them explicitly.

    Self-contained in the experiment to avoid modifying
    ``graph_trainer/compile.py``.
    """
    torch._inductor.config.reorder_for_peak_memory = False
    torch._dynamo.config.capture_scalar_outputs = True

    fsdp_reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward, parallel_dims.pp_enabled
    )

    register_blockmask_pytree_node()

    # Get config-driven joint passes (apply_sac, apply_paged_sac, etc.)
    joint_custom_passes = get_joint_custom_passes_from_config(
        parallel_dims, compile_config, fsdp_reshard_after_forward
    )

    # Append the paged stash pass AFTER config passes
    joint_custom_passes = joint_custom_passes + [paged_stash_joint_pass]

    # Get compiler passes from config
    compiler_passes = get_compiler_passes_from_config(
        model, compile_config, parallel_dims
    )

    fw_compiler, bw_compiler = make_compiler_with_passes(
        compiler_passes, dump_folder=dump_folder
    )

    model_joint_graph_builder = functools.partial(
        joint_graph_builder,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        joint_custom_passes=joint_custom_passes,
        dump_folder=dump_folder,
        compile_config=compile_config,
    )

    model = CompiledModule(
        model, parallel_dims, model_joint_graph_builder, parallelize_inputs
    )
    logger.info("Applied AOT compilation with paged stash joint pass")
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
    function allocates paged stash buffers, creates a joint-graph pass that
    inserts paged_stash.copy/pop + ao.wait_tensor ops, and compiles using
    a self-contained AOT pipeline with this pass appended after config
    passes. Otherwise delegates entirely to graph_trainer's
    ``parallelize_deepseekv3``.
    """
    # Check if paged SAC is requested
    joint_pass_names = getattr(compile_config, "joint_passes", [])
    paged_sac_enabled = "apply_paged_sac" in joint_pass_names

    if not paged_sac_enabled:
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
    # carries {"paged_stash": True} in node.meta["custom"]. The joint-graph
    # pass uses this to identify dynamic activations for paged stashing.
    from torchtitan.models.common import moe as _moe_module

    _moe_module._run_experts_grouped_mm = annotate_fn({"paged_stash": True})(
        _moe_module._run_experts_grouped_mm
    )

    # Set up paged stash buffers
    from ..paged_stash_ops import create_paged_buffers

    moe_config = next(l.moe for l in model.config.layers if l.moe is not None)
    num_experts = moe_config.num_experts
    top_k = moe_config.router.top_k
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

    host_buffer_size_factor = getattr(
        ac_config, "paged_stash_host_buffer_size_factor", 0.0
    )
    buffers, overflow, host_spill = create_paged_buffers(
        model,
        ac_config,
        max_tokens=max_tokens,
        capacity_factor=cf,
        host_buffer_size_factor=host_buffer_size_factor,
    )

    if buffers is not None:
        model._paged_stash_buffers = list(buffers.values())
        model._paged_stash_overflow = overflow
        model._paged_stash_host_spill = host_spill

        logger.info("Graph-based paged SAC enabled")

        from ..paged_stash_graph_pass import apply_paged_stash_pass

        paged_stash_joint_pass = functools.partial(
            apply_paged_stash_pass, paged_buffers=buffers,
        )

        model = _apply_paged_stash_compile(
            model, parallel_dims, compile_config, parallelism,
            dump_folder, paged_stash_joint_pass,
        )
    else:
        model = apply_compile(
            model,
            compile_config=compile_config,
            parallelism=parallelism,
            parallel_dims=parallel_dims,
            dump_folder=dump_folder,
        )

    return model
