# CUDA-Graphable MoE

End-to-end CUDA graph capture for MoE training, with paged activation stashing to eliminate memory fragmentation from dynamic expert routing.

This experiment demonstrates two key capabilities:

1. **CUDA-graphable MoE**: HybridEP eliminates the CPU-GPU synchronization in expert parallel dispatch, making the entire MoE forward+backward pass capturable in a CUDA graph.
2. **Paged stash SAC**: Pre-allocated paged buffers store MoE expert activations for the backward pass, avoiding both recomputation cost and memory fragmentation from dynamic token counts.

## Requirements

- **4+ NVIDIA GPUs** (tested on GB200 NVLink72)
- **DeepEP library** (hybrid-ep branch): Provides the HybridEP all-to-all kernels for CUDA-graph-compatible expert dispatch.

```bash
cd /tmp
git clone --branch hybrid-ep https://github.com/deepseek-ai/deepep.git
cd deepep
CUDA_HOME=/usr/local/cuda pip install -e .
```

- **Environment**: `CUDA_HOME=/usr/local/cuda` must be set for DeepEP JIT compilation.

## Experiments

All experiments use AOT compilation + CUDAGraph + HybridEP on the DeepSeek V3 debugmodel (4 GPUs, DP=2, TP=2, EP=2).

### Experiment 0: CUDAGraph + HybridEP only (no SAC, no paged stash)

Baseline demonstrating that MoE is CUDA-graphable with HybridEP. No activation checkpointing -- all activations saved as regular tensors.

```bash
CUDA_HOME=/usr/local/cuda NCCL_GRAPH_REGISTER=0 NGPU=4 \
  MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_debugmodel \
  ./run_train.sh \
  --parallelism.data_parallel_shard_degree=2 \
  --parallelism.tensor_parallel_degree=2 \
  --parallelism.expert_parallel_degree=2 \
  --parallelism.expert_parallel_comm_backend=hybridep \
  --parallelism.hybridep_non_blocking_expert_capacity_factor=1.0 \
  --compile.passes cudagraph \
  --activation_checkpoint.mode=none \
  --training.steps=10
```

### Experiment 1: Baseline SAC (recompute MoE activations)

Standard SAC marks attention and mm ops as must-save. MoE expert activations are recomputed during backward (SAC's `PREFER_RECOMPUTE` applies to ops not in the save list).

```bash
CUDA_HOME=/usr/local/cuda NCCL_GRAPH_REGISTER=0 NGPU=4 \
  MODULE=cuda_graphable_moe.deepseek_v3 CONFIG=paged_stash_deepseek_v3_debugmodel \
  ./run_train.sh \
  --parallelism.data_parallel_shard_degree=2 \
  --parallelism.tensor_parallel_degree=2 \
  --parallelism.expert_parallel_degree=2 \
  --compile.joint_passes apply_sac \
  --training.steps=10
```

### Experiment 2: SAC with _grouped_mm saved (fragmentation baseline)

Same as Experiment 1, but also saves `_grouped_mm` outputs as regular tensors. This is the fair comparison for paged stash — both save MoE activations, but this one uses standard autograd tensor storage (subject to fragmentation from dynamic shapes).

```bash
CUDA_HOME=/usr/local/cuda NCCL_GRAPH_REGISTER=0 NGPU=4 \
  MODULE=cuda_graphable_moe.deepseek_v3 CONFIG=paged_stash_deepseek_v3_debugmodel \
  ./run_train.sh \
  --parallelism.data_parallel_shard_degree=2 \
  --parallelism.tensor_parallel_degree=2 \
  --parallelism.expert_parallel_degree=2 \
  --compile.joint_passes apply_sac_grouped_mm \
  --activation_checkpoint.mode=none \
  --training.steps=10
```

### Experiment 3: Paged stash SAC (default config)

SAC + paged stash. MoE expert activations are saved in pre-allocated paged buffers instead of regular tensor storage. Eliminates fragmentation from dynamic token counts.

```bash
CUDA_HOME=/usr/local/cuda NCCL_GRAPH_REGISTER=0 NGPU=4 \
  MODULE=cuda_graphable_moe.deepseek_v3 CONFIG=paged_stash_deepseek_v3_debugmodel \
  ./run_train.sh \
  --parallelism.data_parallel_shard_degree=2 \
  --parallelism.tensor_parallel_degree=2 \
  --parallelism.expert_parallel_degree=2 \
  --training.steps=10
```

### Results

All runs: AOT + CUDAGraph + HybridEP, DeepSeek V3 debugmodel, 4 GPUs (DP=2, TP=2, EP=2), 10 steps.

| Metric | Exp 0: No SAC | Exp 1: SAC (recompute) | Exp 2: SAC (save grouped_mm) | Exp 3: Paged stash SAC |
|---|---|---|---|---|
| Joint passes | (none) | `apply_sac` | `apply_sac_grouped_mm` | `apply_sac` + `apply_paged_sac` |
| MoE activation handling | Save (default) | Recompute | Save (regular tensors) | Save (paged buffers via copy/pop) |
| Step 1 loss | 8.16 | 8.15 | 8.18 | 8.20 |
| Step 10 loss | 4.04 | 3.83 | 3.93 | 3.87 |
| Memory | 3.75 GiB | 3.57 GiB | 3.57 GiB | 5.75 GiB |
| Steady-state tps | ~200K | ~200K | ~200K | ~190K |
| Paged stash ops | -- | -- | -- | 20 copy + 20 pop |

**Key observations**:

- All four achieve high throughput at steady state with CUDA graph replay -- demonstrating that MoE is fully CUDA-graphable with HybridEP.
- Exp 0 (no SAC) uses 3.75 GiB -- all activations saved, no recomputation. SAC (Exp 1) reduces this to 3.57 GiB by recomputing cheap ops.
- Exp 3 inserts 20 `paged_stash.copy` ops in the forward graph and 20 `paged_stash.pop` ops in the backward graph. The pass force-saves all 2D activation tensors from the annotated `_run_experts_grouped_mm` region, matching the set of dynamic tensors that Megatron's `saved_tensors_hooks` would intercept. The buffer key match `(dtype, shape[-1])` filters out non-activation tensors (e.g., float32 casts, int32 offsets).
- Exp 3 uses 5.75 GiB (2.18 GiB more than Exp 1) due to pre-allocated paged buffers sized for 5 dynamic activations per expert module. This is the trade-off: pre-allocated pages avoid fragmentation but consume memory upfront. On larger models where dynamic MoE tensors cause significant fragmentation, paged stash can prevent OOM that Exp 2 would hit.
- On this small debugmodel (`dim=256`, `hidden_dim=256`), the fragmentation difference between Exp 2 and Exp 3 is not visible because the dynamic tensors are small. On larger models (e.g., DeepSeek V3 16B with `dim=2048`, `hidden_dim=1408`), the variable-sized expert activations cause measurable fragmentation that paged stash eliminates.

## How It Works

### 1. HybridEP: Eliminating CPU-GPU Sync in MoE Dispatch

Standard EP dispatch requires CPU-GPU synchronization to learn per-rank token counts before sizing the all-to-all output buffer. This breaks CUDA graph capture.

HybridEP pre-computes the output buffer size using a **capacity factor**:
```
num_permuted_tokens = num_tokens * ep_size * min(num_local_experts, top_k) * capacity_factor
```
With `capacity_factor=1.0`, the buffer is worst-case sized. No D2H sync needed. The `DispatchHandle` (communication state) is passed as a graph input placeholder, avoiding partitioner issues with non-tensor values.

### 2. Region Annotation + Pre-Partition Pass (`apply_paged_sac_pass`)

The expert computation function `_run_experts_grouped_mm` is wrapped with
`annotate_fn({"paged_stash": True})` at parallelize time. This uses PyTorch's
`torch.fx.traceback.annotate_fn` — the same mechanism used for EP annotations
(`{"EP": "dispatch"}`) and flex attention (`{"compile_with_inductor": ...}`).
Every FX node traced inside the annotated function carries the annotation in
`node.meta["custom"]["paged_stash"]`.

The pre-partition pass `apply_paged_sac_pass` scans the joint fwd+bwd FX graph and
overrides `MUST_SAVE` on all annotated 2D-output nodes. This forces the min-cut
partitioner to save all expert activations (rather than recomputing them under SAC's
`PREFER_RECOMPUTE`), so they cross the fwd/bwd boundary where the post-partition pass
can intercept them for paged stashing.

This is the graph-level equivalent of Megatron's `saved_tensors_hooks` interception
inside the expert region: Megatron intercepts all tensors saved by autograd inside
`get_paged_stash_context()` and stashes those with `tensor.size(0) == max_num_tokens`.
Here, `annotate_fn` defines the region and the buffer key match identifies which
activations to stash.

### 3. Post-Partition Pass (`enable_paged_stash` via `partition_fn`)

After the min-cut partitioner splits the joint graph into fwd/bwd subgraphs, the
`partition_fn` wrapper runs `enable_paged_stash`. This pass identifies saved tensors
eligible for paged stash:

1. **`can_paged_stash`** checks each saved tensor (fwd outputs beyond `num_fwd_outputs`):
   - Standard gates from `can_offload`: must be in fwd_outputs, not a model output, not a static lifetime input (param/buffer), not a getitem
   - Must be a 2D tensor (activations are `[tokens, hidden]`; 3D weight transposes are excluded)
   - Must carry the `paged_stash` annotation from `annotate_fn`
   - Buffer key match: `(dtype, shape[-1])` must exist in `paged_buffers`
2. **`stash_chosen_sets`** inserts ops for eligible tensors:
   - **In fwd graph**: inserts `paged_stash.copy` after each matched node, replaces the saved tensor output with a compact `page_record` handle
   - **In bwd graph**: inserts `paged_stash.pop` for the corresponding placeholder, restores the activation from the paged buffer

This follows the same pattern as PyTorch's built-in activation offloading (`torch/_functorch/_activation_offloading/`), which uses `can_offload` + `choose_offload_sets` + `offload_chosen_sets` to insert `device_put` ops into partitioned fwd/bwd graphs.

### 4. Buffer Allocation

`create_paged_buffers` pre-allocates paged buffers sized to the number of dynamic
activations per `(dtype, hidden_size)` key. Per `GroupedExperts` module,
`_run_experts_grouped_mm` produces 5 dynamic activations that the paged stash pass
force-saves:

| Activation | Shape | Buffer key | Count per module |
|---|---|---|---|
| `x.bf16()` (expert input) | `[tokens, dim]` | `(dtype, dim)` | 1 |
| `gmm1 output` (silu input) | `[tokens, hidden_dim]` | `(dtype, hidden_dim)` | |
| `silu output` | `[tokens, hidden_dim]` | `(dtype, hidden_dim)` | |
| `gmm2 output` | `[tokens, hidden_dim]` | `(dtype, hidden_dim)` | |
| `h = silu * gmm2` (gmm3 input) | `[tokens, hidden_dim]` | `(dtype, hidden_dim)` | 4 |

Buffer size per key: `max_tokens * ops_per_key * buffer_size_factor * max_in_flight`.

### 5. PP-Aware Buffer Sizing

With pipeline parallelism, multiple microbatches have their activations simultaneously stashed. `get_max_in_flight_microbatches` computes the peak from the PP schedule's warmup depth:

| Schedule | `max_in_flight` (worst-case, rank 0) |
|---|---|
| No PP (`pp=1`) | 1 |
| 1F1B / GPipe | `min(n_microbatches, pp_degree)` |
| Interleaved 1F1B | `(n_local_stages - 1) * microbatches_per_round + 2 * (pp_degree - 1)` |

These formulas are derived from PyTorch's `torch.distributed.pipelining.schedules` (`Schedule1F1B._step_microbatches` and `_get_warmup_ops`).

### 6. CUDA Graph Capture

The `cudagraph` compiler pass wraps the partitioned fwd/bwd graphs with `CUDAGraphWrapper` (warmup -> capture -> replay). Paged stash buffers are module attributes with stable addresses, compatible with graph replay.

## File Structure

```
cuda_graphable_moe/
├── README.md
├── paged_stashing_guide.md     # In-depth technical guide (Megatron comparison, design rationale)
├── configs.py                  # PagedStashActivationCheckpointConfig (buffer device, page size, etc.)
├── train.py                    # PagedStashTrainer — resets paged buffers each training step
├── paged_stash_ops.py          # Triton kernels, PagedStashBuffer, create_paged_buffers,
│                               #   custom ops (paged_stash::copy/pop), PagedStashObserver
├── paged_stash_graph_pass.py   # Joint pass (apply_paged_sac_pass) + post-partition pass (enable_paged_stash)
└── deepseek_v3/
    ├── __init__.py             # Model registry
    ├── config_registry.py      # Pre-built configs with hybridep defaults
    └── parallelize.py          # Parallelization, annotation, buffer allocation, compilation
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `CUDA_HOME` | Yes | Path to CUDA toolkit (e.g., `/usr/local/cuda`) for DeepEP JIT compilation |
| `NCCL_GRAPH_REGISTER` | No | Set to `0` to disable NCCL graph registration if needed |

### Paged Stash Buffer Settings

Exposed through `PagedStashActivationCheckpointConfig`:

| Field | Default | Description |
|---|---|---|
| `paged_stash_buffer_device` | `"cuda"` | Device for the paged buffer (`"cuda"` or `"cpu"`) |
| `paged_stash_page_size` | `64` | Number of tokens per page |
| `paged_stash_buffer_size_factor` | `1.1` | Over-provisioning factor for buffer allocation |

### Default Config Settings

| Setting | Default | Description |
|---|---|---|
| `compile.enable` | `True` | Enable AOT compilation |
| `compile.joint_passes` | `["apply_sac", "apply_paged_sac"]` | Standard SAC + paged stash annotations |
| `compile.passes` | `["cudagraph"]` | CUDA graph capture for fwd/bwd |
| `parallelism.expert_parallel_comm_backend` | `"hybridep"` | HybridEP for CUDA-graph-compatible MoE |
| `parallelism.hybridep_non_blocking_expert_capacity_factor` | `1.0` | Pre-size dispatch buffers (no D2H sync) |

### Available Joint Passes

| Pass | Description |
|---|---|
| `apply_sac` | Standard SAC (save attention/mm, recompute rest) |
| `apply_paged_sac` | Force-save annotated dynamic expert activations for paged stashing (composes with `apply_sac`) |
| `apply_sac_grouped_mm` | SAC + save `_grouped_mm` as regular tensors (fragmentation baseline) |

## Available Configs

| Config Name | Description |
|---|---|
| `paged_stash_deepseek_v3_debugmodel` | Debug model (SDPA attention) |
| `paged_stash_deepseek_v3_671b` | DeepSeek V3 671B |
| `paged_stash_deepseek_v3_debugmodel_mxfp8` | Debug model with MXFP8 quantization on expert grouped GEMMs |
