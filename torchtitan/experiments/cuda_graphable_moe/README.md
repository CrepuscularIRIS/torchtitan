# CUDA-Graphable MoE with Paged Activation Stashing

End-to-end CUDA graph capture for MoE training, with paged activation stashing to eliminate memory fragmentation from dynamic expert routing.

1. **CUDA-graphable MoE**: HybridEP eliminates CPU-GPU synchronization in expert parallel dispatch, making the entire MoE forward+backward capturable in a CUDA graph.
2. **Paged stash SAC**: Pre-allocated paged buffers store MoE expert activations for the backward pass, avoiding both recomputation cost and memory fragmentation from dynamic token counts.
3. **3-level overflow defense**: Host spillover, cross-rank detection, and retry with buffer growth — mirrors Megatron-LM's approach.

## Setup

**Platform**: 4+ NVIDIA GPUs with NVLink (tested on 4x GB200, CUDA 13.2, aarch64).

### Step 1: Install DeepEP

```bash
cd /tmp && git clone --branch hybrid-ep https://github.com/deepseek-ai/deepep.git
cd /tmp/deepep && CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="10.0" pip install -e .
```

Verify: `python -c "import deep_ep; print(deep_ep.__version__)"`

> **Note**: Adjust `TORCH_CUDA_ARCH_LIST` to your GPU architecture (e.g., `"9.0"` for H100).

### Step 2: Install torchtitan

```bash
cd /workspace/torchtitan && pip install -e .
```

### Step 3: Verify

```bash
CUDA_HOME=/usr/local/cuda NCCL_GRAPH_REGISTER=0 NGPU=4 \
  MODULE=cuda_graphable_moe.deepseek_v3 CONFIG=paged_stash_deepseek_v3_debugmodel \
  ./run_train.sh \
  --parallelism.data_parallel_shard_degree=2 \
  --parallelism.tensor_parallel_degree=2 \
  --parallelism.expert_parallel_degree=2 \
  --training.steps=5
```

Expected output:
- `Created 1 paged stash buffers (..., host_buffer=no)`
- `Inserted paged stash ops: 20 copy + wait in fwd, 20 pop + wait in bwd`
- Step 1 loss ~8.1, step 5 loss ~4.6
- `Training completed` + `Process group destroyed`

## Why Paged Stashing

### The Problem

MoE expert activations have **dynamic token counts** — the number of tokens
routed to each expert varies per-batch due to learned routing decisions.
Under HybridEP with capacity-factor padding, these activations are oversized
(padded to worst-case). Standard SAC either:

1. **Recomputes** them (expensive: grouped GEMM + SwiGLU per layer per backward step), or
2. **Saves** them (fragmented: dynamic-shaped allocations create allocator fragmentation,
   especially inside CUDA graphs where the pool is fixed at capture time).

### The Solution

Paged stashing replaces dynamic-shaped saved activations with compact fixed-size
**page_record** handles (int64 tensors encoding page IDs). The actual activation
data is stored in a pre-allocated paged buffer that is managed by Triton kernels.

**Key benefits**:
- **No recomputation**: avoids the cost of re-running grouped GEMM + SwiGLU in backward
- **No fragmentation**: paged buffer is pre-allocated as a single contiguous allocation
- **CUDA graph compatible**: page_record handles are fixed-size, paged buffer addresses
  are stable, Triton kernels are capturable
- **Async stream overlap**: copy/pop kernels run on a dedicated transfer stream
  (ao's `_get_or_create_transfer_stream`), with `ao.wait_tensor` for synchronization

### What it costs

On small debug models, the pre-allocated paged buffer overhead exceeds the savings
(baseline SAC 1.68 GiB vs paged stash 2.96 GiB). The benefit appears on larger models
where activation fragmentation dominates memory consumption.

## Experiments

All experiments use AOT compilation + CUDAGraph + HybridEP on the DeepSeek V3 debugmodel (4 GPUs, DP=2, TP=2, EP=2).

### Experiment 0: CUDAGraph + HybridEP only (no SAC, no paged stash)

Baseline: MoE is CUDA-graphable with HybridEP. No activation checkpointing.

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

Standard SAC with HybridEP. MoE expert activations are recomputed during backward.

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

Expected output:
- `Applied AOT compilation (joint graph export) to the model` (standard compile path, no paged stash)
- `Applied selective activation checkpointing (SAC) graph pass.`
- No `Inserted paged stash ops` line (paged stash is not active)
- Step 1 loss ~8.1, step 10 loss ~3.5

### Experiment 2: Paged stash SAC (default config)

SAC + paged stash. MoE expert activations are saved in pre-allocated paged buffers.
SAC annotates expert activations as `PREFER_RECOMPUTE`, then the paged stash pass
saves them via paging instead of recomputing.

```bash
CUDA_HOME=/usr/local/cuda NCCL_GRAPH_REGISTER=0 NGPU=4 \
  MODULE=cuda_graphable_moe.deepseek_v3 CONFIG=paged_stash_deepseek_v3_debugmodel \
  ./run_train.sh \
  --parallelism.data_parallel_shard_degree=2 \
  --parallelism.tensor_parallel_degree=2 \
  --parallelism.expert_parallel_degree=2 \
  --training.steps=10
```

Expected output:
- `Created 1 paged stash buffers (..., host_buffer=no)`
- `Applied selective activation checkpointing (SAC) graph pass.` (SAC runs first)
- `Applied paged SAC annotation pass (150 annotated nodes found)` (diagnostic)
- `Inserted paged stash ops: 20 copy + wait in fwd, 20 pop + wait in bwd` (paged stash runs after SAC)
- Step 1 loss ~8.0, step 5 loss ~4.5

### Experiment 3: Host spillover test

Undersized CUDA buffer forces activations to spill to pinned host memory (Level 1 overflow defense).

```bash
CUDA_HOME=/usr/local/cuda NCCL_GRAPH_REGISTER=0 NGPU=4 \
  MODULE=cuda_graphable_moe.deepseek_v3 CONFIG=paged_stash_deepseek_v3_debugmodel \
  ./run_train.sh \
  --parallelism.data_parallel_shard_degree=2 \
  --parallelism.tensor_parallel_degree=2 \
  --parallelism.expert_parallel_degree=2 \
  --training.steps=3 \
  --activation_checkpoint.paged_stash_buffer_size_factor=0.30 \
  --activation_checkpoint.paged_stash_host_buffer_size_factor=1.0
```

Expected output:
- `Created 1 paged stash buffers (..., host_buffer=yes)` (host buffer allocated)
- `Paged stash: spilled activations to pinned host on N rank(s)` warning at each step
- Training completes normally (lower memory than Experiment 2 since CUDA buffer is smaller)

### Experiment 4: Overflow retry test

Extremely undersized buffer triggers full overflow and retry with buffer growth (Level 3 defense).

```bash
CUDA_HOME=/usr/local/cuda NCCL_GRAPH_REGISTER=0 NGPU=4 \
  MODULE=cuda_graphable_moe.deepseek_v3 CONFIG=paged_stash_deepseek_v3_debugmodel \
  ./run_train.sh \
  --parallelism.data_parallel_shard_degree=2 \
  --parallelism.tensor_parallel_degree=2 \
  --parallelism.expert_parallel_degree=2 \
  --training.steps=2 \
  --activation_checkpoint.paged_stash_buffer_size_factor=0.20
```

Expected output:
- `Paged stash: stash buffer overflow on N rank(s).`
- `Paged stash: retrying step (attempt 2/2).`
- `PagedStashBuffer grown: N -> M CUDA pages`
- Training completes (buffers grow until sufficient or max retries exhausted)

### Numerics validation

Paged stash produces **identical** loss to baseline SAC (non-computation change).

```bash
# Baseline (SAC, no paged stash, same HybridEP config):
CUDA_HOME=/usr/local/cuda NCCL_GRAPH_REGISTER=0 NGPU=4 \
  MODULE=cuda_graphable_moe.deepseek_v3 CONFIG=paged_stash_deepseek_v3_debugmodel \
  ./run_train.sh \
  --parallelism.data_parallel_shard_degree=2 \
  --parallelism.tensor_parallel_degree=2 \
  --parallelism.expert_parallel_degree=2 \
  --compile.joint_passes apply_sac \
  --training.steps=10 \
  --debug.seed=42 --debug.deterministic

# Paged stash (same model, same seed):
CUDA_HOME=/usr/local/cuda NCCL_GRAPH_REGISTER=0 NGPU=4 \
  MODULE=cuda_graphable_moe.deepseek_v3 CONFIG=paged_stash_deepseek_v3_debugmodel \
  ./run_train.sh \
  --parallelism.data_parallel_shard_degree=2 \
  --parallelism.tensor_parallel_degree=2 \
  --parallelism.expert_parallel_degree=2 \
  --training.steps=10 \
  --debug.seed=42 --debug.deterministic
```

Both runs produce identical loss and grad_norm at every step (verified: 10 steps, 5-digit match at stdout precision, full precision match from TensorBoard).

### SAC composition verification

Verifies that SAC and paged stash compose correctly: SAC annotates nodes first
(expert activations get `PREFER_RECOMPUTE`), then paged stash saves them via
paging instead. The pass ordering prevents SAC from touching paged stash ops.

Run Experiment 1 (SAC only) and Experiment 2 (SAC + paged stash) and compare logs:

**Experiment 1 (SAC only)** — expect:
```
Applied AOT compilation (joint graph export) to the model   ← standard compile path
Applied selective activation checkpointing (SAC) graph pass.
  AC region 0: 32 nodes annotated with MUST_SAVE, 429 nodes annotated with PREFER_RECOMPUTE
  AC region 1: 33 nodes annotated with MUST_SAVE, 447 nodes annotated with PREFER_RECOMPUTE
  ...
```
No paged stash messages. Expert activations are recomputed during backward.

**Experiment 2 (SAC + paged stash)** — expect:
```
Applied AOT compilation with paged stash joint pass         ← paged stash compile path
Applied selective activation checkpointing (SAC) graph pass.
  AC region 0: 32 nodes annotated with MUST_SAVE, 429 nodes annotated with PREFER_RECOMPUTE
  ...                                                       ← same SAC counts as Exp 1
Applied paged SAC annotation pass (150 annotated nodes found)
Inserted paged stash ops: 20 copy + wait in fwd, 20 pop + wait in bwd
```

Key observations:
- SAC region counts are **identical** in both experiments (SAC doesn't know about paged stash)
- Paged stash runs after SAC and inserts its own ops with explicit `MUST_SAVE`
- An assertion in `apply_paged_stash_pass` verifies that every eligible node was
  already annotated by SAC (guards against pass ordering bugs)

## How It Works

### HybridEP: CUDA-graph-compatible MoE dispatch

Standard EP dispatch requires CPU-GPU sync to learn per-rank token counts. HybridEP pre-sizes the buffer using a capacity factor:

```
num_permuted_tokens = num_tokens * ep_size * min(num_local_experts, top_k) * capacity_factor
```

With `capacity_factor=1.0`, the buffer is worst-case sized. No D2H sync needed.

### Paged stash: joint-graph pass (follows PR #2879 cpu_offload_pass)

1. **Region annotation**: `_run_experts_grouped_mm` is wrapped with `annotate_fn({"paged_stash": True})`. Every FX node traced inside carries `node.meta["custom"]["paged_stash"]`.

2. **Joint-graph pass** (`apply_paged_stash_pass`): Operates on the joint fwd+bwd graph before min-cut partitioning. Uses `seq_nr` metadata to classify fwd/bwd nodes (same approach as PR #2879's `cpu_offload_pass`). For each eligible forward node:
   - Inserts `paged_stash.copy` + `ao.wait_tensor(page_record, keepalive=activation)` after the producer
   - Inserts `paged_stash.pop` + `ao.wait_tensor(pop_output)` before backward consumers
   - Redirects backward consumers via `replace_input_with`

3. **Min-cut sees page_records, not activations**: After surgery, the large activation has no backward users — min-cut saves only the compact `page_record` (int64 handle) across the fwd/bwd boundary. The activation is freed after forward.

4. **Stream overlap**: Copy/pop ops use ao's `_get_or_create_transfer_stream` internally. Triton kernels launch on the transfer stream; `ao.wait_tensor` synchronizes the compute stream. Captured into CUDA graphs.

5. **Buffer access**: Via `_PAGED_STASH_REGISTRY[buffer_id]` inside the op implementations — the graph only carries integer `buffer_id` constants.

### 3-level overflow defense

Mirrors Megatron-LM's approach for handling routing skew:

| Level | Mechanism | Trigger | Effect |
|---|---|---|---|
| 1 | Host spillover | CUDA pages exhausted | Triton kernel copies to pinned host; warning logged |
| 2 | Cross-rank detection | Any rank overflows/over-budget | `all_reduce(SUM)` of 3 flags ensures all ranks agree |
| 3 | Retry | Both CUDA + host exhausted, or HybridEP over-budget | Zero grads, grow buffers 2x, reset CUDA graphs, rerun step |

### Buffer sizing

```python
estimated_tokens = max_tokens / capacity_factor   # balanced estimate
cuda_tokens = estimated_tokens * buffer_size_factor * num_ops
host_tokens = estimated_tokens * host_buffer_size_factor * num_ops  # 0 = off
```

`page_record` format: `[num_tokens, spilled_to_host, page_id_0, page_id_1, ...]`

## File Structure

```
cuda_graphable_moe/
├── README.md                   # This file
├── paged_stashing_guide.md     # In-depth technical guide (Megatron comparison, design rationale)
├── configs.py                  # PagedStashActivationCheckpointConfig
├── train.py                    # PagedStashTrainer — overflow detection + retry loop
├── paged_stash_ops.py          # Triton kernels, PagedStashBuffer, _PAGED_STASH_REGISTRY,
│                               #   custom ops (paged_stash::copy/pop), ao stream integration
├── paged_stash_graph_pass.py   # Joint-graph pass (apply_paged_stash_pass) + utility passes
└── deepseek_v3/
    ├── __init__.py             # Model registry
    ├── config_registry.py      # Pre-built configs with hybridep defaults
    └── parallelize.py          # Parallelization, annotation, buffer allocation, compilation
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `CUDA_HOME` | Yes | Path to CUDA toolkit (e.g., `/usr/local/cuda`) for DeepEP JIT |
| `NCCL_GRAPH_REGISTER` | No | Set to `0` to disable NCCL graph registration if needed |

### Paged Stash Config (`PagedStashActivationCheckpointConfig`)

| Field | Default | Description |
|---|---|---|
| `paged_stash_page_size` | `64` | Tokens per page |
| `paged_stash_buffer_size_factor` | `1.1` | CUDA buffer over-provisioning multiplier on estimated tokens |
| `paged_stash_host_buffer_size_factor` | `0.0` | Host (pinned CPU) spillover buffer multiplier (0 = no host buffer) |
| `paged_stash_overflow_detection` | `True` | Enable per-step overflow checking via `all_reduce` |
| `paged_stash_max_retries` | `1` | Max retries on overflow (total attempts = 1 + max_retries) |
| `paged_stash_grow_on_overflow` | `True` | Grow CUDA buffers 2x on overflow before retrying |

### Default Config Settings

| Setting | Default | Description |
|---|---|---|
| `compile.joint_passes` | `["apply_sac", "apply_paged_sac"]` | Standard SAC + paged stash annotations |
| `compile.passes` | `["cudagraph"]` | CUDA graph capture |
| `parallelism.expert_parallel_comm_backend` | `"hybridep"` | CUDA-graph-compatible MoE dispatch |
| `parallelism.hybridep_non_blocking_expert_capacity_factor` | `1.0` | Pre-size dispatch buffers (no D2H sync) |

### Available Joint Passes

| Pass | Description |
|---|---|
| `apply_sac` | Standard SAC (save attention/mm, recompute rest) |
| `apply_paged_sac` | Log annotated expert activations (diagnostic; composes with `apply_sac`) |
| `apply_sac_grouped_mm` | SAC + save `_grouped_mm` as regular tensors (fragmentation baseline) |

## Available Configs

| Config | Description |
|---|---|
| `paged_stash_deepseek_v3_debugmodel` | Debug-scale model for validation |
| `paged_stash_deepseek_v3_671b` | DeepSeek V3 671B |
| `paged_stash_deepseek_v3_debugmodel_mxfp8` | Debug model + MXFP8 quantization on expert GEMMs |

## Further Reading

See [`paged_stashing_guide.md`](paged_stashing_guide.md) for an in-depth technical guide covering:
- Background on the dynamic shape problem in MoE + CUDA graphs
- Megatron-LM implementation comparison (dimension-by-dimension)
- Architecture of the module-level buffer registry
- Relationship to PyTorch's activation offloading API
- FP8/MXFP8 considerations
