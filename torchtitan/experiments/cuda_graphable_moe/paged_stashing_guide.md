# Paged Stashing for MoE Expert Activations

A technical guide explaining how paged stashing works, why it exists, and how Megatron-LM and torchtitan implement it differently.

## Background: The Dynamic Shape Problem in MoE + CUDA Graphs

### How MoE Routing Works

In a Mixture-of-Experts (MoE) transformer layer, each token is routed to a subset of experts (typically top-K out of N total experts). The routing decision is made by a learned gating network:

```
Input tokens x [batch * seq_len, dim]
    |
    v
Router: gate(x) -> scores [num_tokens, num_experts]
    |
    v
Top-K selection -> selected_experts [num_tokens, K], top_scores [num_tokens, K]
    |
    v
Reorder tokens by expert assignment -> routed_input [num_tokens * K, dim]
    |
    v
Expert computation (grouped GEMM) -> routed_output [num_tokens * K, dim]
    |
    v
Combine: weighted sum of expert outputs -> output [num_tokens, dim]
```

Each expert is a small FFN (typically SwiGLU: `silu(x @ w1) * (x @ w3)` then `h @ w2`). With `_grouped_mm`, all experts run as a single batched matrix multiply using offset indices, avoiding a per-expert for-loop.

### Expert Parallelism (EP)

With EP, experts are distributed across ranks. Each rank owns `num_experts / ep_size` local experts. Before expert computation, tokens must be sent to the rank that owns their assigned expert (**dispatch**), and results sent back afterward (**combine**). This is an all-to-all communication pattern.

In token-dropless MoE training, the number of tokens received by each expert varies, resulting in a **dynamic shaped tensor**. PyTorch handles this naturally in eager mode -- tensors are allocated lazily when their shape is known at runtime. However, dynamic shaped tensors pose a fundamental challenge for CUDA graphs: there is no CUDA API that allows allocating GPU memory within a CUDA graph with a size determined in stream order.

### The CUDA Graph Dilemma

The most straightforward way to CUDA-graph MoE is to capture using an **oversized buffer** that covers the worst-case token count any EP rank may receive. This works -- the buffer is static, CUDA graph is happy -- but creates significant memory overhead compared to the eager-mode baseline.

The memory problem is twofold:

1. **The compute buffer must be oversized.** CUDA graph requires static shapes, so the dispatch output buffer is pre-sized to worst-case capacity. During compute, the `_grouped_mm` kernels operate on this oversized buffer (padding is skipped via GPU-side offset indices, but the memory is allocated).

2. **Saved activations inherit the oversized shape.** Autograd saves the `_grouped_mm` outputs for the backward pass. These saved tensors have the oversized shape `[max_capacity, hidden_dim]`. Since each step routes different numbers of tokens to different experts, these tensors -- if naively stored -- fragment the memory allocator.

### Paged Stashing: The Key Insight

Paged stashing **decouples** the need for oversized buffers during compute from the need for properly-sized buffers for storing activations:

- **Compute buffers** (oversized, temporary): Used during expert forward/backward kernels. Sized to worst-case capacity. Freed after compute completes.
- **Stash buffers** (right-sized, paged, persistent): Used to store activations for the backward pass. Sized to actual usage. Persist from forward to backward.

The **stash operation** copies the activation from the oversized compute buffer into the paged stash buffer after each expert layer's forward completes. The **restore operation** copies it back during the backward pass. The key memory saving: the stash packs variable-size activations into a contiguous paged buffer, eliminating fragmentation.

For simple scheduling where activation allocation/deallocation follows a first-in-last-out pattern, stash and restore can be done with **bump allocation** (a simple stack pointer). To accommodate complex scheduling (e.g., pipeline parallelism with interleaved fwd/bwd), **paging** provides the flexibility to allocate and free pages in arbitrary order -- hence the name "paged stashing."

### Prerequisites: Reducing Compute Fragmentation

Paged stashing works best when paired with techniques that reduce **compute fragmentation** -- skipping padded data in the oversized static buffers:

- **HybridEP**: Token dispatch kernels that work with pre-sized (worst-case) output buffers without CPU-GPU synchronization.
- **Host-free Grouped GEMM** (`_grouped_mm` with on-device offsets): Expert computation that uses GPU-side offset indices (`torch.cumsum` on device) rather than CPU token counts, avoiding D2H sync.

---

## The Paged Buffer Design

Both Megatron and torchtitan use the same paged buffer design. Each buffer is organized as `[total_tokens, hidden_size]` backed by a circular free list of page IDs:

```
Buffer:     [page_0 | page_1 | page_2 | ... | page_N]   (each page = page_size tokens)
Free list:  [2, 5, 0, 3, 1, 4, ...]                      (available page IDs)
Head ------>                                               (allocate from head)
                                          <---- Tail       (return to tail)
```

Page management is handled by lightweight GPU kernels fused with the stash/restore operations. The Triton `_paged_stash_copy_kernel` allocates pages from the head and copies token data. The `_paged_stash_pop_kernel` copies back and returns pages to the tail. Both operate entirely on GPU -- no CPU involvement.

A `page_record` tensor `[page_id_0, page_id_1, ...]` tracks which pages hold a given activation. This is the compact handle that crosses the fwd->bwd boundary instead of the full activation tensor.

---

## How It Works in Megatron-LM

Megatron's implementation (PR #2690, `vasunvidia/Megatron-LM`, branch `paged_offloading`) operates at the eager/imperative level using PyTorch's `saved_tensors_hooks` API.

### Step 1: The Oversized Buffer is Created at Dispatch

The HybridEP dispatcher pre-sizes the output buffer to worst-case capacity:

```python
# token_dispatcher.py:1012-1020
budget = int(
    routing_map.shape[0]
    * self.config.moe_router_topk
    * self.moe_expert_rank_capacity_factor   # e.g. 1.0 (worst case)
)
self.num_permuted_tokens = budget

# token_dispatcher.py:1052-1063
dispatched_hidden, ... = hybrid_ep_dispatch(
    ...
    num_permuted_tokens=self.num_permuted_tokens,  # oversized
)
# dispatched_hidden shape: [num_permuted_tokens, hidden_dim]
```

This `dispatched_hidden` is the **oversized compute buffer** -- it arrives at `TEGroupedMLP.forward` as `permuted_local_hidden_states`.

### Step 2: `saved_tensors_hooks` Intercept the Save

Megatron wraps each MoE operation in a `PagedStashContext` that installs `saved_tensors_hooks`:

```python
# experts.py:726-737
offload_context = get_paged_stash_context(
    name="expert_fc1",
    max_num_tokens=permuted_local_hidden_states.shape[0],  # oversized dim
    num_tokens_tensor=tokens_per_expert.sum(),              # actual tokens
    avg_num_tokens=int(max_num_tokens // cap_factor),       # heuristic
)
with offload_context:
    fc1_output, bias_parallel = self.linear_fc1(
        permuted_local_hidden_states, tokens_per_expert
    )
```

When `linear_fc1` runs, autograd saves `permuted_local_hidden_states` for the backward pass. The `pack_fn` hook intercepts this:

```python
# paged_stash.py:742-747 (on_save_for_backward)
if (
    self.max_num_tokens is None
    or tensor.dim() == 0
    or tensor.size(0) != self.max_num_tokens   # only intercept oversized tensors
):
    return tensor.detach()   # pass through non-MoE tensors unchanged
```

Only tensors whose first dimension equals `max_num_tokens` (the oversized budget) are intercepted. All other tensors pass through as plain detached tensors. This heuristic is effective but can theoretically produce false positives.

The intercepted tensor is wrapped in a `PagedTensor`:

```python
# paged_stash.py:810-824
paged_tensor = PagedTensor(
    tensor,                         # the oversized activation
    num_tokens_tensor=...,          # actual token count (GPU scalar)
    max_tokens=self.max_num_tokens, # the oversized dim
    page_size=self.page_size,
)
```

### Step 3: The Stash Copy Happens Asynchronously

After expert compute completes, `paged_stash_group_commit` launches the stash on a separate CUDA stream:

```python
# paged_stash.py:605-625 (stash_paged_tensors)
def stash_paged_tensors(self, pp_schedule_layer):
    current_stream = torch.cuda.current_stream()
    self.pack_stream.wait_stream(current_stream)     # pack_stream waits for compute

    with torch.cuda.stream(self.pack_stream):        # async on pack_stream
        while len(self.paged_tensors_to_stash) > 0:
            paged_tensor = self.paged_tensors_to_stash.pop(0)
            stash_buffer = self.stash_buffers[paged_tensor.dtype][paged_tensor.hidden_size]
            paged_tensor.offload_to_stash(stash_buffer)    # Triton kernel
            self.paged_tensors_stash_in_progress.append(paged_tensor)
```

Inside `offload_to_stash`, after launching the Triton copy kernel:

```python
# paged_stash.py:367-369 (offload_to_stash)
self._original_tensor = self._tensor   # keep reference (copy still in-flight)
self._tensor = None                    # clear autograd's reference
```

At this point, `_original_tensor` holds the oversized buffer alive (the async copy needs it), but `_tensor` is cleared so autograd doesn't reference it.

### Step 4: The Oversized Buffer is Freed

This is the critical question: **when does the oversized buffer actually get freed?**

It happens in the **next layer's** `paged_stash_group_start`, which calls `wait_for_stash_to_complete`:

```python
# paged_stash.py:632-645 (wait_for_stash_to_complete)
def wait_for_stash_to_complete(self):
    current_stream = torch.cuda.current_stream()
    if self._pack_stream_status == 'stashing':
        current_stream.wait_stream(self.pack_stream)    # sync: Triton copy done
        self._pack_stream_status = 'idle'

        # FREE the oversized buffers from the previous layer:
        while len(self.paged_tensors_stash_in_progress) > 0:
            paged_tensor = self.paged_tensors_stash_in_progress.pop(0)
            paged_tensor._original_tensor = None   # <-- THIS is the actual free
```

Setting `_original_tensor = None` drops the last Python reference to the oversized tensor. PyTorch's caching allocator returns its memory to the pool. The `wait_stream` at line 636 guarantees the Triton copy has completed first.

**Why wait in the next layer's `group_start` and not immediately after `group_commit`?** To reduce peak memory. The next expert layer will allocate its own oversized compute buffer. If we freed the previous buffer at `group_commit` time, we'd need to synchronize immediately (blocking the main stream). By deferring to the next `group_start`, the stash copy runs concurrently with other non-expert compute.

### The Complete Timeline

```
Layer N forward:
  group_start  -> wait_for_stash_to_complete()
                    -> sync pack_stream (previous layer's stash is done)
                    -> _original_tensor = None  (free layer N-1's oversized buffer)
  fc1 inside paged_stash_context
                    -> on_save_for_backward intercepts oversized input
                    -> wraps in PagedTensor, queues in paged_tensors_to_stash
  fc2
  group_commit -> stash_paged_tensors() on pack_stream (async)
                    -> Triton copy: oversized buffer -> paged stash buffer
                    -> _original_tensor = _tensor; _tensor = None

Layer N+1 forward:
  group_start  -> wait_for_stash_to_complete()
                    -> sync pack_stream (layer N's stash is done)
                    -> _original_tensor = None  (free layer N's oversized buffer)
```

### The 3-Phase State Machine

**Phase 1 -- `capture` (iteration 1):**
Runs a real forward+backward pass. The `pack_fn` tracks peak concurrent stash usage via a running counter (increment on save, decrement on backward get). This captures the high-water mark per `(dtype, hidden_size)`.

**Phase 2 -- `captured` (iteration 2):**
Buffers are allocated from the high-water mark: `num_tokens = max_tokens[key] * stash_buffer_size_factor`. The capture step doubles as the CUDA graph warmup (needed anyway).

**Phase 3 -- steady state (iteration 3+):**
CUDA graphs are active. Each step resets buffer free lists, replays the stash/restore pattern recorded during capture.

---

## How It Works in torchtitan

torchtitan's implementation (`experiments/cuda_graphable_moe`) operates at the **FX graph level** using AOT compilation with graph passes.

### Step 1: The Oversized Buffer is Created at Dispatch (Same Mechanism)

HybridEP pre-sizes the dispatch output the same way:

```python
# hybridep.py:108-120
def _num_permuted_tokens_for_non_blocking(
    num_tokens, ep_size, num_local_experts, top_k, moe_expert_capacity_factor,
) -> int:
    n = int(num_tokens * ep_size * min(num_local_experts, top_k) * moe_expert_capacity_factor)
    return maybe_align_num_tokens_for_mxfp8(n)
```

During AOT tracing, the fake tensor dispatch produces a **symbolic** output shape using
`ctx.new_dynamic_size()`:

```python
# hybridep.py:223-252 (fake impl for tracing)
@hybridep_dispatch.register_fake
def _dispatch_fake(x, topk_idx, topk_weights, num_experts, non_blocking,
                   moe_expert_capacity_factor, handle):
    ctx = torch.library.get_ctx()
    out_tokens = ctx.new_dynamic_size()   # SymInt, not concrete
    hidden = x.new_empty(out_tokens, x.shape[1])   # [SymInt, hidden_dim]
    scores = x.new_empty(out_tokens, dtype=torch.float32)
    tpe = x.new_empty(num_local_experts, dtype=torch.int64)
    return hidden, scores, tpe
```

This means the joint graph's `_grouped_mm` nodes are traced with `SymInt` first dimensions
(e.g., `shape=(u0, 256)`), enabling downstream passes to identify dynamically-shaped
activations via `isinstance(shape[dim], torch.SymInt)`.

### Step 2: Region Annotation + Fine-Grained MUST_SAVE

`_run_experts_grouped_mm` is wrapped with `annotate_fn({"paged_stash": True})` at
parallelize time. This uses `torch.fx.traceback.annotate_fn` — the same mechanism
used for EP annotations and flex attention — so every FX node traced inside the
function carries `node.meta["custom"]["paged_stash"] = True`.

**Phase 1: `apply_paged_sac_pass` (joint graph, annotation-only)**

This pass runs on the joint fwd+bwd FX graph before partitioning. It **does not set
any recompute tags** — it only logs annotated nodes. SAC's decisions from
`apply_sac_pass` are left completely intact.

```python
# paged_stash_graph_pass.py (apply_paged_sac_pass)
for node in gm.graph.nodes:
    if node.op == "call_function":
        if _has_paged_stash_annotation(node):
            paged_stash_count += 1
            # Log only — no MUST_SAVE, no tag changes
```

**Phase 2: `_apply_paged_stash_must_save` (in partition_fn, before min-cut)**

This function runs inside `partition_fn` where `num_fwd_outputs` is available,
enabling backward-usage analysis. It uses `classify_nodes` from the SAC partitioner
to determine which nodes belong to the forward subgraph, then selectively applies
`MUST_SAVE` only to annotated nodes that are:

1. **Dynamically shaped** (`_has_dynamic_first_dim` — SymInt first dimension)
2. **Needed by backward** (have real tensor backward usages, not just sym-node shape info)

```python
# paged_stash_graph_pass.py (_apply_paged_stash_must_save)
node_info = classify_nodes(joint_module, static_lifetime_input_indices, num_fwd_outputs)
forward_node_names = {n.name for n in node_info.required_fw_nodes}

for node in joint_module.graph.nodes:
    if not _has_paged_stash_annotation(node):
        continue
    backward_usages = [u for u in node.users if u.name not in forward_node_names]
    if len(backward_usages) == 0:
        continue                              # no backward usage — skip
    if all(is_sym_node(u) for u in backward_usages):
        continue                              # only shape info needed — SAC saves syms
    if _has_dynamic_first_dim(node):
        node.meta["recompute"] = CheckpointPolicy.MUST_SAVE  # dynamic + bwd needed
    # else: static shape — leave SAC's decision (MUST_SAVE for expensive, PREFER_RECOMPUTE for cheap)
```

This composes cleanly with SAC:
- `apply_sac_pass` runs first → `PREFER_RECOMPUTE` on cheap ops, `MUST_SAVE` on expensive
- `_apply_paged_stash_must_save` runs → adds `MUST_SAVE` only for dynamic annotated nodes
  with real backward usages (overriding SAC's PREFER_RECOMPUTE for those)
- Static-shaped annotated nodes → SAC's decision stands

Use `--compile.joint_passes apply_sac apply_paged_sac`.

### Step 3: Post-Partition Pass Identifies and Transforms Saved Tensors

After the min-cut partitioner splits the joint graph, `enable_paged_stash` runs via a
custom `partition_fn` wrapper. The wrapper first calls `_apply_paged_stash_must_save`
(Step 2, Phase 2), then the partitioner, then the post-partition pass:

```python
# paged_stash_graph_pass.py
def make_paged_stash_partition_fn(paged_buffers):
    def partition_fn(joint_module, joint_inputs, *, num_fwd_outputs, **kwargs):
        _apply_paged_stash_must_save(joint_module, num_fwd_outputs, ...)
        fw_module, bw_module = min_cut_rematerialization_partition(
            joint_module, joint_inputs, num_fwd_outputs=num_fwd_outputs, **kwargs,
        )
        enable_paged_stash(fw_module, bw_module, num_fwd_outputs, paged_buffers, ...)
        return fw_module, bw_module
    return partition_fn
```

`enable_paged_stash` orchestrates three phases:

**Phase 1: `choose_paged_stash_sets`** — Identifies eligible saved tensors using
`can_paged_stash`, which checks:

```python
# can_paged_stash eligibility logic:
# 1. Standard gates (from can_offload in activation offloading):
#    - Must be in fwd_outputs (saved for backward)
#    - Must not be a model output (first num_fwd_outputs entries)
#    - Must not be a static lifetime input (parameters/buffers)
#    - Must not be a getitem (tuple unpacking, doesn't own storage)
#
# 2. Must carry the paged_stash annotation OR have any dynamic SymInt dim
#
# 3. Must be a dynamic activation (any SymInt dim) or a 2D tensor
#    (fallback for concrete shapes — excludes 3D weight transposes)
#
# 4. Buffer key match: (dtype, shape[-1]) must exist in paged_buffers
```

**Phase 2: `stash_chosen_sets`** — Inserts copy/pop ops for eligible tensors:

The `page_record` is the only thing crossing the fwd->bwd boundary — the full activation
data lives in the fixed-address `PagedStashBuffer`.

### Step 4: What Happens to the Oversized Buffer?

In the AOT path, the lifecycle is different from Megatron's eager path:

1. **The dispatch output** (`hidden` from HybridEP) is the oversized buffer `[max_capacity, dim]`. It flows into `_run_experts_grouped_mm` as input `x`.

2. **`_grouped_mm` outputs** (3 per module) have shape `[max_capacity, hidden_dim]` or `[max_capacity, dim]`. These are the tensors marked `MUST_SAVE` by `apply_paged_sac_pass`.

3. **The min-cut partitioner** places these in the fwd graph output / bwd graph input. They cross the fwd->bwd boundary as saved tensors.

4. **During CUDA graph capture**, these saved tensors are allocated at their oversized shape inside the CUDA graph memory pool. Their addresses are baked into the graph.

5. **During CUDA graph replay**, the same addresses are reused each step. The `_grouped_mm` kernels write to the oversized buffer, but only process `actual_tokens` rows (via GPU-side offsets from `cumsum`). The padded rows contain garbage but are never read.

The oversized compute buffer (dispatch output `hidden`) is NOT saved for backward -- `hybridep::dispatch`'s `save_for_backward` only saves `topk_idx` (the index tensor), not the oversized `hidden`:

```python
# hybridep.py:311-317
def _dispatch_setup_context(ctx, inputs, output):
    x, topk_idx, _, _, _, _, dispatch_handle = inputs
    ctx.save_for_backward(topk_idx)   # only indices, not the oversized hidden
```

So the dispatch buffer is freed after expert compute uses it. The `_grouped_mm` outputs are what persist as saved tensors.

### Step 5: Buffer Allocation — Initial Static + Runtime Resize

Buffers are initially allocated at worst-case size from model structure, then resized
based on observed runtime usage during the CUDAGraph warmup iteration.

**Initial allocation** (`create_paged_buffers`):

```python
# paged_stash_ops.py:create_paged_buffers
ops_per_key: dict[tuple[torch.dtype, int], int] = defaultdict(int)
for _fqn, mod in model.named_modules():
    if isinstance(mod, GroupedExperts):
        ops_per_key[(mod.w1.dtype, mod.w1.shape[-2])] += 4  # gmm1, silu, gmm2, h
        ops_per_key[(mod.w1.dtype, mod.w1.shape[-1])] += 1  # x.bf16()

scaled_max = int(max_tokens * buffer_size_factor * num_ops)
```

**Runtime resize** (mirrors Megatron's `allocate_stash_buffers`):

During CUDAGraph warmup (step 1), the `PagedStashObserver` (status='capture') records
actual and avg token counts via `on_copy`/`on_pop` calls in the `paged_stash.copy` and
`paged_stash.pop` custom op bodies. These mirror Megatron's `on_save_for_backward` /
`on_get_saved_tensor` increment/decrement counters.

After step 1, `allocate_stash_buffers` uses the observed
`max_tokens_across_vp_stages` / `max_avg_tokens_across_vp_stages` to resize buffers
via `buf.resize()`. The resize calls `register_buffer()` on the fwd/bwd GraphModules,
propagating new tensor pointers before CUDAGraph capture on step 2.

Sign convention (mirrors Megatron): positive `stash_buffer_size_factor` uses avg-based
peak (default); negative uses actual-based peak (conservative).

```
Step 1: compile + warmup (eager) → observer records token counts → resize buffers
Step 2: CUDAGraph capture (right-sized buffers — register_buffer propagated)
Step 3+: CUDAGraph replay
```

### Step 6: CUDA Graph Captures at Oversized Shapes

The `CUDAGraphWrapper` captures the compiled fwd/bwd graphs:

```python
# cudagraph.py:109-140
if self.cudagraph is None:
    self.args = args          # static input buffer addresses
    self.cudagraph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(self.cudagraph, pool=self.graph_pool, stream=self.stream):
        self.output = self.runnable(*args)   # all tensor allocations captured here
```

During replay, `copy_non_static_inputs` copies new token data into the static (oversized) input buffers. The paged stash buffer addresses are module attributes -- they never change across replays. The `paged_stash.copy` and `paged_stash.pop` ops are baked into the CUDA graph -- they execute as part of the graph replay, packing activations into pages during forward and restoring them during backward.

### End-to-End Flow

```
parallelize_deepseekv3():
    graph_trainer_parallelize(model, compile=False)    # TP, EP (HybridEP), FSDP
    create_paged_buffers(model, ac_config, max_tokens)
    model._paged_stash_buffers = buffers
    partition_fn = make_paged_stash_partition_fn(buffers)
    _apply_paged_stash_compile(model, ..., partition_fn)

At compile time (first forward):
    AOT export -> joint fwd+bwd graph (SymInt shapes from _dispatch_fake)
    apply_sac_pass: standard SAC annotations (MUST_SAVE / PREFER_RECOMPUTE)
    apply_paged_sac_pass: annotation-only logging (no tag changes)
    partition_fn:
        _apply_paged_stash_must_save: classify_nodes backward-usage analysis
            → MUST_SAVE only for annotated + SymInt + real backward usages
            → static-shaped annotated nodes: leave SAC's decision
        min_cut_rematerialization_partition -> separate fwd/bwd graphs
        enable_paged_stash:
            choose_paged_stash_sets: identify saved tensors (annotation OR SymInt + buffer key)
            stash_chosen_sets: insert paged_stash.copy in fwd, paged_stash.pop in bwd
    cudagraph_pass: wrap fwd/bwd with CUDAGraphWrapper

At runtime:
    Step 1 (observation + warmup):
        PagedStashObserver.paged_stash_reset() → status='capture'
        Reset buffer free lists + overflow flag
        CUDAGraphWrapper warmup (eager) — stash ops run, observer records
        PagedStashObserver.paged_stash_reset() → status='captured'
        allocate_stash_buffers() → resize from observed peak
    Step 2 (CUDAGraph capture):
        Reset buffers (right-sized)
        CUDAGraphWrapper capture (with right-sized buffer addresses)
    Step 3+ (replay):
        Reset buffers
        CUDAGraph replay
        Check overflow
```

### Accounting: Where Ideas Map Between Implementations

| Concept | Megatron | torchtitan |
|---|---|---|
| **Oversized compute buffer** | `dispatched_hidden` from `hybrid_ep_dispatch` `[max_capacity, dim]` | Same: `hidden` from `dispatch_tokens` at traced oversized shape |
| **Which tensors are stashed** | Heuristic: `tensor.size(0) == max_num_tokens` | Region annotation + SymInt backward-usage analysis + buffer key match `(dtype, shape[-1])` |
| **Stash interception point** | `saved_tensors_hooks` pack/unpack at runtime | Pre-partition: `_apply_paged_stash_must_save` (selective MUST_SAVE for SymInt+bwd-needed); post-partition: `can_paged_stash` + graph surgery |
| **Copy/pop mechanism** | Triton kernels called from `PagedTensor.offload_to_stash` / `reload_from_stash` at runtime | `paged_stash::copy` and `paged_stash::pop` custom ops inserted into the FX graph |
| **When oversized buffer is freed** | Explicitly: `_original_tensor = None` in next layer's `group_start` after sync | Implicitly: AOT graph manages tensor lifetimes; dispatch buffer not saved for backward |
| **Stash/restore overlap** | Explicit: `pack_stream` / `unpack_stream` with Pre/Post-scheduler autograd Functions | `torch.ops.streams.*` graph ops: fork/join/event captured into CUDA graph. Sink fwd wait for compute overlap. |
| **Buffer sizing** | Runtime high-water mark from capture iteration (`saved_tensors_hooks` increment/decrement) | Runtime high-water mark from warmup (`PagedStashObserver.on_copy/on_pop` in stash ops). Initial worst-case, resized after step 1. |
| **CUDA graph integration** | 3-phase state machine (begin -> capture -> captured) | 3-step lifecycle: warmup (observe+resize) -> capture -> replay; `PagedStashObserver` mirrors Megatron's state machine |
| **Page management** | Triton kernels with circular free list | Same Triton kernels, same free list design |
| **PP support** | Runtime capture tracks interleaved fwd/bwd via increment/decrement counters | Not supported; will follow Megatron closely when added |

---

## Pipeline Parallelism Considerations

Pipeline parallelism is **not currently supported** with paged stashing in torchtitan.
AOT compilation + CUDA graph capture + PP inter-stage P2P communication is untested.

When PP support is added, it will follow Megatron's approach closely:

- **Megatron**: Uses `PP_PreScheduleFunction` / `PP_PostScheduleFunction` (autograd
  Functions) for schedule-driven stash/reload. During the capture iteration, the
  `on_save_for_backward` / `on_get_saved_tensor` increment/decrement pattern tracks
  the concurrent high-water mark across interleaved microbatches. Buffer sizing is exact
  based on the observed runtime interleaving pattern.

- **torchtitan** (future): Will implement similar schedule-driven coordination.
  The `PagedStashObserver` already uses the increment/decrement pattern which correctly
  captures PP interleaving if the observation iteration includes multiple microbatches.

---

## Relationship to PyTorch Activation Offloading

Our paged stash implementation follows the same architectural pattern as PyTorch's built-in
activation offloading (`torch/_functorch/_activation_offloading/`). Both are post-partition
passes that transform the saved tensor boundary between fwd and bwd graphs. The parallels
are precise enough to serve as a reference for understanding either system.

### The Pattern: Pre-Partition Annotation + Post-Partition Transformation

Both systems use a two-phase approach:

**Phase 1: Mark target nodes on the joint graph (before partitioning)**

Activation offloading uses `config.joint_custom_pass` to set `should_offload`:
```python
# User-provided callback, called on joint graph before partitioning
# (torch/_functorch/config.py:421, graph_compile.py:1759-1762)
def mark_for_offloading(gm, joint_inputs):
    for node in gm.graph.nodes:
        if node.name == "target_op":
            node.meta["should_offload"] = True
    return gm
```

Paged stash uses `apply_paged_sac_pass` (a registered joint pass) to set `should_paged_stash`:
```python
# paged_stash_graph_pass.py (apply_paged_sac_pass)
for node in gm.graph.nodes:
    if node.op == "call_function" and node.target in DEFAULT_PAGED_STASH_OPS:
        node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        node.meta["should_paged_stash"] = True
```

In both cases, the metadata survives partitioning because the partitioner copies
`node.meta` when extracting fwd/bwd subgraphs (`new_node.meta = node.meta` in
`_extract_graph_with_inputs_outputs`).

**Phase 2: Transform saved tensors on partitioned graphs (after partitioning)**

Activation offloading runs inside the partitioner itself, gated by a config flag:
```python
# partitioners.py:1383-1393 (inside default_partition)
if config.enable_activation_offloading:
    enable_activation_offloading(fw_module, bw_module, num_fwd_outputs, ...)
```

Paged stash runs via a `partition_fn` wrapper that calls the partitioner then our pass:
```python
# paged_stash_graph_pass.py (make_paged_stash_partition_fn)
def partition_fn(joint_module, joint_inputs, *, num_fwd_outputs, **kwargs):
    fw_module, bw_module = min_cut_rematerialization_partition(
        joint_module, joint_inputs, num_fwd_outputs=num_fwd_outputs, **kwargs,
    )
    enable_paged_stash(fw_module, bw_module, num_fwd_outputs, paged_buffers)
    return fw_module, bw_module
```

Both approaches are equivalent — they run after the partitioner produces the split graphs.

### Selecting Eligible Saved Tensors

Activation offloading iterates fwd outputs and filters by `should_offload` + eligibility:
```python
# activation_offloading.py:269-303 (choose_offload_sets)
fwd_outputs = fwd_module.graph.find_nodes(op="output")[0].args[0]
model_outputs = fwd_outputs[:num_fwd_outputs]

for node in fwd_module.graph.nodes:
    if node.meta.get("should_offload", False) and can_offload(
        node, fwd_outputs, model_outputs, static_lifetime_input_nodes
    ):
        node.meta["saved_for_offloading"] = True
```

Paged stash uses ``choose_paged_stash_sets`` which iterates saved tensors
(fwd outputs beyond `num_fwd_outputs`) and applies ``can_paged_stash`` with dual
identification — tag OR (nn_module_stack + SymInt) — plus buffer shape match:
```python
# paged_stash_graph_pass.py (choose_paged_stash_sets + can_paged_stash)
fwd_outs = fwd_output.args[0]
saved_tensors = list(fwd_outs[num_fwd_outputs:])

for i, node in enumerate(saved_tensors):
    buf = can_paged_stash(
        node, fwd_outputs_set, model_outputs,
        static_lifetime_input_nodes, paged_buffers,
    )
    # can_paged_stash checks:
    # 1. Standard gates (in fwd_outputs, not model output, not static input, not getitem)
    # 2. Dual: should_paged_stash tag OR (nn_module_stack + SymInt)
    # 3. Buffer key match: (dtype, shape[-1])
    if buf is not None:
        entries.append((i, node, buf))
```

### Transforming the Forward Graph

Activation offloading inserts `device_put` (GPU -> CPU) after the last use of each
offloaded tensor, then replaces the fwd output with the CPU copy:
```python
# activation_offloading.py:131-147 (offload_activation_fw)
with graph.inserting_after(last_user):
    cpu_node = graph.call_function(
        torch.ops.prims.device_put.default,
        args=(node, torch.device("cpu")),
        kwargs={"non_blocking": True},
    )
# Replace in fwd output:
output_node.update_arg(0, tuple(node_to_offload.get(n, n) for n in fwd_outputs))
```

Paged stash inserts `paged_stash.copy` before the fwd output node, then replaces the
saved tensor output with the compact `page_record`:
```python
# paged_stash_graph_pass.py (stash_chosen_sets)
with fwd_module.graph.inserting_before(fwd_output):
    copy_node = fwd_module.graph.call_function(
        torch.ops.paged_stash.copy,
        args=(fwd_node, buf_node, fl_node, flh_node, flt_node, flc_node,
              ovf_node, page_size, hidden_size),
    )
    page_record_node = fwd_module.graph.call_function(
        operator.getitem, args=(copy_node, 0),
    )
# Replace in fwd output:
fwd_outs_list[num_fwd_outputs + saved_idx] = page_record_node
```

The key difference: offloading replaces a GPU tensor with a CPU tensor (same data,
different device). Paged stash replaces a GPU tensor with a compact `page_record`
handle (different data entirely — just page IDs).

### Transforming the Backward Graph

Activation offloading replaces the bwd placeholder with a CPU placeholder, then inserts
`device_put` (CPU -> GPU) before first use:
```python
# activation_offloading.py:323-341 + 150-193
# Replace placeholder:
bwd_offload_node = bwd_module.graph.placeholder(name=fwd_node.name)
bwd_node.replace_all_uses_with(bwd_offload_node)
bwd_module.graph.erase_node(bwd_node)

# Insert reload before first use:
with graph.inserting_before(first_user):
    gpu_node = graph.call_function(
        torch.ops.prims.device_put.default,
        args=(node, original_device),
    )
node.replace_all_uses_with(gpu_node)
```

Paged stash inserts `paged_stash.pop` after the corresponding placeholder, then replaces
all uses:
```python
# paged_stash_graph_pass.py (stash_chosen_sets)
with bwd_module.graph.inserting_after(flc_node):
    pop_node = bwd_module.graph.call_function(
        torch.ops.paged_stash.pop,
        args=(ph, buf_node, fl_node, flh_node, flt_node, flc_node,
              page_size, hidden_size, val.dtype),
    )
ph.replace_all_uses_with(restore_node)
pop_node.args = (ph, *pop_node.args[1:])  # fix self-reference
```

### Summary of Parallels

| Step | Activation Offloading | Paged Stash |
|---|---|---|
| **Pre-partition mark** | `node.meta["should_offload"] = True` | `annotate_fn({"paged_stash": True})` (region annotation) + `_apply_paged_stash_must_save` (selective `MUST_SAVE` for SymInt + backward-needed nodes) |
| **Post-partition hook** | Called inside `default_partition` via `config.enable_activation_offloading` | Called via `partition_fn` wrapper around `min_cut_rematerialization_partition` |
| **Entry point** | `enable_activation_offloading(fw, bw, num_fwd_outputs, ...)` | `enable_paged_stash(fw, bw, num_fwd_outputs, paged_buffers, static_nodes)` |
| **Eligibility filter** | `can_offload()`: not a view, not a model output, contiguous | `can_paged_stash()`: standard gates + (annotation OR any SymInt dim) + buffer key match |
| **Fwd transformation** | Insert `device_put(GPU->CPU)` after last user | Insert `paged_stash.copy` before output; replace with `page_record` |
| **Bwd transformation** | Replace placeholder with CPU version; insert `device_put(CPU->GPU)` before first use | Insert `paged_stash.pop` after placeholder; replace all uses |
| **What crosses fwd->bwd** | CPU tensor (same data, different device) | `page_record` handle (compact page IDs, not activation data) |
| **Optional stream ops** | `add_forward_offload_stream_ops` / `add_backward_reload_stream_ops` for async overlap | Not needed — copy/pop are baked into CUDA graph |

---

## Critical Accounting: Megatron vs. torchtitan

A dimension-by-dimension comparison of the two implementations, identifying where they
align, where they diverge, and what gaps remain.

### 1. Tensor Selection: Which Tensors Get Stashed

**Megatron**: Runtime heuristic via `saved_tensors_hooks`. Every tensor autograd saves
inside a `PagedStashContext` is checked: if `tensor.size(0) == max_num_tokens` (the
oversized budget), it is wrapped in a `PagedTensor` and queued for stashing. The user
configures `stash_modules` to select which submodules (`expert_fc1`, `moe_act`,
`expert_fc2`) are wrapped.

**torchtitan**: Two-phase selection:

1. **Pre-partition** (`_apply_paged_stash_must_save` in `partition_fn`): Uses
   `classify_nodes` backward-usage analysis. For each annotated node:
   - If dynamically shaped (SymInt) AND has real backward tensor usages → `MUST_SAVE`
   - If statically shaped → leave SAC's decision (MUST_SAVE for expensive ops like
     `_grouped_mm`, PREFER_RECOMPUTE for cheap ops like `silu`/`mul`)
   - If no backward usages or only sym-node usages → skip

2. **Post-partition** (`can_paged_stash`): For each saved tensor crossing the fwd/bwd
   boundary:
   - Must have the `paged_stash` annotation OR any dynamic SymInt dimension
   - Must be a dynamic activation (any SymInt dim) or 2D (fallback for concrete shapes)
   - Must match a buffer key `(dtype, shape[-1])`

**Key design parallel**: Both implementations define a "region" containing the expert
computation, then stash activation tensors from that region:
- Megatron: region = `with offload_context:` block; shape check = `size(0) == max_tokens`
- torchtitan: region = `annotate_fn({"paged_stash": True})`; shape check =
  SymInt dim (dynamic) or 2D + buffer key (concrete shapes fallback)

**False positive prevention**: 3D weight transposes (e.g., `w1.bf16().T` with shape
`[num_experts, dim, hidden_dim]`) are inside the annotated region and their last
dimension can match a buffer key. With SymInt enabled, they are excluded because
all their dimensions are concrete (weights have static shapes). With concrete shapes,
the 2D fallback check (`len(shape) == 2`) excludes them.

### 2. Actual Token Count: How `num_tokens` Propagates

**Megatron**: Gets `num_tokens_tensor = tokens_per_expert.sum()` directly in eager mode
when setting up each paged stash context (`experts.py:729`). This is a scalar CUDA tensor
passed explicitly to each `PagedTensor` constructor.

**torchtitan**: Extracts `num_tokens` from the FX graph by tracing backward from
`_grouped_mm` nodes to their `offs` kwarg, then inserting `offsets[-1]` =
`cumsum(tokens_per_expert)[-1]` = `tokens_per_expert.sum()`. The `_find_num_tokens_node`
function (`paged_stash_graph_pass.py:311-368`) does this graph walk.

Both values represent the same quantity. However, there is a **fallback path**: if
`_find_num_tokens_node` cannot trace from a derived tensor (caught by criterion B) back to
a `_grouped_mm`, it falls back to using the **oversized** shape as `num_tokens`:

```python
# paged_stash_graph_pass.py:451-463
if num_tokens_node is None:
    num_tokens_fallback = fwd_module.graph.call_function(
        torch.ops.aten.full.default,
        args=([1], val.shape[0]),  # oversized shape
        kwargs={"dtype": torch.int64, "device": val.device},
    )
```

This means a derived tensor that falls back copies the **full oversized buffer** into the
paged stash — no actual-token optimization. This wastes pages but is correct (pop reads
back all rows). Megatron does not have this issue because every tensor inside the context
shares the same `max_num_tokens`, and the explicit `num_tokens_tensor` is passed uniformly.

In practice, the 10 stashed tensors in the debugmodel all appear to be `_grouped_mm`
outputs (criterion A), so the fallback is not triggered. But for models where criterion B
catches additional derived tensors, the fallback may engage.

### 3. Buffer Sizing Strategy

**Megatron**: Runtime capture — the first iteration (`'capture'` phase) profiles actual
token counts across all layers and VP stages using `saved_tensors_hooks` with
increment/decrement counters (`temp/max_tokens_across_vp_stages` and
`temp/max_avg_tokens_across_vp_stages`). Buffers are allocated based on observed peak.
Two strategies controlled by `stash_buffer_size_factor_cuda`:
- Positive (default 1.10): sizes to `avg_num_tokens` peak × factor. Memory-efficient but
  risks overflow if actual counts exceed the average-based estimate.
- Negative: sizes to `actual_max_tokens` peak × abs(factor). Conservative.

**torchtitan**: Runtime observation via `PagedStashObserver` — mirrors Megatron's approach.
Initial buffers are allocated at worst-case from model structure (`create_paged_buffers`).
During CUDAGraph warmup (step 1), `PagedStashObserver` (status='capture') records
actual and avg token counts via `on_copy`/`on_pop` calls in the `paged_stash.copy`/`pop`
custom op bodies. After warmup, `allocate_stash_buffers` resizes buffers from observed
peak using the same sign convention as Megatron.

The observation mechanism differs from Megatron: Megatron uses `saved_tensors_hooks` at
the autograd level; torchtitan uses recording calls inside the compiled paged stash custom
ops. Both produce the same per-key peak concurrent usage data.

For the debugmodel: initial 42,240 pages → resized to 33,792 pages. Steady-state memory
dropped from 5.75 GiB to 5.52 GiB (230 MiB saved).

### 4. Oversized Buffer Memory Lifecycle

**Megatron**: Explicit three-step lifecycle management:

1. `group_start` → `wait_for_stash_to_complete()` → `_original_tensor = None`
   (free previous layer's oversized buffer after async copy finishes)
2. Expert computation creates new oversized intermediate tensors within the
   `PagedStashContext`
3. `group_commit` → `stash_paged_tensors()` on `pack_stream` (async Triton copy)
   → `_original_tensor = _tensor; _tensor = None`

The oversized buffer from layer N is freed at the start of layer N+1, after the async
copy completes. This is carefully timed memory recycling.

**torchtitan**: Implicit via CUDA graph memory pool:

1. The FX graph represents the full computation statically.
2. `paged_stash.copy` is inserted before the fwd output node — it copies data to
   the paged buffer and produces a compact `page_record`.
3. The original oversized tensor's lifetime is determined by the CUDA graph's internal
   memory pool. When all consumers of an oversized tensor (the copy op and the combine
   op) finish within the graph, the CUDA graph pool recycles that memory for subsequent
   operations. This means the next layer's dispatch can reuse the same memory addresses
   as the previous layer's freed oversized buffers — no explicit lifecycle management
   needed.
4. During CUDA graph replay, the same allocation/deallocation pattern is replayed
   from the captured graph. The pool recycling is deterministic and matches capture.

**Assessment**: torchtitan's approach is **simpler and more correct by construction** —
the compiler handles lifetimes automatically. The trade-off is that CUDA graph capture
bakes in the worst-case allocation pattern from the capture iteration. If the capture
iteration's memory layout is suboptimal, every replay repeats it. But this is inherent
to the CUDA graph approach, not specific to paged stash.

### 5. Async Stream Overlap

**Megatron**: Explicit `pack_stream` and `unpack_stream` for async stash/restore:
- `stash_paged_tensors()` runs Triton copy on `pack_stream`, overlapping with the
  next layer's compute on the main stream.
- `reload_paged_tensors()` runs Triton pop on `unpack_stream`, overlapping with
  backward compute.
- `wait_for_stash_to_complete()` synchronizes the pack_stream with the compute stream.
- Schedule-driven prefetch (`reload_paged_tensors` called from
  `PP_PostScheduleFunction.backward`) starts loading tensors before they are needed.

**torchtitan**: Async stream overlap via `torch.ops.streams.*` FX graph ops (updated
2026-03-24). Follows the same pattern as PyTorch's `enable_activation_offloading`
(`torch/_functorch/_activation_offloading/activation_offloading.py`):

- **Forward**: Each `paged_stash.copy` is wrapped with stream fork/join/event ops:
  ```
  record_event(ready_event, default_stream)     # data ready
  fork(default_stream, copy_stream)              # switch to copy stream
  wait_event(ready_event, copy_stream)           # wait for data
  record_stream(tensor, copy_stream)             # prevent premature free
  --- paged_stash.copy ---                       # Triton copy on copy_stream
  record_event(done_event, copy_stream)          # mark done
  join(copy_stream, default_stream)              # switch back
  wait_event(done_event, default_stream)         # sunk to end of fwd graph
  ```
  The `wait_event` for copy completion is sunk to the end of the forward graph
  (`_sink_forward_copy_wait`), allowing the next layer's compute to overlap with
  the async copy — mirroring `activation_offload_sink_wait`.

- **Backward**: Each `paged_stash.pop` is wrapped with stream ops:
  ```
  fork(default_stream, pop_stream)               # switch to pop stream
  wait_stream(pop_stream, default_stream)        # wait for dependencies
  --- paged_stash.pop ---                        # Triton pop on pop_stream
  record_event(done_event, pop_stream)           # mark done
  join(pop_stream, default_stream)               # switch back
  wait_event(done_event, default_stream)         # wait for pop done
  ```
  The pop group can be prefetched earlier in the graph (moving fork → wait_stream →
  pop → record_event → join before earlier compute nodes while keeping wait_event at
  the original position) — mirroring `activation_reload_prefetch`.

- **CUDA graph compatible**: The `torch.ops.streams.*` ops are captured into the CUDA
  graph during capture. The multi-stream execution pattern is replayed on every graph
  replay.

- **Configuration**: `paged_stash_separate_stream=true` in
  `PagedStashActivationCheckpointConfig`.

**Comparison**:

| Aspect | Megatron | torchtitan |
|---|---|---|
| Stream mechanism | Explicit Python `torch.cuda.Stream()` + stream sync | `torch.ops.streams.*` FX graph ops |
| Overlap point (fwd) | copy overlaps with next layer's compute | Same (wait_event sunk to end of fwd graph) |
| Overlap point (bwd) | prefetch pop before needed | Same pattern available (pop group moveable) |
| CUDA graph compat | Runs outside CUDA graph (eager) | Captured into CUDA graph |
| Overhead | Python-level stream management per step | Zero runtime overhead (baked into graph) |

### 6. Pipeline Parallelism Support

**Megatron**: Full PP integration via `PP_PreScheduleFunction` / `PP_PostScheduleFunction`
(autograd Functions that participate in both forward and backward):
- Schedule recording during the capture phase.
- Schedule-driven prefetch during steady state: `PP_PreScheduleFunction.backward`
  starts reloading the next backward layer's tensors before they are needed.
- Increment/decrement counters track concurrent stash usage across microbatches.
- `paged_stash_init_chunk_handler` initializes per-microbatch state.
- Buffer sizing is exact: based on the observed runtime interleaving pattern.

**torchtitan**: PP is **not currently supported** with paged stashing. AOT compilation
produces a single compiled callable per PP stage, and the interaction between CUDA graph
capture and PP inter-stage P2P communication is untested. When PP support is added, it
will follow Megatron's approach closely (schedule-driven stash/reload, PP-aware buffer
sizing from observed runtime interleaving).

### 7. FP8 / MXFP8 Support

**Megatron**: `PagedTensor` explicitly handles `MXFP8Tensor` — when the tensor is FP8
quantized, it extracts `_columnwise_data`, stashes/restores just the data portion, and
reconstructs the full `MXFP8Tensor` (with scales, quantizer metadata) on reload.
Key details:
- FP8 tensors stored in `uint8` buffers via `tensor.view(buffer.dtype)`
- `'columnwise_scale_inv'` in `grouped_name` triggers scale-specific handling:
  `hidden_size = numel / (max_num_tokens / SCALE_INV_BLOCK_SIZE=32)`
- Separate buffer keys: `(uint8, K)` for qdata, `(uint8, K//32)` for scale

**torchtitan**: MXFP8 works out of the box — **no FP8-specific changes needed**.

Verified (2026-03-24): config `paged_stash_deepseek_v3_debugmodel_mxfp8` runs
with MXFP8 quantization on expert weights via torchao's `MXFP8Converter`. The
paged stash still operates on bf16 activations because:

1. torchao's `_to_mxfp8_then_scaled_grouped_mm` handles quantization **inside**
   the autograd function. Inputs enter as bf16 (or MXTensor from MXFP8 dispatch),
   and outputs are bf16.
2. The FX graph's `_grouped_mm` outputs (which the paged stash selects) are bf16.
3. The `MXTensor` quantized data stays internal to the autograd function — it does
   not cross the fwd/bwd boundary via the paged stash.

This is fundamentally different from Megatron, where TE's `MXFP8Tensor` components
(`_columnwise_data` and `columnwise_scale_inv`) are what autograd saves and the
paged stash intercepts. In torchao, the quantization is transparent to the paged
stash — the stash sees only bf16 tensors.

**Assessment**: torchtitan's approach is **simpler** — no special FP8 handling needed.
The trade-off: Megatron stashes the quantized FP8 data (smaller per-element), while
torchtitan stashes bf16 data (2× larger). For models where the stash buffer is a
significant memory component, stashing FP8 data would save ~50% buffer memory.

**Future optimization — stashing FP8 data directly**: The `_grouped_mm` outputs are bf16
in the FX graph because torchao's `_MXFP8GroupedMM` is an opaque autograd Function
(not traced through by Dynamo). To stash FP8 data directly, two approaches exist:
1. **Quantize at stash boundaries**: Insert quantize (bf16→MXFP8) before
   `paged_stash.copy` and dequantize (MXFP8→bf16) after `paged_stash.pop` in the FX
   graph. Self-contained, no torchao changes needed, but adds quantize/dequantize
   overhead.
2. **Make torchao traceable**: Work with torchao to make `_MXFP8GroupedMM`
   trace-through-able (e.g., via `torch.library` custom ops instead of
   `allow_in_graph` autograd Functions), exposing `MXTensor` components in the FX graph.
   The paged stash would then intercept already-quantized FP8 data.

### 8. Overflow Detection and Handling

**Megatron**: Overflow is detected in the Triton copy kernel
(`avail_pages < required_pages`). Checked at:
1. Iteration start (`paged_stash_reset`) — assertion if previous iteration overflowed.
2. After CUDA graph replay (`check_paged_stash_overflow`) — since graph replay cannot
   raise mid-execution.

**torchtitan**: Same overflow detection in the Triton kernel. Checked after each training
step in `PagedStashTrainer.train_step()` using `overflow.item()`.

**Assessment**: Functionally equivalent. Both check overflow after the step completes and
raise `RuntimeError`. The `overflow.item()` D2H sync is fine because it runs after the
full forward+backward, not during CUDA graph capture.

### 9. Stash/Restore Data Correctness

**Megatron backward**:
- During `'capture'` phase: `on_save_for_backward` truncates tensor to actual tokens;
  `on_get_saved_tensor` re-pads to `max_num_tokens` with zeros.
- During `'captured'` phase: `reload_from_stash` allocates at `original_shape`
  (= `[max_num_tokens, H]`, the full oversized size). Triton pop writes only
  `num_tokens` rows. Padding rows are uninitialized but safe — `_grouped_mm` backward
  only reads `offsets[-1]` rows.

**torchtitan backward**:
- `paged_stash.pop` allocates `[num_pages × page_size, hidden_size]`. Since
  `page_record` is sized to worst-case (`max_num_pages + 1`), `num_pages = max_num_pages
  = ceil(max_num_tokens / page_size)`, so `num_pages × page_size ≥ max_num_tokens`.
  The pop output is at least as large as the oversized buffer.
- The Triton pop kernel writes only `page_record[0]` actual rows. Padding rows are
  uninitialized. `_grouped_mm` backward only reads `offsets[-1]` rows — safe.
- The `register_fake` for pop returns `(create_unbacked_symint(), hidden_size)` — the
  compiler treats the first dimension as unknown/dynamic, which is correct since the
  actual size is data-dependent.

**Assessment**: Both approaches are correct. The pop output may be slightly larger than
`max_num_tokens` in torchtitan (up to `page_size - 1` extra rows due to page alignment),
but the `_grouped_mm` backward only reads valid rows. No correctness issue.

### 10. Process Cleanup / NCCL Reference Management

**Megatron**: `PagedStashManager` is a singleton with natural eager lifecycle. Buffer
deallocation happens when the manager is destroyed. CUDA graph cleanup is handled by
Megatron's `FullCudaGraphWrapper`.

**torchtitan**: Required a fix — `PagedStashBuffer._registered_modules` held strong
Python references to the fwd/bwd `GraphModule` objects (created by
`_register_buffer_attrs` in the graph pass). These kept `CUDAGraphWrapper` →
`torch.cuda.CUDAGraph` → NCCL communicator handles alive, preventing
`destroy_process_group()` from completing. Fixed by clearing `_registered_modules` in
`PagedStashTrainer.close()`.

**Assessment**: This was a torchtitan-specific bug arising from the FX graph approach —
registering buffer attributes on graph modules creates backreferences that the eager
approach does not have. Now fixed.

### Summary Table

| Dimension | Megatron | torchtitan | Status |
|---|---|---|---|
| **Tensor selection** | Runtime heuristic (`size(0) == max_tokens`) | Region annotation + SymInt backward-usage analysis + buffer key | SymInt-aware; composes with SAC |
| **Stash interception** | `saved_tensors_hooks` at runtime | Pre-partition: `_apply_paged_stash_must_save` (selective MUST_SAVE); post-partition: `can_paged_stash` + `stash_chosen_sets` (graph surgery) | Different paradigms, equivalent result |
| **num_tokens source** | `tokens_per_expert.sum()` (eager) | `cumsum(tokens_per_expert)[-1]` (FX graph) | Equivalent |
| **Buffer sizing** | Runtime observed peak + headroom | Runtime observed peak + headroom (via PagedStashObserver) | Both runtime-adaptive now |
| **Oversized buffer lifecycle** | Explicit `_original_tensor` management | Implicit via AOT graph lifetimes | Simpler in torchtitan |
| **Async stream overlap** | `pack_stream` / `unpack_stream` | `torch.ops.streams.*` FX graph ops (fork/join/event) captured into CUDA graph | Both support async overlap |
| **PP support** | Schedule-driven prefetch + runtime capture | Not supported; will follow Megatron closely when added |
| **FP8 support** | `MXFP8Tensor` handling in `PagedTensor` (stashes FP8 data + scale) | Works out of the box — stashes bf16 activations (MXFP8 quantization transparent to stash) | Different approach: Megatron saves FP8, torchtitan saves bf16 |
| **Overflow handling** | Triton check + host assertion | Same | Equivalent |
| **Stash/restore correctness** | Truncate/pad around actual tokens | Copy actual tokens, page-aligned restore | Correct |
| **NCCL cleanup** | Natural eager lifecycle | Required explicit `close()` fix | Fixed |

### Open Questions and Future Work

- **Pipeline parallelism**: PP is not currently supported with paged stashing. AOT
  compilation + CUDA graph capture + PP inter-stage P2P communication is untested and
  likely requires fundamental changes. When implemented, will follow Megatron's approach:
  `PP_PreScheduleFunction` / `PP_PostScheduleFunction` for schedule-driven stash/reload,
  PP-aware buffer sizing from observed runtime interleaving (increment/decrement counters
  across microbatches), and schedule-driven prefetch in backward.

- **Buffer sizing from post-partition counts** (partially resolved): The
  `PagedStashObserver` now sizes buffers from observed runtime usage during warmup. The
  initial static allocation (`create_paged_buffers`) still over-allocates, but this is
  corrected after the observation iteration. A future improvement could skip the initial
  static allocation entirely and allocate only after observation.

- **Async stream overlap in CUDA graphs** (resolved): Implemented using
  `torch.ops.streams.*` FX graph ops (fork/join/event pattern from PyTorch's
  `enable_activation_offloading`). Forward copy ops run on a separate `copy_stream`
  with `wait_event` sunk to end of graph for compute overlap. Backward pop ops run
  on a separate `pop_stream`. All stream ops are captured into the CUDA graph and
  replayed. Config: `paged_stash_separate_stream=true`.

- **Composition with AC decisions** (resolved): The pre-partition pass no longer
  blanket-overrides SAC's `PREFER_RECOMPUTE` to `MUST_SAVE`. Instead,
  `_apply_paged_stash_must_save` uses `classify_nodes` backward-usage analysis to
  selectively apply `MUST_SAVE` only for annotated nodes that are dynamically shaped
  (SymInt) AND have real backward tensor usages. Static-shaped annotated nodes (cheap
  ops like `silu`/`mul`) are left to SAC's cost-based decisions. With concrete shapes,
  SAC saves expensive ops (`_grouped_mm`) and recomputes cheap ones. With SymInt shapes,
  all dynamic annotated ops with backward usages get `MUST_SAVE`.

- **FP8/MXFP8 extension** (works, optimization pending): MXFP8 paged stash runs
  correctly today — stashing bf16 activations. torchao's `_MXFP8GroupedMM` is opaque
  to the FX graph (`allow_in_graph` autograd Function), so FP8 data is not visible.
  To stash FP8 directly (50% buffer memory saving), either: (1) insert explicit
  quantize/dequantize at stash boundaries in the FX graph, or (2) upstream changes
  to torchao to make `_MXFP8GroupedMM` traceable via `torch.library` custom ops.

- **Runtime buffer observation** (resolved): torchtitan now uses `PagedStashObserver`
  with `on_copy`/`on_pop` in the paged stash custom ops (mirrors Megatron's
  `on_save_for_backward`/`on_get_saved_tensor`). During CUDAGraph warmup (step 1),
  the observer records actual/avg token counts. `allocate_stash_buffers` resizes
  buffers from observed peak before capture (step 2). Memory reduced from 5.75 to
  5.52 GiB on the debugmodel.

- **Extending to other regions**: To stash activations from additional MoE components
  (shared experts, router), apply `annotate_fn({"paged_stash": True})` to the target
  function and ensure `create_paged_buffers` creates buffer keys for the new
  `(dtype, hidden_size)` combinations. No changes to the graph pass logic needed.
