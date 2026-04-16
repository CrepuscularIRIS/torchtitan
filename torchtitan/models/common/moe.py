# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, Partial

from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear

from torchtitan.ops.scatter_add import deterministic_scatter_add
from torchtitan.protocols.module import Module


# NOTE: keeping this for-loop implementation for comparison
#       and readability, may remove later
def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    # NOTE: this would incur a synchronization between device and host
    num_tokens_per_expert_list = num_tokens_per_expert.tolist()

    # a tuple of tensors indexed by experts
    # each with shape (tokens_per_expert(varying), dim)
    # NOTE: x is not sliced because padding was removed in #2774, so
    # sum(num_tokens_per_expert) == x.shape[0] always holds.
    x_splits = torch.split(
        x,
        split_size_or_sections=num_tokens_per_expert_list,
        dim=0,
    )
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x_splits):
        h = F.silu(torch.matmul(x_expert, w1[expert_idx].transpose(-2, -1)))
        h = h * torch.matmul(x_expert, w3[expert_idx].transpose(-2, -1))
        h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
        # h shape (tokens_per_expert(varying), dim)
        out_experts_splits.append(h)
    out = torch.cat(out_experts_splits, dim=0)

    return out


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    h = F.silu(
        torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets)
    )
    h = h * torch._grouped_mm(
        x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
    )
    out = torch._grouped_mm(h, w2.bfloat16().transpose(-2, -1), offs=offsets).type_as(x)

    return out


class GroupedExperts(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hidden_dim: int
        num_experts: int
        use_grouped_mm: bool = True

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.w1 = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )
        self.w2 = nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.hidden_dim)
        )
        self.w3 = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, config.dim)
        )
        self.use_grouped_mm = config.use_grouped_mm

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w1, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            w1 = self.w1.to_local()
            # pyrefly: ignore [missing-attribute]
            w2 = self.w2.to_local()
            # pyrefly: ignore [missing-attribute]
            w3 = self.w3.to_local()
        else:
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3

        if self.use_grouped_mm:
            return _run_experts_grouped_mm(w1, w2, w3, x, num_tokens_per_expert)
        else:
            return _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)


class _AuxLossBackward(torch.autograd.Function):
    """Injects auxiliary load-balance loss gradients at the router scores level.

    Identity in forward (returns ``scores`` unchanged). In backward, recomputes
    the aux loss from saved (detached) inputs, derives ``d(aux_loss)/d(scores)``
    via ``torch.autograd.grad``, and adds it to the incoming gradient. This
    avoids ``retain_graph=True`` (compatible with activation checkpointing) and
    keeps the model forward return type unchanged (PP-safe).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        num_experts: int,
        bs: int,
        slen: int,
        top_k: int,
        aux_loss_weight: float,
        aux_loss_type: str,
    ) -> torch.Tensor:
        ctx.save_for_backward(scores, selected_experts_indices)
        ctx.num_experts = num_experts  # pyrefly: ignore [missing-attribute]
        ctx.bs = bs  # pyrefly: ignore [missing-attribute]
        ctx.slen = slen  # pyrefly: ignore [missing-attribute]
        ctx.top_k = top_k  # pyrefly: ignore [missing-attribute]
        ctx.aux_loss_weight = aux_loss_weight  # pyrefly: ignore [missing-attribute]
        ctx.aux_loss_type = aux_loss_type  # pyrefly: ignore [missing-attribute]
        return scores

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None, None]:
        scores, selected_experts_indices = ctx.saved_tensors
        # pyrefly: ignore [missing-attribute]
        with torch.enable_grad():
            scores_detached = scores.detach().requires_grad_(True)
            if (
                ctx.aux_loss_type == "sequence_wise"
            ):  # pyrefly: ignore [missing-attribute]
                aux_loss = MoE._sequence_wise_aux_loss(
                    scores_detached,
                    selected_experts_indices,
                    ctx.bs,  # pyrefly: ignore [missing-attribute]
                    ctx.slen,  # pyrefly: ignore [missing-attribute]
                    ctx.top_k,  # pyrefly: ignore [missing-attribute]
                    ctx.aux_loss_weight,  # pyrefly: ignore [missing-attribute]
                )
            else:
                num_tokens_per_expert = torch.histc(
                    selected_experts_indices.view(-1).float(),
                    bins=ctx.num_experts,  # pyrefly: ignore [missing-attribute]
                    min=0,
                    max=ctx.num_experts,  # pyrefly: ignore [missing-attribute]
                )
                aux_loss = MoE._batch_wise_aux_loss(
                    scores_detached,
                    num_tokens_per_expert,
                    ctx.top_k,  # pyrefly: ignore [missing-attribute]
                    ctx.aux_loss_weight,  # pyrefly: ignore [missing-attribute]
                )
            (aux_grad,) = torch.autograd.grad(aux_loss, scores_detached)
        return grad_scores + aux_grad, None, None, None, None, None, None, None


class TokenChoiceTopKRouter(Module):
    """This class implements token-choice routing. In token-choice top-K routing, each token is
        routed to top K experts based on the router scores.

    Optionally supports node-limited (group-limited) routing where experts are divided into groups
    (e.g., by node), and only num_limited_groups groups are considered before selecting top_k experts.
    This reduces cross-node communication in distributed settings.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int
        gate: Linear.Config
        num_expert_groups: int | None = None  # must be a divisor of num_experts
        num_limited_groups: int | None = None
        top_k: int = 1
        score_func: Literal["softmax", "sigmoid"] = "sigmoid"
        route_norm: bool = False
        route_scale: float = 1.0
        _debug_force_load_balance: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.gate = config.gate.build()
        self.num_experts = config.num_experts
        self.num_expert_groups = config.num_expert_groups
        self.num_limited_groups = config.num_limited_groups
        self.top_k = config.top_k
        self.score_func = config.score_func
        self.route_norm = config.route_norm
        self.route_scale = config.route_scale
        self._debug_force_load_balance = config._debug_force_load_balance
        self.aux_loss_type: str = "sequence_wise"  # set by MoE.__init__

    def _debug_force_load_balance_routing(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Balanced round-robin expert assignment.
        Returns (selected_experts_indices [N, K] LongTensor, top_scores [N, K] FloatTensor).
        """
        n_tokens = scores.size(0)
        # Round-robin indices with exact balance
        selected_experts_indices = (
            torch.arange(
                n_tokens * self.top_k, device=scores.device, dtype=torch.int64
            ).reshape(n_tokens, self.top_k)
            % self.num_experts
        )
        top_scores = scores.gather(dim=1, index=selected_experts_indices)  # [N,K]
        return selected_experts_indices, top_scores

    def _get_node_limited_routing_scores(
        self,
        scores_for_choice: torch.Tensor,
    ) -> torch.Tensor:
        """Select num_limited_groups groups based on group scores,
            and set expert scores in non-selected groups as -inf

        Args:
            scores_for_choice: Router scores with expert_bias (if any), shape (bs*slen, num_experts)

        Returns:
            scores_for_choice: shape (bs*slen, num_experts)
        """
        if self.num_limited_groups is None:
            raise ValueError(
                "num_limited_groups must be set when num_expert_groups is set"
            )
        assert self.num_expert_groups is not None
        if self.num_experts % self.num_expert_groups != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must be divisible by num_expert_groups ({self.num_expert_groups})"
            )
        experts_per_group = self.num_experts // self.num_expert_groups
        if experts_per_group < 2:
            raise ValueError(f"experts_per_group ({experts_per_group}) must be >= 2")
        scores_grouped = scores_for_choice.view(
            -1, self.num_expert_groups, experts_per_group
        )
        top2_scores_in_group, _ = scores_grouped.topk(2, dim=-1)
        group_scores = top2_scores_in_group.sum(dim=-1)
        _, group_idx = torch.topk(
            group_scores, k=self.num_limited_groups, dim=-1, sorted=False
        )
        group_mask = torch.ones_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_idx, False)  # False = selected groups (keep)
        # Mask out experts from non-selected groups
        scores_for_choice = scores_grouped.masked_fill(
            group_mask.unsqueeze(-1), float("-inf")
        ).view(-1, self.num_experts)

        return scores_for_choice

    def forward(
        self,
        x: torch.Tensor,
        expert_bias: torch.Tensor | None = None,
        *,
        aux_loss_weight: float = 0.0,
        bs: int = 0,
        slen: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.
            expert_bias (torch.Tensor | None, optional): Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.
            aux_loss_weight (float): Scaled aux loss weight for this call. 0 disables injection.
            bs (int): Batch size (needed for aux loss). Ignored when aux loss is disabled.
            slen (int): Sequence length (needed for aux loss). Ignored when aux loss is disabled.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores (torch.Tensor):
                    Routing scores for selected experts with shape ``(bs*slen, top_k)``.
                - selected_experts_indices (torch.Tensor):
                    Expert indices selected for each token with shape ``(bs*slen, top_k)``.
                - num_tokens_per_expert (torch.Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        # Compute gate in float32 to help stability of expert load balancing.
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            scores = self.gate(x)

        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        # scored is already float32 from the autocast above.
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores)
        elif self.score_func == "softmax":
            scores = F.softmax(scores, dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        scores_for_choice = scores if expert_bias is None else scores + expert_bias
        # Apply node-limited routing if configured
        if self.num_expert_groups is not None:
            scores_for_choice = self._get_node_limited_routing_scores(scores_for_choice)
        _, selected_experts_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )

        # Inject aux loss gradient into scores before deriving top_scores.
        # _AuxLossBackward is identity in forward; in backward it adds
        # d(aux_loss)/d(scores) to the gradient flowing through scores.
        # Because top_scores is derived from scores below, the aux gradient
        # naturally flows through to gate.weight via autograd.
        if self.training and aux_loss_weight > 0:
            scores = _AuxLossBackward.apply(
                scores,
                selected_experts_indices,
                self.num_experts,
                bs,
                slen,
                self.top_k,
                aux_loss_weight,
                self.aux_loss_type,
            )

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        top_scores = scores.gather(dim=1, index=selected_experts_indices)

        # debug override: balanced round-robin routing
        if self._debug_force_load_balance:
            (
                selected_experts_indices,
                top_scores,
            ) = self._debug_force_load_balance_routing(scores)

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        return top_scores, selected_experts_indices, num_tokens_per_expert


# NOTE: the reason we make this a stateless module is to support
#       expert_tensor_parallel_degree=1 with consistent TP/EP APIs.
class TokenReorderer(Module):
    """This module reorders token indices to match the order of experts, enabling
    efficient parallel processing of tokens by experts.
    """

    def __init__(self, *, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reorders token indices to match the order of experts for MoE routing.

        Args:
            top_scores (torch.Tensor): Routing scores for selected experts,
                shape (batch_size * seq_len, top_k)
            selected_experts_indices (torch.Tensor): Expert indices selected for each token,
                shape (batch_size*seq_len, top_k)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores_experts_sorted: Scores reordered to match expert ordering
                - token_indices_experts_sorted: Token indices reordered to match expert ordering
                - num_tokens_per_expert: Number of tokens assigned to each expert
        """
        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        # Reorder the token indices to match the order of the experts
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        token_indices_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )

        top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = token_indices_experts_sorted // self.top_k

        return (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        )


class MoE(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int = 8
        experts: GroupedExperts.Config
        router: TokenChoiceTopKRouter.Config
        score_before_experts: bool = True
        load_balance_coeff: float | None = 1e-3
        shared_experts: FeedForward.Config | None = None
        aux_loss_weight: float = 0.0
        """Weight for the auxiliary load-balance loss. 0 disables it."""
        aux_loss_type: Literal["sequence_wise", "batch_wise"] = "sequence_wise"
        """Type of auxiliary load-balance loss."""
        aux_loss_local_batch_size: int | None = None
        """Total local batch size (before microbatching). Used to normalize aux
        loss gradients across pipeline-parallel microbatches so they match the
        non-PP case. Set automatically by the trainer; None means use the
        microbatch bs (no normalization)."""

    def __init__(self, config: Config):
        super().__init__()

        num_experts = config.num_experts
        self.experts = config.experts.build()
        self.router = config.router.build()
        self.reorderer = TokenReorderer(
            num_experts=num_experts, top_k=config.router.top_k
        )
        self.shared_experts = (
            config.shared_experts.build() if config.shared_experts is not None else None
        )
        self.score_before_experts = config.score_before_experts
        self.aux_loss_weight = config.aux_loss_weight
        self.aux_loss_local_batch_size = config.aux_loss_local_batch_size
        self.top_k = config.router.top_k

        # Set aux loss type on the router at init time (fixed for model lifetime).
        self.router.aux_loss_type = config.aux_loss_type

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        #       expert_bias is updated outside the model in an optimizer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = config.load_balance_coeff
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None
        # tokens_per_expert will be used to track expert usage and to update the expert bias for load balancing
        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(num_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        # Convert DTensor to local tensor for MoE-internal computation.
        # grad_placements=(Partial(),) ensures x.grad is Partial on the tp_mesh
        # in backward, so gradient reduction (reduce-scatter from Partial to
        # Shard(1)) happens once at the MoE boundary rather than being
        # duplicated inside the MoE.
        #
        # Why grad(x) is Partial on the tp_mesh across all parallelism:
        # - TP only / TP+EP with ETP=TP: TP-sharded expert weights (Colwise on
        #   w1/w3, Rowwise on w2) produce Partial output gradients.
        # - TP+EP with ETP=1: each TP rank processes a disjoint token subset
        #   (via ReordererSequenceParallel), so grad(x) is non-zero only at
        #   each rank's token positions(Partial).
        #
        # This holds for all MoE components (router.gate, routed experts, shared
        # experts) and regardless of score_before_experts.
        if isinstance(x, DTensor):
            assert (
                x.device_mesh.ndim == 1
            ), f"Expected 1D mesh, got {x.device_mesh.ndim}D mesh"
            assert x.device_mesh.mesh_dim_names == (
                "tp",
            ), f"Expected TP mesh, got mesh_dim_names={x.device_mesh.mesh_dim_names}"
            x = x.to_local(grad_placements=(Partial(),))
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        # Compute the per-call aux loss weight, scaled so accumulated gradients
        # across PP microbatches match the single-batch case.
        local_bs = self.aux_loss_local_batch_size or bs
        scaled_aux_weight = self.aux_loss_weight * bs / local_bs

        # top_scores and selected_experts_indices shape (bs*slen, top_k)
        # num_tokens_per_expert shape (num_experts,)
        (top_scores, selected_experts_indices, num_tokens_per_expert,) = self.router(
            x,
            self.expert_bias,
            aux_loss_weight=scaled_aux_weight,
            bs=bs,
            slen=slen,
        )

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # and also to count the expert usage
        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        # top_scores_experts_sorted and token_indices_experts_sorted shape (bs*slen*top_k,)
        # num_tokens_per_expert shape (num_experts,)
        # NOTE: the reason we need to compute num_tokens_per_expert again is:
        #       1st computation in router is to update self.tokens_per_expert
        #       which would be the same across all TP ranks.
        #       2nd computation in reorderer is for the actual routing and experts computation
        #       which would be sharded over TP ranks if expert_tensor_parallel_degree==1.
        #       If tensor_paralllel_degree == expert_tensor_parallel_degree, they agree.
        (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        ) = self.reorderer(top_scores, selected_experts_indices)

        # shape (bs*slen*top_k, dim)
        routed_input = x[token_indices_experts_sorted]

        if self.score_before_experts:
            routed_input = (
                routed_input.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        # shape (bs*slen*top_k, dim)
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        # shared expert
        # Note: we execute the shared expert before scoring the output of the routed expert
        # to "implicitly" overlap the shared expert compute with token combine communication
        out = (
            self.shared_experts(x)
            if self.shared_experts is not None
            else torch.zeros_like(x)
        )

        if not self.score_before_experts:
            routed_output = (
                routed_output.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        out = deterministic_scatter_add(
            out,
            token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim),
            routed_output,
        )
        out = out.reshape(bs, slen, dim)
        return out

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        assert isinstance(buffer_device, torch.device)

        with torch.device(buffer_device):
            self.tokens_per_expert = torch.zeros(
                self.experts.num_experts, dtype=torch.float32
            )
            if self.load_balance_coeff is not None:
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )

    @staticmethod
    def _sequence_wise_aux_loss(
        scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
        bs: int,
        slen: int,
        top_k: int,
        aux_loss_weight: float,
    ) -> torch.Tensor:
        """Sequence-wise auxiliary load-balance loss (DeepSeek-V3 Eqs 17-20).

        Computes per-sequence load-balance loss from router scores and expert
        assignments, then averages across sequences.

        Args:
            scores: Router scores after sigmoid/softmax, shape ``(bs*slen, num_experts)``.
            selected_experts_indices: Top-k expert indices, shape ``(bs*slen, top_k)``.
            bs: Batch size.
            slen: Sequence length.
            top_k: Number of experts per token.
            aux_loss_weight: Scalar weight for the loss.
        """
        num_experts = scores.size(-1)
        # (B, S, N) — per-sequence view
        scores_per_seq = scores.view(bs, slen, num_experts)

        # Eq 19: normalize scores so they sum to 1 per token
        denom = scores_per_seq.sum(dim=-1, keepdim=True) + 1e-20
        probs_per_seq = scores_per_seq / denom

        # Eq 20: P_i = mean probability per expert per sequence — (B, N)
        p_i = probs_per_seq.mean(dim=1)

        # Eq 18: f_i = expert selection frequency per sequence — (B, N)
        indices_per_seq = selected_experts_indices.view(bs, -1)  # (B, S*K)
        offset = (
            torch.arange(bs, device=indices_per_seq.device).unsqueeze(1) * num_experts
        )
        flat_indices = (indices_per_seq + offset).reshape(-1)
        counts = torch.bincount(flat_indices.long(), minlength=bs * num_experts)
        counts = counts.reshape(bs, num_experts).to(dtype=scores.dtype)
        f_i = counts * (num_experts / (top_k * slen))

        # Eq 17: per-sequence balance loss, averaged over sequences
        return (f_i * p_i).sum(dim=1).mean() * aux_loss_weight

    @staticmethod
    def _batch_wise_aux_loss(
        scores: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        top_k: int,
        aux_loss_weight: float,
    ) -> torch.Tensor:
        """Batch-wise auxiliary load-balance loss.

        Args:
            scores: Router scores after sigmoid/softmax, shape ``(bs*slen, num_experts)``.
            num_tokens_per_expert: Token counts per expert, shape ``(num_experts,)``.
            top_k: Number of experts per token.
            aux_loss_weight: Scalar weight for the loss.
        """
        num_experts = scores.size(-1)
        total_tokens = scores.size(0)
        p_i = scores.mean(dim=0)
        f_i = num_tokens_per_expert.to(scores.dtype) * (
            num_experts / (top_k * total_tokens)
        )
        return (f_i * p_i).sum() * aux_loss_weight


def apply_moe_load_balance_config(
    moe_cfg: MoE.Config,
    *,
    training_config,
    pp_enabled: bool = False,
) -> None:
    """Apply CLI training config overrides to a MoE.Config.

    Called from each model's ``update_from_config`` for every MoE layer.
    CLI values override model-config defaults when explicitly set (non-zero
    for weights, non-None for coefficients).
    """
    if training_config.moe_aux_loss_weight > 0:
        moe_cfg.aux_loss_weight = training_config.moe_aux_loss_weight
        moe_cfg.aux_loss_type = training_config.moe_aux_loss_type
    if training_config.moe_load_balance_coeff is not None:
        moe_cfg.load_balance_coeff = training_config.moe_load_balance_coeff
    if moe_cfg.aux_loss_weight > 0:
        if moe_cfg.aux_loss_type == "batch_wise" and pp_enabled:
            raise ValueError(
                "batch_wise MoE aux loss is incompatible with pipeline "
                "parallelism because per-microbatch token-to-expert counts "
                "do not reflect the full batch. Use sequence_wise instead."
            )
        moe_cfg.aux_loss_local_batch_size = training_config.local_batch_size
