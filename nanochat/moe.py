"""
Mixture-of-Experts (MoE) block for nanochat.

Two execution modes:
  - **Replicated** (default): every rank holds every expert's weights, dispatch is local.
    Works for any num_experts, but expert weights consume O(E) VRAM per GPU which kills
    larger configs (E=8 h=4096 OOMs at d26).
  - **Expert Parallel (EP)**: expert weights are sharded across ranks. Each rank hosts
    num_experts // world_size experts. Tokens are routed via `dist.all_to_all_single` to
    the rank hosting their chosen expert. Memory per GPU drops by world_size at the cost
    of two all-to-all collectives per forward. Requires `num_experts % world_size == 0`.

Design details (both modes):
- `num_experts` routed experts + `num_shared_experts` always-active experts.
- Top-k softmax routing, renormalized across chosen experts.
- Padded-capacity dispatch so all tensor shapes are static (torch.compile-friendly).
- Switch-style aux loss: λ·E·Σ_e(f_e · p_e), `f_e` detached, `p_e` carries grad.
- Expert weights as 3D nn.Parameter; Muon orthogonalizes each (D_in, D_out) slice via
  batched polar-express (requires the leading-dim fix already shipped in nanochat.optim).
- Compute-matched default: `expert_hidden_dim = 4·n_embd / (top_k + num_shared_experts)`
  rounded to multiples of 64. Override via `config.expert_hidden_dim`.

`forward(x)` returns `(y, aux_loss)`. GPT sums aux_loss across blocks and adds to main CE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from nanochat.common import get_dist_info


class _AllToAllSingle(torch.autograd.Function):
    """Autograd-enabled wrapper around dist.all_to_all_single.

    `torch.distributed.all_to_all_single` is not differentiable; calling it would silently
    break the backward graph (gradients on the expert weights would come back as None).
    The backward of all_to_all is another all_to_all that reverses the sender/receiver
    partitioning — so wrap both directions in a custom autograd.Function.
    """

    @staticmethod
    def forward(ctx, x):
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x.contiguous())
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = torch.empty_like(grad_output)
        dist.all_to_all_single(grad_input, grad_output.contiguous())
        return grad_input


def _all_to_all_single_autograd(x):
    return _AllToAllSingle.apply(x)


def _round_to(x: int, multiple: int) -> int:
    return max(multiple, ((x + multiple - 1) // multiple) * multiple)


def default_expert_hidden(n_embd: int, top_k: int, num_shared_experts: int, head_dim_multiple: int = 64) -> int:
    """Compute-matched: total active FFN params per token ~= dense MLP (hidden=4*n_embd)."""
    active_factor = max(1, top_k + num_shared_experts)
    raw = (4 * n_embd) // active_factor
    return _round_to(raw, head_dim_multiple)


class MoE(nn.Module):
    """Mixture-of-Experts MLP replacement. Drop-in for nanochat.gpt.MLP when num_experts > 1."""

    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.num_experts = config.num_experts
        self.top_k = min(config.top_k, config.num_experts)
        self.num_shared_experts = config.num_shared_experts
        self.capacity_factor = config.capacity_factor
        self.aux_loss_coef = config.moe_aux_loss_coef

        # Expert hidden sizing: explicit override > auto compute-matched.
        explicit = getattr(config, "expert_hidden_dim", -1)
        if explicit is not None and explicit > 0:
            self.expert_hidden_dim = explicit
        else:
            self.expert_hidden_dim = default_expert_hidden(
                config.n_embd, self.top_k, self.num_shared_experts
            )

        # Decide EP vs replicated at init time. EP requires:
        #   - config.expert_parallel == True
        #   - world_size > 1 (single-GPU = replicated by definition)
        #   - num_experts >= world_size AND num_experts % world_size == 0
        ep_requested = bool(getattr(config, "expert_parallel", False))
        _, rank, _, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        if ep_requested and world_size > 1 and self.num_experts >= world_size and self.num_experts % world_size == 0:
            self.expert_parallel = True
            self.E_per_rank = self.num_experts // world_size
        else:
            self.expert_parallel = False
            self.E_per_rank = self.num_experts

        # FP8 for routed-expert matmuls (independent of --fp8 which handles nn.Linear only).
        self.moe_expert_fp8 = bool(getattr(config, "moe_expert_fp8", False))

        # Router: replicated across ranks (decision must be consistent globally).
        self.router_weight = nn.Parameter(torch.empty(self.num_experts, self.n_embd))

        # Routed experts: (E_per_rank, D, H) / (E_per_rank, H, D). Under EP each rank owns
        # a disjoint slice; under replicated mode E_per_rank == num_experts so this is full.
        D, H = self.n_embd, self.expert_hidden_dim
        self.w_fc = nn.Parameter(torch.empty(self.E_per_rank, D, H))
        self.w_proj = nn.Parameter(torch.empty(self.E_per_rank, H, D))

        # Shared experts: always active for every token, no routing — stay replicated.
        if self.num_shared_experts > 0:
            S = self.num_shared_experts
            self.ws_fc = nn.Parameter(torch.empty(S, D, H))
            self.ws_proj = nn.Parameter(torch.empty(S, H, D))
        else:
            self.register_parameter("ws_fc", None)
            self.register_parameter("ws_proj", None)

    @torch.no_grad()
    def init_weights(self):
        """Mirrors nanochat.gpt.MLP init: c_fc uniform +/- 0.4*s, c_proj zeros.

        Under EP each rank initializes its own expert slice. We intentionally *don't*
        all_reduce the init to make every rank's slice identical — the experts are
        supposed to be different. Torch's default rng gives different numbers per rank
        as long as the torch seed is different, but we currently set the same seed
        (compute_init uses torch.manual_seed(42)) so actually all ranks produce
        identical experts here. That's fine: it just means training has to break the
        symmetry via different routing traffic per rank, which it does naturally.
        """
        s = (3.0 ** 0.5) * self.n_embd ** -0.5
        torch.nn.init.uniform_(self.router_weight, -s * 0.02, s * 0.02)
        torch.nn.init.uniform_(self.w_fc, -s * 0.4, s * 0.4)
        torch.nn.init.zeros_(self.w_proj)
        if self.ws_fc is not None:
            torch.nn.init.uniform_(self.ws_fc, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(self.ws_proj)

    # -------------------------------------------------------------------------
    # Forward — replicated (single-GPU / single-rank / EP-disabled) path
    # -------------------------------------------------------------------------
    def _forward_replicated(self, x_flat, topk_probs, topk_idx, N, E, k, D):
        # Padded-capacity dispatch (static shapes).
        capacity = max(1, int(math.ceil(self.capacity_factor * k * N / E)))

        flat_idx = topk_idx.reshape(-1)
        flat_probs = topk_probs.reshape(-1)
        flat_tok = torch.arange(N, device=x_flat.device).repeat_interleave(k)

        sort_order = flat_idx.argsort(stable=True)
        sorted_idx = flat_idx[sort_order]
        sorted_tok = flat_tok[sort_order]
        sorted_probs = flat_probs[sort_order]

        counts = torch.zeros(E, dtype=torch.long, device=x_flat.device)
        counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx))
        starts = F.pad(counts[:-1].cumsum(0), (1, 0))
        pos_within = torch.arange(N * k, device=x_flat.device) - starts[sorted_idx]
        keep = (pos_within < capacity).to(x_flat.dtype)
        pos_clamped = pos_within.clamp(max=capacity - 1)

        flat_scatter_idx = sorted_idx * capacity + pos_clamped

        expert_inputs_flat = torch.zeros(E * capacity, D, dtype=x_flat.dtype, device=x_flat.device)
        expert_inputs_flat.scatter_add_(
            0,
            flat_scatter_idx.unsqueeze(-1).expand(-1, D),
            x_flat[sorted_tok] * keep.unsqueeze(-1),
        )
        expert_inputs = expert_inputs_flat.view(E, capacity, D)

        w_fc = self.w_fc.to(x_flat.dtype)
        w_proj = self.w_proj.to(x_flat.dtype)
        if self.moe_expert_fp8:
            # Lazy import to avoid circular deps at module load time.
            # Prefer the fused grouped-matmul path (single cuBLAS kernel) when available;
            # fall back to the per-expert Python loop otherwise.
            try:
                from nanochat.fp8 import fp8_expert_bmm_grouped as _bmm_fn
            except Exception:
                from nanochat.fp8 import fp8_expert_bmm as _bmm_fn
            hidden = _bmm_fn(expert_inputs, w_fc)
            hidden = F.relu(hidden).square()
            expert_output = _bmm_fn(hidden, w_proj)
        else:
            hidden = torch.bmm(expert_inputs, w_fc)
            hidden = F.relu(hidden).square()
            expert_output = torch.bmm(hidden, w_proj)

        expert_output_flat = expert_output.view(E * capacity, D)
        gathered = expert_output_flat[flat_scatter_idx]
        weighted = gathered * (sorted_probs * keep).unsqueeze(-1)

        routed_out = torch.zeros_like(x_flat)
        routed_out.scatter_add_(
            0,
            sorted_tok.unsqueeze(-1).expand(-1, D),
            weighted,
        )
        return routed_out

    # -------------------------------------------------------------------------
    # Forward — Expert Parallel path (all_to_all_single)
    # -------------------------------------------------------------------------
    def _forward_ep(self, x_flat, topk_probs, topk_idx, N, E, k, D):
        ws = self.world_size
        Ep = self.E_per_rank  # experts hosted by this rank
        # Per-rank per-expert capacity. After all-to-all, each expert aggregates tokens
        # from `ws` ranks, so the actual per-expert compute buffer is (ws*capacity, D).
        capacity = max(1, int(math.ceil(self.capacity_factor * k * N / E)))

        flat_idx = topk_idx.reshape(-1)          # (N*k,) global expert id
        flat_probs = topk_probs.reshape(-1)       # (N*k,)
        flat_tok = torch.arange(N, device=x_flat.device).repeat_interleave(k)

        sort_order = flat_idx.argsort(stable=True)
        sorted_idx = flat_idx[sort_order]
        sorted_tok = flat_tok[sort_order]
        sorted_probs = flat_probs[sort_order]

        counts = torch.zeros(E, dtype=torch.long, device=x_flat.device)
        counts.scatter_add_(0, flat_idx, torch.ones_like(flat_idx))
        starts = F.pad(counts[:-1].cumsum(0), (1, 0))
        pos_within = torch.arange(N * k, device=x_flat.device) - starts[sorted_idx]
        keep = (pos_within < capacity).to(x_flat.dtype)
        pos_clamped = pos_within.clamp(max=capacity - 1)

        flat_scatter_idx = sorted_idx * capacity + pos_clamped

        # Local dispatch buffer sized for the FULL global expert list (E * capacity).
        # The first (E_per_rank * capacity) slots target experts hosted by rank 0,
        # the next chunk targets rank 1's experts, etc. — so it's already organized
        # for a direct all_to_all_single.
        send_buffer = torch.zeros(E * capacity, D, dtype=x_flat.dtype, device=x_flat.device)
        send_buffer.scatter_add_(
            0,
            flat_scatter_idx.unsqueeze(-1).expand(-1, D),
            x_flat[sorted_tok] * keep.unsqueeze(-1),
        )

        # All-to-all (differentiable): after this call, this rank has received chunks from
        # every sender, each of size (E_per_rank * capacity, D). Backward is another
        # all_to_all that returns gradients to the original sender ranks.
        recv_buffer = _all_to_all_single_autograd(send_buffer)

        # Reshape to (ws, E_per_rank, capacity, D), transpose to (E_per_rank, ws, cap, D),
        # then flatten the (ws, cap) dims so each local expert sees ws*capacity tokens.
        local_inputs = recv_buffer.view(ws, Ep, capacity, D).transpose(0, 1).contiguous()
        local_inputs = local_inputs.view(Ep, ws * capacity, D)

        # Local expert forward on my E_per_rank experts.
        w_fc = self.w_fc.to(x_flat.dtype)       # (Ep, D, H)
        w_proj = self.w_proj.to(x_flat.dtype)   # (Ep, H, D)
        hidden = torch.bmm(local_inputs, w_fc)
        hidden = F.relu(hidden).square()
        local_outputs = torch.bmm(hidden, w_proj)  # (Ep, ws*cap, D)

        # Reverse the transpose to prepare for all_to_all back:
        return_buffer = local_outputs.view(Ep, ws, capacity, D).transpose(0, 1).contiguous()
        return_buffer = return_buffer.view(ws * Ep * capacity, D)

        back_buffer = _all_to_all_single_autograd(return_buffer)
        # back_buffer layout matches the original send_buffer layout: (E * capacity, D).

        # Combine: gather outputs back to each token's position, weighted by its gate prob.
        gathered = back_buffer[flat_scatter_idx]                           # (N*k, D)
        weighted = gathered * (sorted_probs * keep).unsqueeze(-1)          # zero for overflow
        routed_out = torch.zeros_like(x_flat)
        routed_out.scatter_add_(
            0,
            sorted_tok.unsqueeze(-1).expand(-1, D),
            weighted,
        )
        return routed_out

    def forward(self, x):
        B, T, D = x.shape
        N = B * T
        k = self.top_k
        E = self.num_experts
        x_flat = x.view(N, D)

        # Shared-expert branch: always active, no routing, fully replicated.
        shared_out = torch.zeros_like(x_flat)
        if self.ws_fc is not None:
            ws_fc = self.ws_fc.to(x.dtype)
            ws_proj = self.ws_proj.to(x.dtype)
            for s in range(self.num_shared_experts):
                h = x_flat @ ws_fc[s]
                h = F.relu(h).square()
                shared_out = shared_out + h @ ws_proj[s]

        # Router: replicated, produces consistent routing across ranks (each rank routes
        # only its own tokens, but uses the same router weights).
        router_logits = F.linear(x_flat, self.router_weight.to(x.dtype))
        routing_probs = F.softmax(router_logits.float(), dim=-1)
        topk_probs_f, topk_idx = routing_probs.topk(k, dim=-1)
        topk_probs_f = topk_probs_f / (topk_probs_f.sum(dim=-1, keepdim=True) + 1e-9)
        topk_probs = topk_probs_f.to(x.dtype)

        # Aux loss: computed locally per rank (no all_reduce — gradient contributions are
        # added across ranks anyway by the grad reduction in DistMuonAdamW).
        with torch.no_grad():
            one_hot_top1 = F.one_hot(topk_idx[:, 0], num_classes=E).to(routing_probs.dtype)
            f = one_hot_top1.mean(dim=0)
        p = routing_probs.mean(dim=0)
        aux_loss = self.aux_loss_coef * float(E) * (f * p).sum()

        # Route + expert compute.
        if self.expert_parallel:
            routed_out = self._forward_ep(x_flat, topk_probs, topk_idx, N, E, k, D)
        else:
            routed_out = self._forward_replicated(x_flat, topk_probs, topk_idx, N, E, k, D)

        y = (routed_out + shared_out).view(B, T, D)
        return y, aux_loss
