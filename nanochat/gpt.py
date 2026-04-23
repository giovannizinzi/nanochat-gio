"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW
from nanochat.moe import MoE

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # MoE config. num_experts <= 1 disables MoE entirely (falls back to dense MLP).
    num_experts: int = 1
    top_k: int = 2
    num_shared_experts: int = 0
    capacity_factor: float = 1.25
    moe_aux_loss_coef: float = 0.01
    expert_hidden_dim: int = -1  # -1 = auto (compute-matched to dense active FFN)
    expert_parallel: bool = False  # if True (+ num_experts % world_size == 0), shard experts across GPUs
    moe_expert_fp8: bool = False   # if True, use FP8 matmuls for routed expert bmm (per-expert tensorwise scaling)
    # Layers [0, moe_first_layer) stay dense (MLP); layers [moe_first_layer, n_layer) are MoE.
    # Paper arxiv.org/abs/2506.12119 Table 6: moe_first_layer=1 ("1dense + SE") beats fully-MoE.
    moe_first_layer: int = 0
    # Expert-choice routing (Zhou et al. 2022 NeurIPS): experts pick top-C tokens instead of
    # tokens picking top-k experts. Perfect load balance by construction; no aux loss, no drops.
    moe_expert_choice: bool = False
    # Aux-loss-free load balancing (DeepSeek-V3, arxiv 2408.15664). Per-expert bias updated
    # each step by sign(avg_load − load_e). Replaces Switch aux loss without gradient noise.
    moe_auxfree_bias: bool = False
    moe_auxfree_bias_lr: float = 1e-3
    # Gate-prob scale before combining expert outputs (DeepSeek-V3: 2.827).
    moe_routed_scaling: float = 1.0
    # ScatterMoE Triton kernels (Tan et al. 2024) — drop-in replacement for the replicated
    # dispatch+bmm+combine path. Saves VRAM (no padded dispatch buffer) and keeps MFU at E>=16.
    moe_scattermoe: bool = False
    # Gradient checkpointing for MoE blocks: recomputes FFN activations in backward instead
    # of storing them. Breaks the K*D_e activation-memory ceiling (lets d22 run E=16/K=2/D_e=2048
    # on 80GB H100s). Default False = bit-identical to previous behavior.
    moe_grad_checkpoint: bool = False
    # Dense FFN activation. "relu2" (default) = Primer-style ReLU², keeps prior behavior.
    # "swiglu" = SwiGLU (Shazeer 2020, arxiv 2002.05202; used in Llama/Qwen/DeepSeek).
    # Applies only to the dense MLP path; MoE experts retain their own FFN.
    ffn_type: str = "relu2"
    # Router/logit-stabilization z-loss (Chowdhery et al. PaLM, arxiv 2204.02311 §5):
    #   aux = z_loss_coef * mean((logsumexp(logits))**2)
    # Keeps log-partition Z close to 0, which stabilizes bf16/fp16 training. 0.0 = disabled
    # (bit-identical to prior behavior). Typical value: 1e-4.
    z_loss_coef: float = 0.0
    # NoPE (Haviv et al. 2022, arxiv 2203.16634): skip rotary embeddings; rely on the causal
    # mask to encode position implicitly. Saves ~1-2% per step (no rope tensor materialization).
    use_rope: bool = True
    # Chunked cross-entropy: chunk logits+CE computation along the sequence dimension to
    # avoid materializing a (B, T, V) fp32 logits tensor at once. 0 = disabled (one-shot CE).
    # Typical values: 128 or 256. Loss is bit-identical to the one-shot path.
    chunked_ce_chunk_size: int = 0
    # MLA (DeepSeek-V2 Multi-head Latent Attention, arxiv 2405.04434). When > 0, K and V
    # are computed from a shared low-rank latent: x -> c_kv_a -> (B,T,r) -> {c_k_b, c_v_b}
    # -> (B, T, n_kv_head, head_dim). Simplified variant without the decoupled RoPE head;
    # RoPE applies to full Q and K as usual. 0 = disabled (standard MHA/GQA via c_k/c_v).
    mla_lora_rank: int = 0


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        # MLA: K and V decoded from a shared low-rank latent of width mla_lora_rank.
        # Otherwise: standard MHA/GQA with direct c_k, c_v projections.
        self.use_mla = getattr(config, "mla_lora_rank", 0) > 0
        if self.use_mla:
            self.mla_rank = config.mla_lora_rank
            self.c_kv_a = Linear(self.n_embd, self.mla_rank, bias=False)
            self.c_k_b = Linear(self.mla_rank, self.n_kv_head * self.head_dim, bias=False)
            self.c_v_b = Linear(self.mla_rank, self.n_kv_head * self.head_dim, bias=False)
        else:
            self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache, doc_mask=None):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        if self.use_mla:
            # DeepSeek-V2 style: shared low-rank latent, then per-KV decompression.
            latent = self.c_kv_a(x)  # (B, T, r)
            k = self.c_k_b(latent).view(B, T, self.n_kv_head, self.head_dim)
            v = self.c_v_b(latent).view(B, T, self.n_kv_head, self.head_dim)
        else:
            k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
            v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding.
        # cos_sin is None when config.use_rope=False (NoPE — causal mask encodes position).
        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window. When doc_mask is
            # provided, flash_attn_func forces the SDPA path with cross-doc blocking added.
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size, doc_mask=doc_mask)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


def _swiglu_hidden_dim(n_embd, multiple_of=64):
    """SwiGLU compute-matched hidden dim: (8/3)*n_embd rounded up to `multiple_of`.
    Llama/DeepSeek use this to keep params ~ matched to 4*n_embd ReLU² FFN (two input
    projections vs one, so 2*(8/3) ≈ 16/3 vs 4+4/3 ≈ not quite equal — close enough)."""
    hidden = int(8 * n_embd / 3)
    hidden = ((hidden + multiple_of - 1) // multiple_of) * multiple_of
    return hidden


class MLP(nn.Module):
    """Dense FFN. Switches activation via config.ffn_type.
    - "relu2": c_fc(D -> 4D) -> ReLU² -> c_proj(4D -> D)   (default, Primer-style)
    - "swiglu": c_fc(D -> 2H) -> chunk gate,value -> SiLU(gate)*value -> c_proj(H -> D)
                where H = round((8/3)*D, 64)  (Shazeer 2020 / Llama convention)
    Parameter names c_fc and c_proj are preserved in both modes so init_weights() and
    existing checkpoints that key on these names keep working.
    """
    def __init__(self, config):
        super().__init__()
        self.ffn_type = config.ffn_type
        if self.ffn_type == "relu2":
            self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
            self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)
        elif self.ffn_type == "swiglu":
            hidden = _swiglu_hidden_dim(config.n_embd)
            self.hidden = hidden
            # c_fc projects to 2*hidden: first half is the gate, second half is the value.
            self.c_fc = Linear(config.n_embd, 2 * hidden, bias=False)
            self.c_proj = Linear(hidden, config.n_embd, bias=False)
        else:
            raise ValueError(f"Unknown ffn_type: {self.ffn_type} (expected 'relu2' or 'swiglu')")

    def forward(self, x):
        if self.ffn_type == "relu2":
            x = self.c_fc(x)
            x = F.relu(x).square()
            x = self.c_proj(x)
            return x
        else:  # swiglu
            g, v = self.c_fc(x).chunk(2, dim=-1)
            x = F.silu(g) * v
            return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.use_moe = config.num_experts > 1 and layer_idx >= config.moe_first_layer
        self.mlp = MoE(config) if self.use_moe else MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, doc_mask=None):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache, doc_mask=doc_mask)
        if self.use_moe:
            mlp_out, aux = self.mlp(norm(x))
            x = x + mlp_out
        else:
            x = x + self.mlp(norm(x))
            aux = torch.zeros((), device=x.device, dtype=torch.float32)
        return x, aux


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            if block.attn.use_mla:
                # MLA: latent projection + per-K/V decompression. Init all with the standard scale.
                torch.nn.init.uniform_(block.attn.c_kv_a.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_k_b.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v_b.weight, -s, s)
            else:
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            if block.use_moe:
                block.mlp.init_weights()  # MoE does its own init (zero c_proj => zero contribution at init)
            else:
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
                torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized 
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)

        For MoE: only top_k / num_experts of routed expert params see a given token per forward.
        Shared experts and the router are always active (included in the active count).
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel())
        # Exclude inactive MoE expert params: (num_experts - top_k) / num_experts of routed experts.
        # Only layers >= moe_first_layer have experts; earlier layers are dense MLP and contribute nothing here.
        moe_inactive = 0
        num_moe_layers = max(0, self.config.n_layer - self.config.moe_first_layer) if self.config.num_experts > 1 else 0
        if num_moe_layers > 0:
            moe = self.transformer.h[self.config.moe_first_layer].mlp
            expert_hidden = moe.expert_hidden_dim
            inactive_per_layer = (self.config.num_experts - self.config.top_k) * 2 * self.config.n_embd * expert_hidden
            moe_inactive = inactive_per_layer * num_moe_layers
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude - moe_inactive) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        For MoE, 'active_*' fields count only the parameters active per token
        (top_k out of num_experts routed experts, plus shared experts).
        Following DeepSeek convention of reporting both total and active params.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers).
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        # Under Expert Parallel, this rank only holds E/world_size routed experts. The local
        # transformer_matrices count is therefore missing the (world_size-1)/world_size of
        # routed experts that live on other ranks. Upgrade to a "global" count so downstream
        # scaling-law math (active_* etc.) sees the full logical model.
        moe_inactive = 0
        ep_missing = 0  # routed-expert params held on OTHER ranks under EP
        num_moe_layers = max(0, self.config.n_layer - self.config.moe_first_layer) if self.config.num_experts > 1 else 0
        if num_moe_layers > 0:
            moe = self.transformer.h[self.config.moe_first_layer].mlp
            expert_hidden = moe.expert_hidden_dim
            per_expert_ffn = 2 * self.config.n_embd * expert_hidden
            # Global routed-expert params (across all ranks if EP is on)
            routed_params_per_layer_global = self.config.num_experts * per_expert_ffn
            # Inactive per token = (num_experts - top_k) fraction of routed params
            inactive_per_layer = routed_params_per_layer_global - (self.config.top_k * per_expert_ffn)
            moe_inactive = inactive_per_layer * num_moe_layers
            if getattr(moe, "expert_parallel", False):
                local_routed_per_layer = moe.E_per_rank * per_expert_ffn
                ep_missing = (routed_params_per_layer_global - local_routed_per_layer) * num_moe_layers
        transformer_matrices_global = transformer_matrices + ep_missing
        total_global = wte + value_embeds + lm_head + transformer_matrices_global + scalars
        active_transformer_matrices = transformer_matrices_global - moe_inactive
        active_total = total_global - moe_inactive
        # Expose globals as the canonical "total" values — matches the replicated case and is
        # what scaling-law analysis wants.
        transformer_matrices = transformer_matrices_global
        total = total_global
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'active_transformer_matrices': active_transformer_matrices,
            'scalars': scalars,
            'moe_inactive': moe_inactive,
            'total': total,
            'active_total': active_total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5, muon_qk_clip_tau=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        # MoE router weights want AdamW (flexible triage), not Muon (which orthogonalizes).
        # Pull them out of matrix_params before Muon grouping.
        router_params = []
        # Under Expert Parallel mode, each rank holds different routed-expert weights.
        # Those must NOT be collective-reduced during the optimizer step (unlike
        # replicated matrix params). Collect them separately so setup_optimizer can
        # mark the Muon group as non-replicated.
        ep_expert_params = []
        if self.config.num_experts > 1:
            for block in self.transformer.h:
                if not block.use_moe:  # skip dense layers under mixed-layout (1dense+SE)
                    continue
                router_params.append(block.mlp.router_weight)
                if getattr(block.mlp, "expert_parallel", False):
                    ep_expert_params.append(block.mlp.w_fc)
                    ep_expert_params.append(block.mlp.w_proj)
            exclude_ids = {id(p) for p in router_params + ep_expert_params}
            matrix_params = [p for p in matrix_params if id(p) not in exclude_ids]
        # Collect attention c_q/c_k params so we can put them in a dedicated Muon
        # group tagged for QK-Clip (MuonClip, Kimi K2 arxiv 2507.20534 §A). We pull
        # them out of the generic matrix_params only when qk_clip is actually
        # enabled, so default behavior (tau=0) is bit-identical to before.
        qk_params = []
        if muon_qk_clip_tau > 0.0:
            qk_param_ids = set()
            for block in self.transformer.h:
                qk_params.append(block.attn.c_q.weight)
                qk_params.append(block.attn.c_k.weight)
                qk_param_ids.add(id(block.attn.c_q.weight))
                qk_param_ids.add(id(block.attn.c_k.weight))
            matrix_params = [p for p in matrix_params if id(p) not in qk_param_ids]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(qk_params) + len(router_params) + len(ep_expert_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # MoE router: AdamW, small params across layers. Keep same defaults as the matrix AdamW group below.
        if router_params:
            param_groups.append(dict(
                kind='adamw', params=router_params,
                lr=matrix_lr * dmodel_lr_scale,
                betas=(0.8, 0.96), eps=1e-10, weight_decay=0.0,
            ))
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
                replicated=True,
            ))
        # Dedicated Muon group for QK params when MuonClip is enabled (arxiv 2507.20534 §A).
        # Split by shape to respect Muon's same-shape-stacking assumption (GQA gives c_q
        # and c_k different shapes when n_kv_head < n_head).
        if qk_params:
            for shape in sorted({p.shape for p in qk_params}):
                group_params = [p for p in qk_params if p.shape == shape]
                param_groups.append(dict(
                    kind='muon', params=group_params, lr=matrix_lr,
                    momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
                    replicated=True,
                    is_qk=True,
                    qk_tau=muon_qk_clip_tau,
                ))
        # Expert Parallel routed-expert groups: each rank holds a unique slice of experts,
        # so the optimizer must NOT reduce gradients across ranks. Muon still orthogonalizes
        # each (D, H) slice independently via its batched polar-express (leading dims OK).
        if ep_expert_params:
            for shape in sorted({p.shape for p in ep_expert_params}):
                group_params = [p for p in ep_expert_params if p.shape == shape]
                param_groups.append(dict(
                    kind='muon', params=group_params, lr=matrix_lr,
                    momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
                    replicated=False,
                ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', doc_lens=None):
        B, T = idx.size()

        # Build per-batch same-document mask from doc_lens if provided.
        # doc_lens: (B, max_segs) int32, segment lengths summing to row_capacity (T+1 from dataloader).
        # We interpret boundaries against the length-T `idx` view (row[:, :-1] in the dataloader).
        # Result: same_doc (B, T, T) bool, True where query pos q_i and key pos k_j share a doc.
        doc_mask = None
        if doc_lens is not None and kv_cache is None:
            doc_lens = doc_lens.to(device=idx.device, dtype=torch.int32)  # defensive: ensure aligned with idx
            cum = doc_lens.cumsum(dim=-1).to(torch.int32)  # (B, max_segs); cumsum upcasts to int64 by default
            positions = torch.arange(T, device=idx.device, dtype=torch.int32)  # (T,)
            # doc_id[b, p] = number of segment-end positions <= p; equivalently, the segment index
            # containing position p. Broadcast: (1, T, 1) >= (B, 1, max_segs) -> (B, T, max_segs).
            doc_id = (positions[None, :, None] >= cum[:, None, :]).sum(dim=-1)  # (B, T)
            doc_mask = (doc_id[:, :, None] == doc_id[:, None, :])  # (B, T, T), True where same doc

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        # When config.use_rope=False (NoPE), pass None through to the attention layers so they
        # skip apply_rotary_emb entirely. Causal attention implicitly encodes position.
        if self.config.use_rope:
            assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
            assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
            assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
            T0 = 0 if kv_cache is None else kv_cache.get_pos()
            cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
        else:
            cos_sin = None

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        # Accumulate MoE aux loss across layers (zero-valued for dense blocks).
        aux_loss_total = torch.zeros((), device=x.device, dtype=torch.float32)
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x, block_aux = block(x, ve, cos_sin, self.window_sizes[i], kv_cache, doc_mask=doc_mask)
            aux_loss_total = aux_loss_total + block_aux
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits). Chunked path is bit-identical to the one-shot
        # path for mean CE, but avoids the (B, T, V) fp32 logits materialization.
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]

        chunk = self.config.chunked_ce_chunk_size
        use_chunked_ce = (
            chunk > 0 and targets is not None and loss_reduction == 'mean'
            and self.config.z_loss_coef == 0.0  # z_loss needs full logits; skip for now
        )

        if use_chunked_ce:
            # Loop over seq dim; compute logits + CE per chunk; aggregate as sum/count.
            B_, T_, _ = x.size()
            total_nats = torch.zeros((), dtype=torch.float32, device=x.device)
            total_count = torch.zeros((), dtype=torch.float32, device=x.device)
            for s in range(0, T_, chunk):
                e = min(s + chunk, T_)
                x_c = x[:, s:e, :]
                t_c = targets[:, s:e]
                lg = self.lm_head(x_c)
                lg = lg[..., :self.config.vocab_size].float()
                lg = softcap * torch.tanh(lg / softcap)
                loss_c = F.cross_entropy(
                    lg.reshape(-1, lg.size(-1)), t_c.reshape(-1),
                    ignore_index=-1, reduction='sum',
                )
                total_nats = total_nats + loss_c
                total_count = total_count + (t_c.reshape(-1) != -1).sum().float()
            loss = total_nats / total_count.clamp(min=1.0)
            # Skip the return-stack fork: we already have our scalar mean loss.
            return loss, aux_loss_total

        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            # PaLM-style z-loss on logits (arxiv 2204.02311 §5): penalize (logsumexp(logits))**2
            # to keep the log-partition Z near 0. Stabilizes bf16/fp16 training.
            # At z_loss_coef=0.0 (default) this branch is a no-op and behavior is bit-identical.
            if self.config.z_loss_coef > 0.0:
                log_Z = torch.logsumexp(logits, dim=-1)  # (B, T)
                if loss_reduction == 'mean':
                    z_loss = (log_Z ** 2).mean()
                elif loss_reduction == 'sum':
                    z_loss = (log_Z ** 2).sum()
                else:  # 'none': keep per-position, caller handles reduction
                    z_loss = (log_Z ** 2).view(-1)
                loss = loss + self.config.z_loss_coef * z_loss
            # Always return (ce_loss, aux_loss) pair. Caller sums them for backward and logs them separately.
            # When z_loss_coef > 0, the z_loss contribution is folded into the ce_loss return value.
            return loss, aux_loss_total
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
