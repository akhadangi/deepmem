"""DeepMem architecture adapted from the author's supplied research implementation.

The paper configuration is in configs/paper_r2.json. Optional PRISM and
vector-gated backbones are experimental features carried over from the
original source; the published paper reference uses scalar FastWeight.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@dataclass
class LosslessSubbandByteLadderConfig:
    """Causal lossless-subband byte ladder LM.

    The carrier stream is a fixed 8-bit binary representation of each byte.  The
    dyadic analysis transform over this carrier is exactly invertible in real
    arithmetic: every 2048-byte carrier sequence can be transformed to length
    1024 with doubled channels, then recursively down to min_length, without
    dropping information.  The semantic stream is learned and predictive; the
    carrier/detail sidebands preserve local byte fidelity and are injected back
    into the byte decoder with causal delays.
    """

    vocab_size: int = 256
    block_size: int = 2048
    d_model: int = 384
    n_heads: int = 6
    n_initial_layers: int = 2
    n_final_layers: int = 8
    max_width: int = 704
    max_heads: int = 11
    custom_sem_widths: Optional[List[int]] = None
    min_length: int = 16
    carrier_dim: int = 8
    lift_rank: int = 64
    carrier_proj_rank: int = 64
    mlp_ratio: float = 4.0
    ladder_mlp_ratio: float = 4.0
    ladder_mlp_ratios: Optional[List[float]] = None
    ladder_heads: Optional[List[int]] = None
    dropout: float = 0.0
    eps: float = 1e-6
    rope_theta: float = 10000.0
    grad_checkpointing: bool = False
    down_depths: Optional[List[int]] = None
    up_refine_depths: Optional[List[int]] = None
    multiscale_levels: Optional[List[int]] = None
    multiscale_init: float = 0.05
    carrier_scale_init: float = 0.1
    disable_carrier_to_sem: bool = False
    disable_multiscale: bool = False
    initial_mixer: str = "fastweight" # attention | fastweight | prism
    down_mixer: str = "fastweight"    # attention | fastweight | prism
    up_mixer: str = "attention"       # attention | fastweight | prism
    final_mixer: str = "attention"    # attention | fastweight | prism
    chunk_size: int = 64
    conv_kernel: int = 4
    prism_rank: int = 2               # rank of the PRISM content tape (0 disables the tape branch)
    fastweight_gate: str = "scalar"   # scalar (per-head) | vector (GLA-style per-key-dim) forget gates

    # Optional zero-loop TRM-style bottleneck reasoner. A dedicated FastWeight
    # mixer is shared across R iterative updates of a separate latent state z:
    #   z_0 = 0
    #   z_{r+1} = z_r + M(N(s + z_r))
    #   s_out = s + z_R
    # The base bottleneck representation s is held fixed during the R updates.
    bottleneck_shared_state: bool = False
    shared_state_steps: int = 2

    def __post_init__(self) -> None:
        if self.block_size < 2 or (self.block_size & (self.block_size - 1)) != 0:
            raise ValueError("block_size must be a power of two and >= 2")
        if self.min_length < 1 or (self.min_length & (self.min_length - 1)) != 0:
            raise ValueError("min_length must be a power of two and >= 1")
        if self.min_length >= self.block_size:
            raise ValueError("min_length must be smaller than block_size")
        if self.block_size % self.min_length != 0:
            raise ValueError("block_size must be divisible by min_length")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.carrier_dim != 8:
            raise ValueError("carrier_dim must be 8 for fixed lossless byte-bit carrier")
        if self.lift_rank <= 0 or self.carrier_proj_rank <= 0:
            raise ValueError("lift_rank and carrier_proj_rank must be positive")
        if self.custom_sem_widths is not None:
            if len(self.custom_sem_widths) != len(self.lengths):
                raise ValueError(f"custom_sem_widths must have {len(self.lengths)} entries")
            if int(self.custom_sem_widths[0]) != self.d_model:
                raise ValueError("custom_sem_widths[0] must equal d_model")
            for w in self.custom_sem_widths:
                if int(w) <= 0 or int(w) % 2 != 0:
                    raise ValueError("all custom semantic widths must be positive even integers")
        _mixer_kinds = {"attention", "fastweight", "prism"}
        if self.initial_mixer not in _mixer_kinds:
            raise ValueError("initial_mixer must be one of 'attention', 'fastweight', 'prism'")
        if self.down_mixer not in _mixer_kinds:
            raise ValueError("down_mixer must be one of 'attention', 'fastweight', 'prism'")
        if self.up_mixer not in _mixer_kinds:
            raise ValueError("up_mixer must be one of 'attention', 'fastweight', 'prism'")
        if self.final_mixer not in _mixer_kinds:
            raise ValueError("final_mixer must be one of 'attention', 'fastweight', 'prism'")
        if self.prism_rank < 0:
            raise ValueError("prism_rank must be >= 0 (0 disables the content tape)")
        if self.fastweight_gate not in {"scalar", "vector"}:
            raise ValueError("fastweight_gate must be 'scalar' or 'vector'")
        if self.shared_state_steps <= 0:
            raise ValueError("shared_state_steps must be positive")
        if self.bottleneck_shared_state and self.down_mixer != "fastweight":
            raise ValueError("bottleneck_shared_state currently requires --down_mixer fastweight")
        if self.chunk_size <= 0 or (self.chunk_size & (self.chunk_size - 1)) != 0:
            raise ValueError("chunk_size must be a positive power of two")
        if self.conv_kernel <= 0:
            raise ValueError("conv_kernel must be positive")

    @property
    def lengths(self) -> List[int]:
        xs: List[int] = []
        length = self.block_size
        while length >= self.min_length:
            xs.append(length)
            if length == self.min_length:
                break
            length //= 2
        return xs

    @property
    def n_levels(self) -> int:
        return len(self.lengths)

    @property
    def sem_widths(self) -> List[int]:
        if self.custom_sem_widths is not None:
            return [int(x) for x in self.custom_sem_widths]
        # Default: preserve the original coarse-detail width shape: quickly reach a
        # wide semantic stream, then stay flat. Use --sem_widths for an explicit
        # down-the-pyramid width schedule.
        return [min(self.d_model * (2**i), self.max_width) for i, _ in enumerate(self.lengths)]

    @property
    def carrier_widths(self) -> List[int]:
        return [self.carrier_dim * (2**i) for i, _ in enumerate(self.lengths)]

    @property
    def resolved_down_depths(self) -> List[int]:
        n = self.n_levels - 1
        if self.down_depths is not None:
            if len(self.down_depths) != n:
                raise ValueError(f"down_depths must have {n} entries")
            return [max(0, int(x)) for x in self.down_depths]
        return [1 for _ in range(n)]

    @property
    def resolved_up_refine_depths(self) -> List[int]:
        n = self.n_levels - 1
        if self.up_refine_depths is not None:
            if len(self.up_refine_depths) != n:
                raise ValueError(f"up_refine_depths must have {n} entries")
            return [max(0, int(x)) for x in self.up_refine_depths]
        return [0 for _ in range(n)]


    @property
    def resolved_ladder_mlp_ratios(self) -> List[float]:
        n = self.n_levels - 1
        if self.ladder_mlp_ratios is not None:
            if len(self.ladder_mlp_ratios) != n:
                raise ValueError(f"ladder_mlp_ratios must have {n} entries")
            return [float(x) for x in self.ladder_mlp_ratios]
        return [float(self.ladder_mlp_ratio) for _ in range(n)]

    @property
    def resolved_ladder_heads(self) -> List[int]:
        n = self.n_levels - 1
        if self.ladder_heads is not None:
            if len(self.ladder_heads) != n:
                raise ValueError(f"ladder_heads must have {n} entries")
            heads = [int(x) for x in self.ladder_heads]
            for h in heads:
                if h <= 0:
                    raise ValueError("all ladder_heads entries must be positive")
            return heads
        return [int(self.max_heads) for _ in range(n)]

    @property
    def resolved_multiscale_levels(self) -> List[int]:
        if self.disable_multiscale:
            return []
        if self.multiscale_levels is not None:
            return [int(x) for x in self.multiscale_levels if int(x) > 0 and int(x) < self.n_levels]
        return list(range(1, self.n_levels))


def _parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return []
    return [int(x) for x in value.split(",") if x.strip()]


def _parse_float_list(value: Optional[str]) -> Optional[List[float]]:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return []
    return [float(x) for x in value.split(",") if x.strip()]


def _best_heads(width: int, preferred: int) -> int:
    # Prefer attention head dimensions that are even, so RoPE can rotate pairs.
    for h in range(min(preferred, width), 0, -1):
        if width % h == 0 and (width // h) % 2 == 0:
            return h
    for h in range(min(preferred, width), 0, -1):
        if width % h == 0:
            return h
    return 1


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


_ROPE_CACHE: Dict[Tuple[int, int, torch.device, torch.dtype, float], Tuple[torch.Tensor, torch.Tensor]] = {}


def _build_rope_cache(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    key = (seq_len, head_dim, device, dtype, theta)
    hit = _ROPE_CACHE.get(key)
    if hit is not None:
        return hit
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.repeat_interleave(freqs, 2, dim=-1)
    cos = emb.cos().to(dtype=dtype)[None, None, :, :]
    sin = emb.sin().to(dtype=dtype)[None, None, :, :]
    _ROPE_CACHE[key] = (cos, sin)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float, rope_theta: float):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires even head_dim")
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.out._residual_proj = True  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(bsz, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        cos, sin = _build_rope_cache(seq_len, self.head_dim, x.device, q.dtype, self.rope_theta)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.out(y)


class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float, dropout: float):
        super().__init__()
        hidden = int(ratio * dim)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, dim, bias=False)
        self.fc2._residual_proj = True  # type: ignore[attr-defined]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x), approximate="tanh")))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float, dropout: float, eps: float, rope_theta: float):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=eps)
        self.attn = CausalSelfAttention(dim, n_heads, dropout, rope_theta)
        self.norm2 = RMSNorm(dim, eps=eps)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def _round_up_to_multiple(x: int, multiple: int) -> int:
    return ((int(x) + multiple - 1) // multiple) * multiple


def _valid_chunk_size(seq_len: int, preferred: int) -> int:
    """Choose the largest power-of-two chunk <= preferred that divides seq_len."""
    c = min(int(preferred), int(seq_len))
    # force power-of-two at or below c
    c = 1 << (c.bit_length() - 1)
    while c > 1 and seq_len % c != 0:
        c //= 2
    return max(1, c)


_CAUSAL_MASK_CACHE: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}


def _cached_causal_mask(n: int, diagonal: int, device: torch.device) -> torch.Tensor:
    key = (n, diagonal, device)
    mask = _CAUSAL_MASK_CACHE.get(key)
    if mask is None:
        mask = torch.ones(n, n, device=device, dtype=torch.bool).tril(diagonal)
        _CAUSAL_MASK_CACHE[key] = mask
    return mask


def _chunked_gated_linear_recurrence_vector(
    q: torch.Tensor,           # [B, H, L, D]
    k: torch.Tensor,           # [B, H, L, D]
    v: torch.Tensor,           # [B, H, L, D]
    log_decay: torch.Tensor,   # [B, H, L, D], bounded in (-0.1, 0)
    chunk_size: int,
) -> torch.Tensor:
    """GLA-style vector-gated linear recurrence (per-key-dimension forgetting):

        S_t = Diag(exp(log_decay_t)) S_{t-1} + k_t v_t^T
        o_t = S_t^T q_t

    Same chunked-parallel, loop-free structure as the scalar path below, with
    the decay applied per key dimension. Gates are bounded below at -0.1/step
    (half-life >= ~7 steps), so exp(-G) within a chunk of size C stays under
    e^{0.1*C}: plain fp32 is safe with no log-space machinery. Forgetting
    faster than that is the depthwise conv's job, which covers short-range
    structure exactly. The inter-chunk pass generalizes the scalar matmul to
    an einsum over a [N, N, D] decay tensor (still tiny; still no loop).
    """
    B, H, L, D = q.shape
    C = _valid_chunk_size(L, chunk_size)
    if L % C != 0:
        raise ValueError(f"L={L} must be divisible by chosen chunk_size={C}")
    N = L // C
    q = q.view(B, H, N, C, D)
    k = k.view(B, H, N, C, D)
    v = v.view(B, H, N, C, D)
    G = log_decay.view(B, H, N, C, D).cumsum(dim=3)           # chunk-local cumsum
    q_in = q * torch.exp(G)
    k_in = k * torch.exp(-G)                                  # bounded by the gate floor
    scores = torch.matmul(q_in, k_in.transpose(-2, -1))       # [B, H, N, C, C]
    scores = scores.masked_fill(~_cached_causal_mask(C, 0, q.device), 0.0)
    o_intra = torch.matmul(scores, v)
    g_total = G[..., -1, :]                                   # [B, H, N, D]
    k_dec = k * torch.exp(g_total.unsqueeze(3) - G)
    KV = torch.matmul(k_dec.transpose(-2, -1), v)             # [B, H, N, D, D]
    Gc = g_total.cumsum(dim=2)
    P = Gc - g_total
    Dm = torch.exp(P.unsqueeze(3) - Gc.unsqueeze(2))          # [B, H, N, N, D]
    Dm = Dm.masked_fill(~_cached_causal_mask(N, -1, q.device).view(1, 1, N, N, 1), 0.0)
    states_in = torch.einsum("bhnmk,bhmkv->bhnkv", Dm, KV)
    o_inter = torch.einsum("bhnck,bhnkv->bhncv", q_in, states_in)
    return (o_intra + o_inter).reshape(B, H, L, D)


def chunked_gated_linear_recurrence(
    q: torch.Tensor,           # [B, H, L, D]
    k: torch.Tensor,           # [B, H, L, D]
    v: torch.Tensor,           # [B, H, L, D]
    log_decay: torch.Tensor,   # [B, H, L], non-positive
    chunk_size: int,
) -> torch.Tensor:
    """Chunked-parallel causal gated linear recurrence.

    S_t = exp(log_decay_t) * S_{t-1} + k_t v_t^T
    o_t = S_t^T q_t

    This is copied in spirit from the fastweight baseline, but parameterized by
    per-level width/head count so it can be used inside the lossless ladder.
    """
    if log_decay.dim() == 4:
        return _chunked_gated_linear_recurrence_vector(q, k, v, log_decay, chunk_size)
    if log_decay.dim() != 3:
        raise ValueError("scalar gates must have shape [batch, heads, length]")
    B, H, L, D = q.shape
    C = _valid_chunk_size(L, chunk_size)
    if L % C != 0:
        raise ValueError(f"L={L} must be divisible by chosen chunk_size={C}")
    N = L // C

    q = q.view(B, H, N, C, D)
    k = k.view(B, H, N, C, D)
    v = v.view(B, H, N, C, D)
    g = log_decay.view(B, H, N, C)

    G = g.cumsum(dim=-1)

    G_i = G.unsqueeze(-1)
    G_j = G.unsqueeze(-2)
    log_decay_mat = G_i - G_j
    causal = _cached_causal_mask(C, 0, q.device)
    log_decay_mat = log_decay_mat.masked_fill(~causal, float("-inf"))
    decay_mat = log_decay_mat.exp()

    scores = torch.matmul(q, k.transpose(-2, -1))
    scores = scores * decay_mat
    o_intra = torch.matmul(scores, v)

    g_total = G[..., -1]
    decay_to_end = (g_total.unsqueeze(-1) - G).exp()
    k_decayed = k * decay_to_end.unsqueeze(-1)
    KV = torch.matmul(k_decayed.transpose(-2, -1), v)

    # Exact loop-free inter-chunk states. With Gc = cumsum_n(g_total) and
    # P_n = Gc_{n-1} (exclusive cumsum), the state entering chunk n is
    #     states_in[n] = sum_{m<n} exp(P_n - Gc_m) * KV[m],
    # i.e. the same lower-triangular decay-matrix construction as the
    # intra-chunk path, applied at chunk granularity (N x N, tiny). All
    # exponents are <= 0 by construction, so this cannot overflow; masked
    # entries follow the -inf -> exp -> 0 pattern used above. Replaces the
    # former Python loop of N sequential state updates with one matmul.
    Gc = g_total.cumsum(dim=-1)                                   # [B, H, N]
    P = Gc - g_total                                              # [B, H, N]
    chunk_log_mat = P.unsqueeze(-1) - Gc.unsqueeze(-2)            # [B, H, N, N]
    chunk_causal = _cached_causal_mask(N, -1, q.device)
    chunk_log_mat = chunk_log_mat.masked_fill(~chunk_causal, float("-inf"))
    chunk_decay = chunk_log_mat.exp()
    states_in = torch.matmul(chunk_decay, KV.reshape(B, H, N, D * D)).view(B, H, N, D, D)

    q_decayed = q * G.exp().unsqueeze(-1)
    o_inter = torch.matmul(q_decayed, states_in)
    return (o_intra + o_inter).view(B, H, L, D)


class GatedLinearRecurrenceMixer(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float, eps: float, chunk_size: int, conv_kernel: int, gate_mode: str = "scalar"):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.chunk_size = int(chunk_size)
        self.conv_kernel = int(conv_kernel)
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj._residual_proj = True  # type: ignore[attr-defined]
        self.gate_mode = str(gate_mode)
        if self.gate_mode not in ("scalar", "vector"):
            raise ValueError(f"gate_mode must be 'scalar' or 'vector', got {gate_mode}")
        self.decay_proj = nn.Linear(dim, n_heads if self.gate_mode == "scalar" else dim, bias=True)
        nn.init.constant_(self.decay_proj.bias, -4.6 if self.gate_mode == "scalar" else -2.2)
        self.conv = nn.Conv1d(
            in_channels=3 * dim,
            out_channels=3 * dim,
            kernel_size=self.conv_kernel,
            groups=3 * dim,
            padding=self.conv_kernel - 1,
            bias=False,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H = self.n_heads
        Dh = self.head_dim
        qkv = self.qkv_proj(x)
        qkv = qkv.transpose(1, 2)
        qkv = self.conv(qkv)[..., :L]
        qkv = qkv.transpose(1, 2)
        qkv = F.silu(qkv)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, H, Dh).transpose(1, 2).contiguous()
        k = k.view(B, L, H, Dh).transpose(1, 2).contiguous()
        v = v.view(B, L, H, Dh).transpose(1, 2).contiguous()
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.gate_mode == "vector":
            raw = self.decay_proj(x).view(B, L, H, Dh).permute(0, 2, 1, 3)
            log_decay = -0.1 * torch.sigmoid(raw)      # bounded (-0.1, 0) per step
        else:
            log_decay = -F.softplus(self.decay_proj(x)).transpose(1, 2)
        device_type = x.device.type if x.device.type in ("cuda", "cpu") else "cuda"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            y = chunked_gated_linear_recurrence(q.float(), k.float(), v.float(), log_decay.float(), self.chunk_size)
        y = y.to(dtype=x.dtype)
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(y)


class SwiGLUMLP(nn.Module):
    """SwiGLU with GELU-equivalent parameter budget.

    hidden = round_16(2/3 * ratio * dim), so ratio=4 is close to a standard
    GELU MLP with hidden=4*dim in parameter count.
    """
    def __init__(self, dim: int, ratio: float, dropout: float):
        super().__init__()
        hidden = _round_up_to_multiple(int(2.0 * ratio * dim / 3.0), 16)
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)
        self.w_down._residual_proj = True  # type: ignore[attr-defined]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.dropout(F.silu(self.w_gate(x)) * self.w_up(x)))


class FastWeightBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float, dropout: float, eps: float, chunk_size: int, conv_kernel: int, gate_mode: str = "scalar"):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=eps)
        self.mixer = GatedLinearRecurrenceMixer(dim, n_heads, dropout, eps, chunk_size, conv_kernel, gate_mode)
        self.norm2 = RMSNorm(dim, eps=eps)
        self.mlp = SwiGLUMLP(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SharedMixerStateReasoner(nn.Module):
    """TRM-style mixer-only iterative state at the deepest semantic bottleneck.

    The encoder bottleneck representation ``base`` is not overwritten while the
    internal state is refined.  One FastWeight mixer parameter set is shared
    across every reasoning step:

        z_0     = 0
        u_r     = base + z_r
        z_{r+1} = z_r + Mixer(RMSNorm(u_r))
        output  = base + z_R

    There is no MLP in the reasoning cell and no per-step parameters.  Thus R
    changes compute, not parameter count.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        steps: int,
        dropout: float,
        eps: float,
        chunk_size: int,
        conv_kernel: int,
        gate_mode: str = "scalar",
    ):
        super().__init__()
        if steps <= 0:
            raise ValueError("steps must be positive")
        self.steps = int(steps)
        self.state_norm = RMSNorm(dim, eps=eps)
        self.mixer = GatedLinearRecurrenceMixer(
            dim, n_heads, dropout, eps, chunk_size, conv_kernel, gate_mode
        )

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        state = torch.zeros_like(base)
        for _ in range(self.steps):
            state = state + self.mixer(self.state_norm(base + state))
        return base + state


def signed_logspace_scan(z: torch.Tensor, log_decay: torch.Tensor) -> torch.Tensor:
    """Exact, fully parallel solve of the gated recurrence

        S_t = exp(log_decay_t) * S_{t-1} + z_t        (along the LAST dim)

    for signed inputs, with zero chunking and zero Python loops.

    z         : [B, H, C, L]  signed write stream, L contiguous (coalesced scans)
    log_decay : [B, H, 1, L]  non-positive per-step log forget factors (broadcast over C)
    returns   : [B, H, C, L]  the prefix states S_t, in float32

    Method (Heinsen-style log-space prefix scan): with G_t = cumsum(log_decay),
        S_t = exp(G_t) * sum_{s<=t} exp(-G_s) * z_s.
    exp(-G_s) grows without bound over long horizons, so the sum is evaluated
    with logcumsumexp (internally max-shifted, hence stable for any horizon and
    any data-dependent gate). Sign is handled by scanning the positive and
    negative parts of z in one stacked call. Intended for DIAGONAL states
    (C = head_dim), where the pos/neg doubling costs only ~2 activation
    tensors; do not feed materialized outer products through this.
    """
    G = log_decay.float().cumsum(dim=-1)                     # [B,H,1,L]
    zf = z.float()
    mag = torch.cat([zf.clamp(min=0.0), (-zf).clamp(min=0.0)], dim=2)   # [B,H,2C,L]
    # clamp before log so no -inf is ever materialized (keeps backward NaN-free);
    # the where() then routes exact zeros to a large finite negative instead.
    log_mag = torch.where(
        mag > 0,
        torch.log(mag.clamp(min=1e-38)),
        torch.full_like(mag, -1e30),
    )
    lcse = torch.logcumsumexp(log_mag - G, dim=-1)           # [B,H,2C,L]
    s = torch.exp(lcse + G)
    s_pos, s_neg = s.chunk(2, dim=2)
    return s_pos - s_neg


class PrismResonatorMixer(nn.Module):
    """PRISM v3: a learned causal resonator filterbank with a low-rank content tape.

    Token mixing is performed by a bank of M damped oscillators shared across
    the layer, mixed per channel:

        kernel_c(tau) = sum_m exp(-r_m tau) * (a_{c,m} cos(theta_m tau)
                                             + b_{c,m} sin(theta_m tau))
        y = (u * g)  (*)  kernel               (causal convolution)

    evaluated as ONE FFT convolution (pad to 2L, rfft/irfft). Pole rates r_m
    and frequencies theta_m are learnable, initialized on the dyadic ladder of
    the subband pyramid (pure decays with half-lives 4..2L, resonators with
    periods 4..L). Kernels are built by two tiny [D,M]x[M,L] GEMMs and
    l1-normalized per channel (BIBO-stable by construction). Data dependence
    enters through the elementwise pre-gate g and the output gate (Hyena
    style). Compute is GEMM/FFT-shaped end to end: no chunking, no Python
    loops, no logcumsumexp.

    Content-based recall is restored by a rank-r tape with FIXED bounded
    per-head decay (half-lives L/4 and L/16):

        T_t = lam_h T_{t-1} + beta_t (k_t u~_t^T),   o_t = q_t^T T_t

    Because lam is data-independent and its total decay over the block is at
    most 16 ln2 nats, the scan has the closed form
        T = e^{G} * cumsum(e^{-G} z),  G_t = t log lam,
    computed with ONE plain fp32 cumsum -- exact, stable, no log-space
    machinery. beta is input-dependent (sparse writes), k/q are RMS-normed,
    the write value u~ is per-position RMS-normed so tape writes are bounded.

    Streaming dual form for generation: the filterbank is a sum of M complex
    exponentials, i.e. exactly M diagonal recurrent states per channel; the
    tape carries H x r x Dh. Both update with one FMA per byte.

    Activation-memory note (level 0 of the reference config): FFT buffers are
    [B, D, 2L] fp32/complex64 transients (~130-270 MB) and the tape cumsum is
    [B, H*r*Dh, L] fp32 (~134 MB at r=2) -- below the fastweight kernel's
    per-chunk score/decay matrices.
    """

    def __init__(self, dim: int, n_heads: int, rank: int, dropout: float, eps: float, conv_kernel: int, max_len: int):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        if rank < 0:
            raise ValueError(f"prism rank must be >= 0, got {rank}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.rank = int(rank)
        self.max_len = int(max_len)
        self.conv_kernel = int(conv_kernel)
        self.eps = float(eps)

        qk_dim = n_heads * self.rank
        in_channels = 2 * dim + 2 * qk_dim          # pre-gate g, value u, tape k, tape q
        self.in_proj = nn.Linear(dim, in_channels, bias=False)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=self.conv_kernel,
            groups=in_channels,
            padding=self.conv_kernel - 1,
            bias=False,
        )

        # --- resonator bank -------------------------------------------------
        n_dec = max(1, int(math.log2(max(4, self.max_len))))   # pure decays
        n_osc = n_dec                                          # resonators
        self.n_modes = n_dec + n_osc
        half_lives = 4.0 * (2.0 ** torch.arange(n_dec, dtype=torch.float32))
        periods = 4.0 * (2.0 ** torch.arange(n_osc, dtype=torch.float32))
        osc_half_lives = 2.0 * periods
        hl = torch.cat([half_lives, osc_half_lives]).clamp(max=4.0 * self.max_len)
        rate = math.log(2.0) / hl                              # per-step decay rate
        # softplus parameterization keeps rates positive: r = softplus(nu)
        self.pole_nu = nn.Parameter(torch.log(torch.expm1(rate)))
        theta = torch.cat([torch.zeros(n_dec), 2.0 * math.pi / periods])
        self.pole_theta = nn.Parameter(theta)
        self.mix_a = nn.Parameter(torch.randn(dim, self.n_modes) * 0.02)
        self.mix_b = nn.Parameter(torch.randn(dim, self.n_modes) * 0.02)
        self.kernel_gain = nn.Parameter(torch.ones(dim))

        # --- content tape ---------------------------------------------------
        self.beta_proj = nn.Linear(dim, n_heads, bias=True)
        self.beta_proj._zero_init = True  # type: ignore[attr-defined]
        self.q_gain = nn.Parameter(torch.ones(max(1, self.rank)))
        self.k_gain = nn.Parameter(torch.ones(max(1, self.rank)))
        self.tape_gain = nn.Parameter(torch.ones(self.head_dim))
        self.tape_scale = nn.Parameter(torch.tensor(0.1))
        # fixed bounded decays: half of the heads at L/4, half at L/16
        hl_tape = torch.where(
            torch.arange(n_heads) % 2 == 0,
            torch.tensor(self.max_len / 4.0),
            torch.tensor(self.max_len / 16.0),
        )
        self.register_buffer("tape_log_decay", -math.log(2.0) / hl_tape, persistent=False)

        self.gate_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj._residual_proj = True  # type: ignore[attr-defined]
        self.dropout = nn.Dropout(dropout)
        self.reset_gate_biases()

    def reset_gate_biases(self) -> None:
        with torch.no_grad():
            nn.init.constant_(self.beta_proj.bias, -1.0)  # sparse tape writes

    def _rms(self, t: torch.Tensor, gain: torch.Tensor, dim: int) -> torch.Tensor:
        if t.size(dim) > 1:
            t = t * torch.rsqrt(t.pow(2).mean(dim=dim, keepdim=True) + self.eps)
        shape = [1] * t.dim()
        shape[dim] = -1
        return t * gain.view(shape).to(t.dtype)

    def _build_kernel(self, L: int, device: torch.device) -> torch.Tensor:
        """[D, L] causal kernels from the resonator bank; l1-normalized rows."""
        tau = torch.arange(L, device=device, dtype=torch.float32)
        rate = F.softplus(self.pole_nu.float())                       # [M]
        env = torch.exp(-tau.unsqueeze(1) * rate.unsqueeze(0))        # [L, M]
        ang = tau.unsqueeze(1) * self.pole_theta.float().unsqueeze(0) # [L, M]
        basis_c = env * torch.cos(ang)                                # [L, M]
        basis_s = env * torch.sin(ang)
        kernel = self.mix_a.float() @ basis_c.T + self.mix_b.float() @ basis_s.T  # [D, L]
        kernel = kernel / (kernel.abs().sum(dim=-1, keepdim=True) + self.eps)
        return kernel * self.kernel_gain.float().unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        if L > self.max_len:
            raise ValueError(f"sequence length {L} exceeds prism max_len {self.max_len}")
        H, r, Dh = self.n_heads, self.rank, self.head_dim
        qk_dim = H * r

        proj = self.in_proj(x)
        proj = self.conv(proj.transpose(1, 2))[..., :L]        # [B, C, L]
        proj = F.silu(proj)
        g = proj[:, :D]
        u = proj[:, D:2 * D]
        u_g = u * g                                            # [B, D, L]

        device_type = x.device.type if x.device.type in ("cuda", "cpu") else "cuda"

        # 1) resonator filterbank: one causal FFT convolution
        with torch.amp.autocast(device_type=device_type, enabled=False):
            n = 2 * L
            kernel = self._build_kernel(L, x.device)           # [D, L]
            U = torch.fft.rfft(u_g.float(), n=n)               # [B, D, n//2+1]
            Kf = torch.fft.rfft(kernel, n=n)                   # [D, n//2+1]
            y = torch.fft.irfft(U * Kf.unsqueeze(0), n=n)[..., :L]  # [B, D, L]

        # 2) bounded-decay content tape: one plain fp32 cumsum
        if r > 0:
            k = proj[:, 2 * D:2 * D + qk_dim].reshape(B, H, r, L)
            q = proj[:, 2 * D + qk_dim:].reshape(B, H, r, L)
            k = self._rms(k, self.k_gain, dim=2)
            q = self._rms(q, self.q_gain, dim=2)
            u_t = self._rms(u.reshape(B, H, Dh, L), torch.ones_like(self.tape_gain), dim=2)
            beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2).unsqueeze(2)  # [B,H,1,L]
            z = beta * (k.unsqueeze(3) * u_t.unsqueeze(2)).flatten(2, 3)          # [B,H,r*Dh,L]
            with torch.amp.autocast(device_type=device_type, enabled=False):
                t_idx = torch.arange(L, device=x.device, dtype=torch.float32)
                G = self.tape_log_decay.float().view(1, H, 1, 1) * t_idx.view(1, 1, 1, L)  # [1,H,1,L]
                T = torch.exp(G) * torch.cumsum(torch.exp(-G) * z.float(), dim=-1)
                o_tape = torch.einsum("bhrl,bhrdl->bhdl", q.float(), T.view(B, H, r, Dh, L))
            o_tape = self._rms(o_tape, self.tape_gain, dim=2)
            y = y + self.tape_scale.float() * o_tape.reshape(B, D, L)

        y = y.to(dtype=x.dtype).transpose(1, 2)                # [B, L, D]
        y = self.dropout(y * F.silu(self.gate_proj(x)))
        return self.o_proj(y)


class PrismBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float, dropout: float, eps: float, rank: int, conv_kernel: int, max_len: int):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=eps)
        self.mixer = PrismResonatorMixer(dim, n_heads, rank, dropout, eps, conv_kernel, max_len)
        self.norm2 = RMSNorm(dim, eps=eps)
        self.mlp = SwiGLUMLP(dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def make_mixer_block(kind: str, dim: int, n_heads: int, mlp_ratio: float, cfg: LosslessSubbandByteLadderConfig) -> nn.Module:
    if kind == "attention":
        return TransformerBlock(dim, n_heads, mlp_ratio, cfg.dropout, cfg.eps, cfg.rope_theta)
    if kind == "fastweight":
        return FastWeightBlock(dim, n_heads, mlp_ratio, cfg.dropout, cfg.eps, cfg.chunk_size, cfg.conv_kernel, cfg.fastweight_gate)
    if kind == "prism":
        return PrismBlock(dim, n_heads, mlp_ratio, cfg.dropout, cfg.eps, cfg.prism_rank, cfg.conv_kernel, cfg.block_size)
    raise ValueError(f"unknown mixer kind: {kind}")


class LowRankMLP(nn.Module):
    """Pointwise low-rank function used inside lifting steps.

    With zero-initialized output projection, lifting starts as an exact lazy
    wavelet: coarse=even, detail=odd.  During training P/U can change, while
    invertibility is preserved algebraically by the inverse lifting equations.
    """

    def __init__(self, dim: int, rank: int, eps: float, zero_init: bool = True):
        super().__init__()
        rank = max(1, int(rank))
        self.norm = RMSNorm(dim, eps=eps)
        self.fc1 = nn.Linear(dim, rank, bias=False)
        self.fc2 = nn.Linear(rank, dim, bias=False)
        if zero_init:
            self.fc2._zero_init = True  # type: ignore[attr-defined]
            nn.init.zeros_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(self.norm(x))))


class LowRankProject(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, eps: float):
        super().__init__()
        rank = max(1, int(rank))
        self.norm = RMSNorm(in_dim, eps=eps)
        self.fc1 = nn.Linear(in_dim, rank, bias=False)
        self.fc2 = nn.Linear(rank, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(self.norm(x))))


class FixedByteBitCarrier(nn.Module):
    """Fixed exact 8-bit carrier for bytes.

    Each byte is represented by its binary bits as -1/+1 floats. This is a
    parameter-free injective map from {0..255} to R^8, so the carrier contains
    exact byte identity before the learned invertible lifting transform.
    """

    def __init__(self):
        super().__init__()
        table = torch.empty(256, 8, dtype=torch.float32)
        for value in range(256):
            for bit in range(8):
                table[value, bit] = 1.0 if ((value >> bit) & 1) else -1.0
        self.register_buffer("table", table, persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.table[input_ids]


class LosslessSubbandAnalysis(nn.Module):
    """One causal dyadic analysis stage.

    Carrier path is exactly invertible:
        even   = K[:, 0::2]
        odd    = K[:, 1::2]
        detail = odd - P(even)
        coarse = even + U(detail)
        K_next = concat(coarse, detail)

    Inverse:
        coarse, detail = split(K_next)
        even = coarse - U(detail)
        odd  = detail + P(even)

    The semantic path is predictive/learned. It receives the subband carrier and
    then runs one or more causal Transformer blocks at the shortened length.
    """

    def __init__(
        self,
        sem_in: int,
        sem_out: int,
        carrier_in: int,
        n_heads: int,
        depth: int,
        mlp_ratio: float,
        cfg: LosslessSubbandByteLadderConfig,
        mixer_kind: str = "attention",
    ):
        super().__init__()
        self.sem_in = sem_in
        self.sem_out = sem_out
        self.carrier_in = carrier_in
        self.carrier_out = 2 * carrier_in
        self.disable_carrier_to_sem = cfg.disable_carrier_to_sem

        self.predict_odd = LowRankMLP(carrier_in, cfg.lift_rank, cfg.eps, zero_init=True)
        self.update_even = LowRankMLP(carrier_in, cfg.lift_rank, cfg.eps, zero_init=True)

        self.sem_pair = nn.Linear(2 * sem_in, sem_out, bias=False)
        self.carrier_to_sem = LowRankProject(self.carrier_out, sem_out, cfg.carrier_proj_rank, cfg.eps)
        self.carrier_scale = nn.Parameter(torch.tensor(float(cfg.carrier_scale_init)))
        self.pre_norm = RMSNorm(sem_out, eps=cfg.eps)
        self.blocks = nn.ModuleList(
            [
                make_mixer_block(mixer_kind, sem_out, n_heads, mlp_ratio, cfg)
                for _ in range(depth)
            ]
        )

    def carrier_lift(self, carrier: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        even = carrier[:, 0::2, :]
        odd = carrier[:, 1::2, :]
        detail = odd - self.predict_odd(even)
        coarse = even + self.update_even(detail)
        return torch.cat([coarse, detail], dim=-1), detail

    def carrier_unlift(self, parent: torch.Tensor) -> torch.Tensor:
        coarse, detail = parent.chunk(2, dim=-1)
        even = coarse - self.update_even(detail)
        odd = detail + self.predict_odd(even)
        bsz, parent_len, dim = even.shape
        child = torch.empty(bsz, parent_len * 2, dim, dtype=parent.dtype, device=parent.device)
        child[:, 0::2, :] = even
        child[:, 1::2, :] = odd
        return child

    def forward(self, sem: torch.Tensor, carrier: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sem.size(1) % 2 != 0 or carrier.size(1) % 2 != 0:
            raise ValueError("analysis requires even sequence length")
        sem_pair = torch.cat([sem[:, 0::2, :], sem[:, 1::2, :]], dim=-1)
        carrier_next, detail = self.carrier_lift(carrier)
        abstract = self.sem_pair(sem_pair)
        if not self.disable_carrier_to_sem:
            abstract = abstract + self.carrier_scale * self.carrier_to_sem(carrier_next)
        abstract = self.pre_norm(abstract)
        for block in self.blocks:
            abstract = block(abstract)
        return abstract, carrier_next, detail


class CausalShiftedSynthesisFuse(nn.Module):
    """Causal super-resolution fusion from parent semantic token to children.

    A parent token j has seen the complete child span [2j, 2j+1]. Its right-child
    proposal is safe at child 2j+1. Its left-child proposal would leak to 2j, so
    the left child receives parent j-1. This is the delayed-synthesis rule.
    """

    def __init__(
        self,
        sem_child: int,
        sem_parent: int,
        carrier_child: int,
        refine_depth: int,
        n_heads: int,
        mlp_ratio: float,
        cfg: LosslessSubbandByteLadderConfig,
    ):
        super().__init__()
        self.sem_child = sem_child
        self.unpack = nn.Linear(sem_parent, 2 * sem_child, bias=False)
        self.carrier_to_sem = LowRankProject(carrier_child, sem_child, cfg.carrier_proj_rank, cfg.eps)
        self.carrier_scale = nn.Parameter(torch.tensor(float(cfg.carrier_scale_init)))
        self.gate = nn.Linear(3 * sem_child, sem_child, bias=False)
        self.mix = nn.Linear(3 * sem_child, sem_child, bias=False)
        self.norm = RMSNorm(sem_child, eps=cfg.eps)
        self.refine = nn.ModuleList(
            [
                make_mixer_block(cfg.up_mixer, sem_child, n_heads, mlp_ratio, cfg)
                for _ in range(refine_depth)
            ]
        )

    def shifted_unpack(self, parent: torch.Tensor) -> torch.Tensor:
        bsz, parent_len, _ = parent.shape
        proposal = self.unpack(parent).view(bsz, parent_len, 2, self.sem_child)
        left_prop = proposal[:, :, 0, :]
        right_prop = proposal[:, :, 1, :]
        safe_left = torch.zeros_like(left_prop)
        if parent_len > 1:
            safe_left[:, 1:, :] = left_prop[:, :-1, :]
        child = torch.empty(bsz, parent_len * 2, self.sem_child, device=parent.device, dtype=parent.dtype)
        child[:, 0::2, :] = safe_left
        child[:, 1::2, :] = right_prop
        return child

    def forward(self, skip_sem: torch.Tensor, parent_sem: torch.Tensor, carrier_child: torch.Tensor) -> torch.Tensor:
        up = self.shifted_unpack(parent_sem)
        carrier_feat = self.carrier_scale * self.carrier_to_sem(carrier_child)
        z = torch.cat([skip_sem, up, carrier_feat], dim=-1)
        delta = self.mix(z)
        gate = torch.sigmoid(self.gate(z))
        y = self.norm(skip_sem + gate * delta)
        for block in self.refine:
            y = block(y)
        return y


class AllScaleCausalFusion(nn.Module):
    """Inject causally available coarse/detail subbands into byte resolution.

    Feature token j at level ell summarizes a span of size 2**ell ending at
    position (j+1)*span-1.  We expand it to byte resolution and delay it by
    span-1 positions.  Thus every contribution at byte position t is derived
    only from spans whose end <= t.
    """

    def __init__(self, cfg: LosslessSubbandByteLadderConfig):
        super().__init__()
        self.cfg = cfg
        self.levels = cfg.resolved_multiscale_levels
        sem_widths = cfg.sem_widths
        carrier_widths = cfg.carrier_widths
        self.sem_projs = nn.ModuleDict()
        self.detail_projs = nn.ModuleDict()
        self.sem_scales = nn.ParameterDict()
        self.detail_scales = nn.ParameterDict()
        for level in self.levels:
            self.sem_projs[str(level)] = nn.Linear(sem_widths[level], cfg.d_model, bias=False)
            self.sem_scales[str(level)] = nn.Parameter(torch.tensor(float(cfg.multiscale_init)))
            # detail for transition level-1 -> level has width carrier_widths[level-1]
            self.detail_projs[str(level)] = LowRankProject(carrier_widths[level - 1], cfg.d_model, cfg.carrier_proj_rank, cfg.eps)
            self.detail_scales[str(level)] = nn.Parameter(torch.tensor(float(cfg.multiscale_init)))
        self.norm = RMSNorm(cfg.d_model, eps=cfg.eps)

    @staticmethod
    def causal_delay_expand(feat: torch.Tensor, span: int, block_size: int) -> torch.Tensor:
        bsz, _, dim = feat.shape
        expanded = feat.repeat_interleave(span, dim=1)
        out = torch.zeros(bsz, block_size, dim, dtype=feat.dtype, device=feat.device)
        delay = span - 1
        if delay >= block_size:
            return out
        n = min(block_size - delay, expanded.size(1))
        out[:, delay : delay + n, :] = expanded[:, :n, :]
        return out

    def forward(self, sem_skips: List[torch.Tensor], details: List[torch.Tensor], block_size: int) -> torch.Tensor:
        if not self.levels:
            return torch.zeros(sem_skips[0].size(0), block_size, self.cfg.d_model, device=sem_skips[0].device, dtype=sem_skips[0].dtype)
        out = torch.zeros(sem_skips[0].size(0), block_size, self.cfg.d_model, device=sem_skips[0].device, dtype=sem_skips[0].dtype)
        for level in self.levels:
            span = 2 ** level
            sem = self.sem_projs[str(level)](sem_skips[level])
            det = self.detail_projs[str(level)](details[level - 1])
            out = out + self.sem_scales[str(level)] * self.causal_delay_expand(sem, span, block_size)
            out = out + self.detail_scales[str(level)] * self.causal_delay_expand(det, span, block_size)
        return self.norm(out)


class LosslessSubbandByteLadderLM(nn.Module):
    def __init__(self, cfg: LosslessSubbandByteLadderConfig):
        super().__init__()
        self.cfg = cfg
        self.lengths = cfg.lengths
        self.sem_widths = cfg.sem_widths
        self.carrier_widths = cfg.carrier_widths
        self.down_depths = cfg.resolved_down_depths
        self.up_refine_depths = cfg.resolved_up_refine_depths
        self.ladder_mlp_ratios = cfg.resolved_ladder_mlp_ratios
        self.ladder_heads = cfg.resolved_ladder_heads

        self.byte_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.carrier = FixedByteBitCarrier()
        self.drop = nn.Dropout(cfg.dropout)

        self.initial = nn.ModuleList(
            [
                make_mixer_block(cfg.initial_mixer, cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg)
                for _ in range(cfg.n_initial_layers)
            ]
        )

        downs: List[nn.Module] = []
        for i in range(len(self.lengths) - 1):
            sem_in = self.sem_widths[i]
            sem_out = self.sem_widths[i + 1]
            carrier_in = self.carrier_widths[i]
            heads = _best_heads(sem_out, self.ladder_heads[i])
            downs.append(LosslessSubbandAnalysis(sem_in, sem_out, carrier_in, heads, self.down_depths[i], self.ladder_mlp_ratios[i], cfg, mixer_kind=cfg.down_mixer))
        self.downs = nn.ModuleList(downs)

        # Dedicated mixer-only shared-state reasoner at the final encoder
        # bottleneck.  It is a new parameter set, shared across R iterations;
        # the original DeepMem analysis stack is otherwise untouched.
        self.bottleneck_reasoner: Optional[SharedMixerStateReasoner] = None
        if cfg.bottleneck_shared_state:
            bottom_dim = self.sem_widths[-1]
            bottom_heads = _best_heads(bottom_dim, self.ladder_heads[-1])
            self.bottleneck_reasoner = SharedMixerStateReasoner(
                dim=bottom_dim,
                n_heads=bottom_heads,
                steps=cfg.shared_state_steps,
                dropout=cfg.dropout,
                eps=cfg.eps,
                chunk_size=cfg.chunk_size,
                conv_kernel=cfg.conv_kernel,
                gate_mode=cfg.fastweight_gate,
            )

        ups: List[nn.Module] = []
        # up_refine_depths is indexed by child level: 0 means up into level 0.
        for i in reversed(range(len(self.lengths) - 1)):
            sem_child = self.sem_widths[i]
            sem_parent = self.sem_widths[i + 1]
            carrier_child = self.carrier_widths[i]
            heads = _best_heads(sem_child, self.ladder_heads[i])
            ups.append(CausalShiftedSynthesisFuse(sem_child, sem_parent, carrier_child, self.up_refine_depths[i], heads, self.ladder_mlp_ratios[i], cfg))
        self.ups = nn.ModuleList(ups)

        self.multiscale = AllScaleCausalFusion(cfg)

        self.final = nn.ModuleList(
            [
                make_mixer_block(cfg.final_mixer, cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg)
                for _ in range(cfg.n_final_layers)
            ]
        )
        self.norm_f = RMSNorm(cfg.d_model, eps=cfg.eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Keep the reference DeepMem residual-init scale unchanged.  The shared
        # reasoner is an added ablation module; changing this count would also
        # change initialization of every pre-existing DeepMem residual projection.
        self.residual_layers = cfg.n_initial_layers + sum(self.down_depths) + sum(self.up_refine_depths) + cfg.n_final_layers
        self.residual_init_scale = 1.0 / math.sqrt(max(1.0, 2.0 * float(self.residual_layers)))
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
            elif getattr(module, "_residual_proj", False):
                nn.init.normal_(module.weight, mean=0.0, std=0.02 * self.residual_init_scale)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, GatedLinearRecurrenceMixer):
            nn.init.constant_(module.decay_proj.bias, -4.6 if module.gate_mode == "scalar" else -2.2)
        if isinstance(module, PrismResonatorMixer):
            module.reset_gate_biases()

    def encode_pyramid(self, input_ids: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        bsz, seq_len = input_ids.shape
        if seq_len != self.cfg.block_size:
            raise ValueError(f"LosslessSubbandByteLadderLM expects seq_len == block_size={self.cfg.block_size}; got {seq_len}")
        sem = self.drop(self.byte_emb(input_ids))
        carrier = self.carrier(input_ids).to(dtype=sem.dtype)

        for block in self.initial:
            if self.cfg.grad_checkpointing and self.training:
                sem = torch.utils.checkpoint.checkpoint(block, sem, use_reentrant=False)
            else:
                sem = block(sem)

        sem_skips: List[torch.Tensor] = [sem]
        carrier_skips: List[torch.Tensor] = [carrier]
        details: List[torch.Tensor] = []
        for down_idx, down in enumerate(self.downs):
            if self.cfg.grad_checkpointing and self.training:
                sem, carrier, detail = torch.utils.checkpoint.checkpoint(down, sem, carrier, use_reentrant=False)
            else:
                sem, carrier, detail = down(sem, carrier)

            # Apply the shared-state reasoner only after the final analysis
            # stage (the 16-token bottleneck in the reference configuration).
            if down_idx == len(self.downs) - 1 and self.bottleneck_reasoner is not None:
                if self.cfg.grad_checkpointing and self.training:
                    sem = torch.utils.checkpoint.checkpoint(self.bottleneck_reasoner, sem, use_reentrant=False)
                else:
                    sem = self.bottleneck_reasoner(sem)

            sem_skips.append(sem)
            carrier_skips.append(carrier)
            details.append(detail)
        return sem_skips, carrier_skips, details

    def decode_from_pyramid(self, sem_skips: List[torch.Tensor], carrier_skips: List[torch.Tensor], details: List[torch.Tensor]) -> torch.Tensor:
        y = sem_skips[-1]
        for up_idx, up in enumerate(self.ups):
            child_level = len(self.lengths) - 2 - up_idx
            skip_sem = sem_skips[child_level]
            carrier_child = carrier_skips[child_level]
            if self.cfg.grad_checkpointing and self.training:
                y = torch.utils.checkpoint.checkpoint(up, skip_sem, y, carrier_child, use_reentrant=False)
            else:
                y = up(skip_sem, y, carrier_child)
        if not self.cfg.disable_multiscale:
            y = y + self.multiscale(sem_skips, details, self.cfg.block_size)
        return y

    def reconstruct_carrier_from_bottom(self, bottom: torch.Tensor) -> torch.Tensor:
        carrier = bottom
        for down in reversed(self.downs):
            carrier = down.carrier_unlift(carrier)  # type: ignore[attr-defined]
        return carrier

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        sem_skips, carrier_skips, details = self.encode_pyramid(input_ids)
        y = self.decode_from_pyramid(sem_skips, carrier_skips, details)
        for block in self.final:
            if self.cfg.grad_checkpointing and self.training:
                y = torch.utils.checkpoint.checkpoint(block, y, use_reentrant=False)
            else:
                y = block(y)
        y = self.norm_f(y)
        logits = self.lm_head(y)
        out: Dict[str, torch.Tensor] = {"logits": logits}
        if targets is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.cfg.vocab_size),
                targets[:, 1:].contiguous().view(-1),
                reduction="mean",
            )
            out["loss"] = loss
        return out

    def parameter_breakdown(self) -> Dict[str, int]:
        return {
            "byte_embedding": count_parameters(self.byte_emb),
            "initial_blocks": count_parameters(self.initial),
            "down_ladder": count_parameters(self.downs),
            "bottleneck_shared_state_reasoner": (
                count_parameters(self.bottleneck_reasoner) if self.bottleneck_reasoner is not None else 0
            ),
            "up_ladder": count_parameters(self.ups),
            "multiscale_fusion": count_parameters(self.multiscale),
            "final_blocks": count_parameters(self.final),
            "final_norm": count_parameters(self.norm_f),
            "lm_head": count_parameters(self.lm_head),
        }

