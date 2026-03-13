"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import time
from dataclasses import dataclass, asdict
from typing import Sequence, Tuple

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

def verify_macos_env():
    if sys.platform != "darwin":
        raise RuntimeError(f"This script requires macOS with Metal. Detected platform: {sys.platform}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS (Metal Performance Shaders) is not available. Ensure you are running on Apple Silicon with a compatible PyTorch build.")
    print("Environment verified: macOS detected with Metal (MPS) hardware acceleration available.")
    print()

verify_macos_env()

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------


@dataclass
class HRMConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768

    n_embd: int = 768
    n_head: int = 6
    n_kv_head: int = 6
    n_layer: int = 12
    window_pattern: str = "SSSL"

    hidden_size: int | None = None
    intermediate_size: int | None = None
    batch_size: int = 256
    head_dim: int = 64
    is_causal: bool = True

    H_cycles: int = 2
    L_cycles: int = 2

    cycle_per_data: int = 16

    norm_eps: float = 1e-6
    rope_base: float = 10000.0
    forward_dtype: str = "float32"  # change to float32 if your hardware doesn't support bfloat16

    seed: int = 7

    def __post_init__(self):
        if self.hidden_size is None:
            self.hidden_size = self.n_embd
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

    @property
    def num_layers(self) -> int:
        return self.n_layer

    @property
    def seq_len(self) -> int:
        return self.sequence_len


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------

CosSin = Tuple[torch.Tensor, torch.Tensor]


def trunc_normal_init_(x: torch.Tensor, std: float):
    return nn.init.trunc_normal_(x, std=std).mul_(1.1368472343385565)  # Scale by a constant


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos_sin: CosSin):
    # q, k: [..., seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    cos, sin = cos_sin
    return ((x * cos.unsqueeze(-2)) + (rotate_half(x) * sin.unsqueeze(-2))).to(x.dtype)


class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, batch_output_dims: Sequence[int] = (),
                 **kwargs):
        super().__init__()
        self.in_features = in_features

        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((*batch_output_dims, out_features, in_features), **kwargs),
                               std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,), **kwargs))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.view(-1, self.in_features).to(input.dtype),
                        self.bias.to(input.dtype) if self.bias is not None else None)


class CastedScaledEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Scale to the same std as most parameters
        self.scale = embedding_dim ** 0.5
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=1.0 / self.scale)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.scale * self.weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, **kwargs):
        super().__init__()
        self.gate_up_proj = CastedLinear(hidden_size, intermediate_size, bias=False, batch_output_dims=(2,), **kwargs)
        self.down_proj = CastedLinear(intermediate_size, hidden_size, bias=False, **kwargs)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, is_causal, **kwargs):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.is_causal = is_causal

        self.qkv_proj = CastedLinear(hidden_size, self.num_heads * self.head_dim, bias=False, batch_output_dims=(3,),
                                     **kwargs)
        self.o_proj = CastedLinear(head_dim * num_heads, hidden_size, bias=False, **kwargs)
        with torch.no_grad():
            self.o_proj.weight.zero_()

    def forward(self, hidden_states: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        # hidden_states, qkv: [..., seq_len, hidden_size]
        qkv = self.qkv_proj(hidden_states)

        # Split head (last dimension of projected qkv)
        qkv = qkv.view(*qkv.shape[:-1], self.num_heads, -1)
        query, key, value = qkv.chunk(3, dim=-1)
        # Rotary embedding
        query = apply_rotary_pos_emb(query, cos_sin)
        key = apply_rotary_pos_emb(key, cos_sin)
        # PyTorch SDPA attention
        attn_output = F.scaled_dot_product_attention(query.transpose(-2, -3), key.transpose(-2, -3),
                                                     value.transpose(-2, -3), is_causal=self.is_causal).transpose(-2,
                                                                                                                  -3)
        # attn_output: [..., seq_len, num_heads, head_dim]
        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1)
        return self.o_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, config: HRMConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size if config.hidden_size is not None else config.n_embd
        intermediate_size = config.intermediate_size if config.intermediate_size is not None else 4 * hidden_size
        self.attn = Attention(
            hidden_size=hidden_size,
            head_dim=config.head_dim,
            num_heads=hidden_size // config.head_dim,
            is_causal=config.is_causal
        )
        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size
        )
        self.norm = lambda x: F.rms_norm(x, (x.shape[-1],), eps=config.norm_eps)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # Post Norm
        x = self.norm(x + self.attn(x, **kwargs))
        return self.norm(x + self.mlp(x))


class HRMRecurrentBlock(nn.Module):
    def __init__(self, config: HRMConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(config) for _layer_idx in range(config.num_layers)])

    def forward(self, x: torch.Tensor, n: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x + n
        for layer in self.layers:
            h = layer(h, **kwargs)
        return h


# HRMCarry is a tuple containing two latent states(z_H, z_L)
HRMCarry = Tuple[torch.Tensor, torch.Tensor]
datatype = {
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
}


class HRM(nn.Module):
    def __init__(self, config: HRMConfig) -> None:
        super().__init__()
        self.H_cycles = config.H_cycles
        self.L_cycles = config.L_cycles

        self.hidden_size = config.hidden_size if config.hidden_size is not None else config.n_embd
        self.max_seq_len = config.sequence_len
        self.dtype = datatype[config.forward_dtype]

        self.batch_size = config.batch_size

        # Backbone Layers
        self.H_level = HRMRecurrentBlock(config)
        self.L_level = HRMRecurrentBlock(config)

        # RoPE
        self.rope = RotaryEmbedding(config.head_dim, config.sequence_len, config.rope_base)
        # I/O Layers
        self.embed = CastedScaledEmbedding(config.vocab_size, self.hidden_size, cast_to=self.dtype)
        self.lm_head = CastedLinear(self.hidden_size, config.vocab_size, bias=False)

        # [修复 1]: 移除了 buffer 注册，因为对于独立样本(Shuffle后)不需要跨Batch保存状态。
        # 我们将在 forward 内部动态生成初始状态。

    # [修复 2]: 这些 carry 管理函数对于独立样本任务不再需要，可以保留空实现或删除
    def keep_carry(self, ):
        pass

    def restore_carry(self, ):
        pass

    def init_carry(self, ):
        pass

    def init_weights(self):
        return

    def estimate_flops(self):
        return 0.0

    def num_scaling_params(self):
        return {"total": sum(p.numel() for p in self.parameters())}

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        adam_betas=(0.8, 0.95),
        scalar_lr=0.5,
    ):
        del scalar_lr
        embedding_params = list(self.embed.parameters())
        lm_head_params = list(self.lm_head.parameters())
        embedding_param_ids = {id(p) for p in embedding_params}
        lm_head_param_ids = {id(p) for p in lm_head_params}
        other_params = [
            p for p in self.parameters()
            if id(p) not in (embedding_param_ids | lm_head_param_ids)
        ]

        muon_params = [p for p in other_params if p.ndim == 2]
        other_adamw_params = [p for p in other_params if p.ndim != 2]

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
        if other_adamw_params:
            param_groups.append(
                dict(kind='adamw', params=other_adamw_params, lr=matrix_lr, betas=adam_betas, eps=1e-10, weight_decay=weight_decay),
            )
        for shape in sorted({p.shape for p in muon_params}):
            group_params = [p for p in muon_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, reduction: str = 'mean'):
        x = self.embed(idx)
        B, T = idx.shape
        cos, sin = self.rope()
        seq_info = dict(cos_sin=(cos[:T], sin[:T]))

        z_H = torch.zeros(B, T, self.hidden_size, device=x.device, dtype=x.dtype)
        z_L = torch.zeros(B, T, self.hidden_size, device=x.device, dtype=x.dtype)

        with torch.no_grad():
            for _i in range(self.H_cycles * self.L_cycles - 1):
                z_L = self.L_level(z_L, z_H + x, **seq_info)
                if (_i + 1) % self.L_cycles == 0:
                    z_H = self.H_level(z_H, z_L, **seq_info)

        z_L = self.L_level(z_L, z_H + x, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        logits = self.lm_head(z_H)
        logits = logits.float()
        softcap = 15
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction,
            )
            return loss
        return logits


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # PyTorch SDPA without FlashAttention 3
        # Expand heads for KV based on GQA
        k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
        v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
        
        # Transpose to [B, H, T, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply mask for sliding window
        window = window_size[0]
        if window > 0 and window < T:
            # Mask out tokens outside the window
            mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
            mask = mask.triu(diagonal=1 - window)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    # Move scalars to correct device and dtype
    step_t = step_t.to(device=p.device, dtype=p.dtype)
    lr_t = lr_t.to(device=p.device, dtype=p.dtype)
    beta1_t = beta1_t.to(device=p.device, dtype=p.dtype)
    beta2_t = beta2_t.to(device=p.device, dtype=p.dtype)
    eps_t = eps_t.to(device=p.device, dtype=p.dtype)
    wd_t = wd_t.to(device=p.device, dtype=p.dtype)
    
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Move scalars to correct device and dtype
    momentum_t = momentum_t.to(device=stacked_params.device, dtype=stacked_params.dtype)
    lr_t = lr_t.to(device=stacked_params.device, dtype=stacked_params.dtype)
    wd_t = wd_t.to(device=stacked_params.device, dtype=stacked_params.dtype)
    beta2_t = beta2_t.to(device=stacked_params.device, dtype=stacked_params.dtype)

    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    
    # Needs to match second_momentum_buffer.dtype for lerp_
    beta2_cast = beta2_t.to(second_momentum_buffer.dtype)
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2_cast)
    
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        
        # Compile conditionally
        compiler_kwargs = {"dynamic": False, "fullgraph": True}
        if device_type in ("cuda", "cpu"):
            self.adamw_step_fused = torch.compile(adamw_step_fused, **compiler_kwargs)
            self.muon_step_fused = torch.compile(muon_step_fused, **compiler_kwargs)
        else:
            self.adamw_step_fused = adamw_step_fused
            self.muon_step_fused = muon_step_fused

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            self.adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params_with_grad = [p for p in group['params'] if p.grad is not None]
        if not params_with_grad:
            return
        p = params_with_grad[0]
        state = self.state[p]
        num_params = len(params_with_grad)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state or state["momentum_buffer"].shape[0] != num_params:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state or state["second_momentum_buffer"].shape[0] != num_params:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params_with_grad])
        stacked_params = torch.stack(params_with_grad)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        self.muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params_with_grad, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "L"    # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**16 # ~65K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 4               # number of transformer layers
DEVICE_BATCH_SIZE = 8   # per-device batch size (reduce if OOM)
MODEL_IMPL = os.environ.get("MODEL_IMPL", "hrm").lower()  # "hrm" | "gpt"
TRAIN_TIME_BUDGET = float(os.environ.get("TRAIN_TIME_BUDGET", TIME_BUDGET))
MAX_TRAIN_STEPS = int(os.environ.get("MAX_TRAIN_STEPS", "0"))  # 0 means disabled

if MODEL_IMPL not in {"hrm", "gpt"}:
    raise ValueError(f"Unsupported MODEL_IMPL={MODEL_IMPL!r}. Expected 'hrm' or 'gpt'.")
if TRAIN_TIME_BUDGET <= 0:
    raise ValueError(f"TRAIN_TIME_BUDGET must be > 0, got {TRAIN_TIME_BUDGET}.")

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")

# Detect device
device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_type)

# Autocast context
if device_type == "cuda":
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
elif device_type == "cpu":
    autocast_ctx = torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
else:
    import contextlib
    autocast_ctx = contextlib.nullcontext()

H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return HRMConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )


def build_model(config: HRMConfig) -> nn.Module:
    if MODEL_IMPL == "gpt":
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device=device)
        model.init_weights()
        return model

    model = HRM(config).to(device=device)
    model.init_weights()
    return model

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

model = build_model(config)

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

# torch.compile is unstable on MPS, only use on CUDA
if device_type == "cuda":
    model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Model impl: {MODEL_IMPL}")
print(f"Time budget: {TRAIN_TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0.0
total_training_time = 0.0
step = 0

def sync_device(device_type):
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()

while True:
    sync_device(device_type)
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_training_time / TRAIN_TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding
    if train_loss_f > 100:
        raise RuntimeError(f"Loss exploded at step={step}: loss={train_loss_f:.6f}")

    sync_device(device_type)
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, TRAIN_TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    if MAX_TRAIN_STEPS > 0 and step >= MAX_TRAIN_STEPS:
        break

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TRAIN_TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
if device_type == "cuda":
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
else:
    peak_vram_mb = 0.0

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
