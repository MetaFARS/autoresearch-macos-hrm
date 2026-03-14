#!/usr/bin/env python3
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
import json
import subprocess
from pathlib import Path

import fcntl
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests


def _import_prepare():
    orig_platform = sys.platform
    orig_mps_is_available = torch.backends.mps.is_available
    try:
        sys.platform = "darwin"
        torch.backends.mps.is_available = lambda: True
        import prepare
        return prepare
    finally:
        sys.platform = orig_platform
        torch.backends.mps.is_available = orig_mps_is_available


prepare = _import_prepare()
MAX_SEQ_LEN = prepare.MAX_SEQ_LEN
TIME_BUDGET = prepare.TIME_BUDGET
Tokenizer = prepare.Tokenizer
make_dataloader = prepare.make_dataloader
evaluate_bpb = prepare.evaluate_bpb

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
    forward_dtype: str = "bfloat16"  # change to float32 if your hardware doesn't support bfloat16

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
        if os.environ.get("ZERO_O_PROJ_INIT", "1").strip() != "0":
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
        self.hidden_size = hidden_size
        self.prenorm = (os.environ.get("PRENORM", "0") or "0").strip() == "1"
        self.layer_scale_init = float(os.environ.get("LAYER_SCALE_INIT", "0") or "0")
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
        if self.layer_scale_init > 0:
            self.ls_attn = nn.Parameter(torch.full((self.hidden_size,), self.layer_scale_init))
            self.ls_mlp = nn.Parameter(torch.full((self.hidden_size,), self.layer_scale_init))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # Post Norm
        if self.prenorm:
            h = self.norm(x)
            attn_out = self.attn(h, **kwargs)
            if self.layer_scale_init > 0:
                attn_out = attn_out * self.ls_attn
            x = x + attn_out
            h = self.norm(x)
            mlp_out = self.mlp(h)
            if self.layer_scale_init > 0:
                mlp_out = mlp_out * self.ls_mlp
            return x + mlp_out
        attn_out = self.attn(x, **kwargs)
        if self.layer_scale_init > 0:
            attn_out = attn_out * self.ls_attn
        x = self.norm(x + attn_out)
        mlp_out = self.mlp(x)
        if self.layer_scale_init > 0:
            mlp_out = mlp_out * self.ls_mlp
        return self.norm(x + mlp_out)


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

        z_L = self.L_level(torch.zeros_like(x), x, **seq_info)
        z_H = self.H_level(torch.zeros_like(x), z_L, **seq_info)

        logits = self.lm_head(z_H)
        logits = logits.float()
        softcap = float(os.environ.get("LOGITS_SOFTCAP", "15") or "15")
        if softcap > 0:
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

def _env_str(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


def _env_int(name: str, default: int) -> int:
    v = _env_str(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = _env_str(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_betas(name: str, default: tuple[float, float]) -> tuple[float, float]:
    v = _env_str(name)
    if v is None:
        return default
    parts = [p.strip() for p in v.split(",") if p.strip()]
    if len(parts) != 2:
        return default
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return default


# Model architecture
ASPECT_RATIO = _env_int("ASPECT_RATIO", 64)       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = _env_int("HEAD_DIM", 32)               # target head dimension for attention
H_CYCLES = _env_int("H_CYCLES", 1)
L_CYCLES = _env_int("L_CYCLES", 1)
FORWARD_DTYPE = _env_str("FORWARD_DTYPE", "bfloat16") or "bfloat16"

# Optimization
TOTAL_BATCH_SIZE = _env_int("TOTAL_BATCH_SIZE", 2**16)  # tokens per optimizer step (effective)
EMBEDDING_LR = _env_float("EMBEDDING_LR", 0.6)          # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = _env_float("UNEMBEDDING_LR", 0.004)    # learning rate for lm_head (Adam)
MATRIX_LR = _env_float("MATRIX_LR", 0.04)               # learning rate for matrix parameters (Muon)
SCALAR_LR = _env_float("SCALAR_LR", 0.5)                # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = _env_float("WEIGHT_DECAY", 0.2)          # cautious weight decay for Muon
ADAM_BETAS = _env_betas("ADAM_BETAS", (0.8, 0.95))      # Adam beta1, beta2
WARMUP_RATIO = _env_float("WARMUP_RATIO", 0.0)          # fraction of time budget for LR warmup
WARMDOWN_RATIO = _env_float("WARMDOWN_RATIO", 0.5)      # fraction of time budget for LR warmdown
FINAL_LR_FRAC = _env_float("FINAL_LR_FRAC", 0.0)        # final LR as fraction of initial

# Model size
DEPTH = _env_int("DEPTH", 4)                    # number of transformer layers
DEVICE_BATCH_SIZE = _env_int("DEVICE_BATCH_SIZE", 4)  # per-device batch size (reduce if OOM)
TRAIN_TIME_BUDGET = float(os.environ.get("TRAIN_TIME_BUDGET", TIME_BUDGET))
MAX_TRAIN_STEPS = int(os.environ.get("MAX_TRAIN_STEPS", "0"))  # 0 means disabled

if TRAIN_TIME_BUDGET <= 0:
    raise ValueError(f"TRAIN_TIME_BUDGET must be > 0, got {TRAIN_TIME_BUDGET}.")

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
seed = _env_int("SEED", 42)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
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
        head_dim=HEAD_DIM,
        H_cycles=H_CYCLES,
        L_cycles=L_CYCLES,
        forward_dtype=FORWARD_DTYPE,
    )


def build_model(config: HRMConfig) -> nn.Module:
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
grad_accum_steps = max(1, (TOTAL_BATCH_SIZE + tokens_per_fwdbwd - 1) // tokens_per_fwdbwd)
TOTAL_BATCH_SIZE = grad_accum_steps * tokens_per_fwdbwd

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

print("Model impl: hrm")
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


def _git_short_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def _parse_best_val_bpb(lines: list[str]) -> float | None:
    best = None
    for line in lines[1:]:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        try:
            v = float(parts[1])
        except ValueError:
            continue
        if v <= 0:
            continue
        if best is None or v < best:
            best = v
    return best


def _elite_distance(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    keys = [
        ("depth", 2.0),
        ("head_dim", 1.0),
        ("aspect_ratio", 0.1),
        ("h_cycles", 1.0),
        ("l_cycles", 1.0),
        ("logits_softcap", 0.1),
        ("zero_o_proj_init", 1.0),
        ("prenorm", 1.0),
        ("layer_scale_init", 5.0),
    ]
    d = 0.0
    for k, w in keys:
        if k not in a or k not in b:
            continue
        va = a[k]
        vb = b[k]
        if isinstance(va, str) or isinstance(vb, str):
            d += w * (0.0 if va == vb else 1.0)
        else:
            try:
                d += w * abs(float(va) - float(vb))
            except Exception:
                pass
    if a.get("forward_dtype") != b.get("forward_dtype"):
        d += 1.0
    return d


def _append_result(val_bpb: float, peak_vram_mb: float):
    results_path = _env_str("RESULTS_PATH", None)
    if results_path is None:
        return
    p = Path(results_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    desc = _env_str("EXP_DESC", None)
    cfg_snapshot = dict(
        depth=DEPTH,
        aspect_ratio=ASPECT_RATIO,
        head_dim=HEAD_DIM,
        device_bs=DEVICE_BATCH_SIZE,
        total_bs=TOTAL_BATCH_SIZE,
        h_cycles=H_CYCLES,
        l_cycles=L_CYCLES,
        forward_dtype=FORWARD_DTYPE,
        embedding_lr=EMBEDDING_LR,
        matrix_lr=MATRIX_LR,
        unembedding_lr=UNEMBEDDING_LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        warmdown_ratio=WARMDOWN_RATIO,
        final_lr_frac=FINAL_LR_FRAC,
        logits_softcap=float(os.environ.get("LOGITS_SOFTCAP", "15") or "15"),
        zero_o_proj_init=int(os.environ.get("ZERO_O_PROJ_INIT", "1") or "1"),
        prenorm=int(os.environ.get("PRENORM", "0") or "0"),
        layer_scale_init=float(os.environ.get("LAYER_SCALE_INIT", "0") or "0"),
        seed=seed,
    )
    if desc is None:
        desc = json.dumps(
            cfg_snapshot,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    memory_gb = peak_vram_mb / 1024.0
    commit = _git_short_hash()
    line_tpl = f"{commit}\t{val_bpb:.6f}\t{memory_gb:.1f}\t"
    lock_path = p.with_suffix(p.suffix + ".lock")
    with open(lock_path, "a+") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        if not p.exists():
            p.write_text("commit\tval_bpb\tmemory_gb\tstatus\tdescription\n", encoding="utf-8")
        existing = p.read_text(encoding="utf-8").splitlines(keepends=True)
        best = _parse_best_val_bpb(existing)
        status = "keep" if best is None or val_bpb < best else "discard"
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"{line_tpl}{status}\t{desc}\n")
        best_path = p.with_name("best.json")
        new_best = best is None or val_bpb < best
        if new_best:
            best_payload = dict(commit=commit, val_bpb=val_bpb, memory_gb=memory_gb, description=desc, config=cfg_snapshot)
            best_path.write_text(json.dumps(best_payload, ensure_ascii=False), encoding="utf-8")

        elite_path = p.with_name("elite.json")
        elite = {}
        if elite_path.exists():
            try:
                elite = json.loads(elite_path.read_text(encoding="utf-8"))
            except Exception:
                elite = {}
        best_rec = elite.get("best")
        second_rec = elite.get("second")
        promising_rec = elite.get("promising")
        cur = dict(commit=commit, val_bpb=val_bpb, memory_gb=memory_gb, description=desc, config=cfg_snapshot)

        def better(x, y):
            if x is None:
                return True
            return float(y) < float(x)

        if best_rec is None or better(best_rec.get("val_bpb"), val_bpb):
            second_rec = best_rec
            best_rec = cur
        elif second_rec is None or better(second_rec.get("val_bpb"), val_bpb):
            second_rec = cur

        margin = float(os.environ.get("PROMISING_MARGIN", "0.05") or "0.05")
        base_cfg = (best_rec or {}).get("config") or {}
        cur_dist = _elite_distance(cfg_snapshot, base_cfg)
        cur_ok = best_rec is None or val_bpb <= float(best_rec.get("val_bpb")) + margin
        if cur_ok:
            best_dist = _elite_distance((best_rec or {}).get("config") or {}, base_cfg)
            second_dist = _elite_distance((second_rec or {}).get("config") or {}, base_cfg)
            prom_dist = _elite_distance((promising_rec or {}).get("config") or {}, base_cfg) if promising_rec else -1.0
            if cur_dist > prom_dist and cur_dist > best_dist and cur_dist > second_dist:
                promising_rec = cur

        elite_out = dict(best=best_rec, second=second_rec, promising=promising_rec)
        elite_path.write_text(json.dumps(elite_out, ensure_ascii=False), encoding="utf-8")
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


_append_result(val_bpb=val_bpb, peak_vram_mb=peak_vram_mb)
