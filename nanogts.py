# Golden Tree Snake (GTS) fork, 2026.
"""nanoGTS: GTS3 in one file. A bidirectional, attention-free, ternary masked-language model, its data and its training.

GTS3 (checkpoints/bert110m/phase3) is the best GTS masked LM: 14 layers of width 768, 112.9M parameters, trained on
7.38B tokens of English Wikipedia to validation loss 2.615 / masked accuracy 53.2%. This file is the method in plain
PyTorch, with no Triton and nothing from the rest of the repository. Its parameter names are the repository's, so the
GTS3 checkpoints load as they are (``load``), and it computes the same function as mamba_ssm/models/gts_encoder.py
with the same gradients. Checked on GTS3's own weights in float64 against the repository model: logits agree to
3e-14, the loss exactly, and all 199 parameter gradients (the straight-through ones included) to 3e-14. In float32 a
few tokens differ: a branch logit within rounding of zero can turn the other way when sums run in another order.
``selftest`` checks the fast paths here against their definitions.

The model
---------
token embedding (tied with the output layer, no positions) -> 14 x [x + Mixer(RMSNorm(x))] -> RMSNorm -> logits.
There is no attention and no MLP. Each block's mixer is a forest of binary trees in the manner of fast feedforward
networks (Belcak & Wattenhofer, 2023): a node computes logit = <x, node_in> + bias and adds coef * node_out to the
output; a token walks one root-to-leaf path (right if logit > 0). The forest has two kinds of tree, summed:

* bank: 32 depth-0 trees (every token visits all of them), so they are dense channels. They carry all the context:
  each tree is a channel of a bidirectional Mamba-2-style SSM. Token s writes dt_s * logit_s * B_s into the tree's
  state; token t reads <C_t, decayed state> from both directions, excluding itself:
      ctx[t] = sum_{s<t} exp(a_{s+1}+..+a_t) <C_fwd[t], B[s]> dt_s logit_s + sum_{s>t} exp(a_t+..+a_{s-1}) <C_bwd[t], B[s]> dt_s logit_s
  with a = -exp(A_log) * softplus(dt) <= 0 on one clock per head (8 heads of 4 trees, state 16). B, C_fwd, C_bwd and
  dt come from one small projection (ctx_proj). coef = gelu(logit) + ctx.
* deep: 4 stateless trees of depth 9 (1,023 nodes each, 10 visited per token). They hold most of the mixer's
  parameters and touch 1% of them per token. coef = gelu(logit). Branches train with a straight-through gradient: the
  hard step going forward, sigmoid(logit) going backward, so a branch logit learns from the difference between the
  outputs of its two subtrees, each followed down by the token's own decisions.

Stateful deep trees (an experiment, off by default; not GTS3): with --deep-ctx-levels K every node on the top K
levels of each deep tree is a channel of the same kind of SSM, written only by the tokens that visit it:
      ctx[t, a] = sum_{s != t, s visited a} decay(s -> t) <C[t], B[s]> dt_s logit_s(a),   coef = gelu(logit) + ctx
so the root hears every token, as a bank tree does, and a level-k node only the tokens that made the same k
decisions: context clustered by the trees' own routing. One clock per tree; B, C_fwd and C_bwd (state
--deep-ctx-state) from one more ternary projection per layer. Routes still depend on the logits alone, so the context
is computed after the walk and changes coefficients, not paths. Every token reads every stateful node, on its path or
not: the straight-through gradient compares a path with its alternative, and the alternative's coefficients are what
the token would have read there. The tokens' own writes are not moved in that comparison (the visits carry no
gradient). It costs nothing at K = 0 (GTS3's checkpoints load and compute as before) and with K = 4 adds 0.56M
parameters. On a GPU the context runs in the bank's scan, but the walk runs in PyTorch's gathers, not the Triton walk.
Starting from GTS3: train --init checkpoints/bert110m/phase3/checkpoint.pt --deep-ctx-levels 4 ...

Each kind first mixes neighbours with its own centred depthwise conv (width 3, identity at init) and quantises the
result to 8-bit integers per token. node_in, node_out and ctx_proj are ternary: absmean codes in {-1, 0, 1} times
one scale per group of 128 weights, recomputed every step from latent float weights (straight-through). Embeddings,
norms, convs, biases and the decay parameters stay in float.

The training (GTS3)
-------------------
RoBERTa-style masked LM on English Wikipedia (wikimedia/wikipedia 20231101.en; validation from the last shard) in
BERT's uncased WordPiece: 512-token windows of the article stream starting with [CLS], 15% of tokens chosen, of those
80% [MASK], 10% random, 10% kept; loss at the chosen positions only. AdamW (0.9, 0.98, eps 1e-6, weight decay 0.01 on
matrices), batch 64 x 512, gradient clipping 1.0, bf16 autocast, linear warmup then cosine to 10% of the peak.
Three phases, each resuming the previous one's weights, optimizer and sampler, re-warming from its last rate:

    python archive/nanogts.py prep  --out wiki1 --shards 0-6
    python archive/nanogts.py train --data wiki1 --out run --steps 53753  --lr 1.5e-3 --warmup 1000               # 2.815 / 50.5%
    python archive/nanogts.py prep  --out wiki2 --shards 7-19
    python archive/nanogts.py train --data wiki2 --out run --steps 108829 --lr 7.5e-4 --warmup 1000 --resume run/ckpt.pt  # 2.713 / 51.7%
    python archive/nanogts.py prep  --out wiki3 --shards 0-39
    python archive/nanogts.py train --data wiki3 --out run --steps 225197 --lr 5e-4  --warmup 2000 --resume run/ckpt.pt  # 2.615 / 53.2%

(--steps is the absolute step to end at. GTS3 fitted each phase's length to a time budget on one A100; these are the
step counts it reached: 1.76B + 1.80B + 3.81B tokens.)

    python archive/nanogts.py sample --ckpt run/ckpt.pt          # fill-mask examples
    python archive/nanogts.py selftest [--kernels]               # checks the fast paths against their definitions

Speed: on a GPU with Triton installed, the repository's training kernels run (they are copied in below, verbatim):
the bank's context as a chunked scan, linear in length; the deep trees' walk and straight-through gradient in
registers; the ternary quantiser fused. Blocks are torch.compile'd for the gradient steps (--no-compile turns it off),
and the deep-tree and vocabulary matrices are padded to multiples of 64 rows. That is the configuration GTS3 trained in
(about 188K tokens/s on one A100). Without Triton, or on a CPU, the same function runs in plain PyTorch: the bank's
context in the quadratic form (memory ~ length^2 per head; --micro-batch splits a batch, gradients accumulate exactly)
and the deep trees through a torch.autograd.Function that walks them with gathers. ``selftest`` checks the PyTorch
paths against their definitions in float64; ``selftest --kernels`` checks the Triton paths against the PyTorch ones
(without a GPU: TRITON_INTERPRET=1 python archive/nanogts.py selftest --kernels).
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------------------------------------- model


@dataclass
class GTSConfig:
    d_model: int = 768
    n_layer: int = 14
    vocab_size: int = 30522
    bank_trees: int = 32
    bank_heads: int = 8
    bank_state: int = 16
    deep_trees: int = 4
    deep_depth: int = 9
    deep_ctx_levels: int = 0  # stateful deep trees: the top levels whose nodes carry context (0: GTS3, stateless)
    deep_ctx_state: int = 16
    d_conv: int = 3
    ternary_group: int = 128
    act_bits: int = 8
    route_ste_temp: float = 1.0
    pad_token_id: int = 0
    norm_eps: float = 1e-5


def group_size(n, g):
    """Largest group size <= g that divides n."""
    if g is None or g >= n:
        return n
    while n % g:
        g -= 1
    return g


def ternary(w, group):
    """BitNet b1.58's absmean quantiser per group of weights along each row, straight-through to the latent weights."""
    g = group_size(w.shape[-1], group)
    if use_kernels(w):  # the same values in one Triton kernel
        return absmean_ternary_fused(w, g)
    wg = w.reshape(*w.shape[:-1], -1, g)
    scale = wg.abs().mean(-1, keepdim=True).clamp(min=1e-8)
    wq = ((wg / scale).clamp(-1, 1).round() * scale).reshape(w.shape)
    return wq.detach() + (w - w.detach())


def quantize_activations(x, bits):
    """Per-token absmax integer activations (rounded in float32), straight-through."""
    qmax = 2 ** (bits - 1) - 1
    xf = x.float()
    scale = qmax / xf.abs().amax(-1, keepdim=True).clamp(min=1e-5)
    xq = ((xf * scale).round().clamp(-qmax - 1, qmax) / scale).to(x.dtype)
    return xq.detach() + (x - x.detach())


def _f32(t):
    """At least float32 (the context and the walk; float64 stays float64)."""
    return t.to(torch.promote_types(t.dtype, torch.float32))


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Trees(nn.Module):
    """What both kinds of tree share: the centred depthwise conv and the node tables (UltraFastBERT's FFF init)."""

    def __init__(self, cfg, n_trees, depth):
        super().__init__()
        d = cfg.d_model
        self.cfg, self.n_trees, self.depth = cfg, n_trees, depth
        self.n_nodes = 2 ** (depth + 1) - 1  # per tree
        total = n_trees * self.n_nodes
        k_in, k_out = math.sqrt(1.0 / d), math.sqrt(1.0 / (n_trees * (depth + 1)))  # out: 1 / sqrt(nodes on a path)
        self.node_in = nn.Parameter(torch.empty(total, d).uniform_(-k_in, k_in))
        self.node_bias = nn.Parameter(torch.empty(total).uniform_(-k_in, k_in))
        self.node_bias._no_weight_decay = True
        self.node_out = nn.Parameter(torch.empty(total, d).uniform_(-k_out, k_out))
        self.conv1d = nn.Conv1d(d, d, cfg.d_conv, groups=d, padding=cfg.d_conv // 2, bias=True)
        with torch.no_grad():  # the identity: the trees start by routing on the token itself
            self.conv1d.weight.zero_()
            self.conv1d.weight[:, 0, cfg.d_conv // 2] = 1.0
            self.conv1d.bias.zero_()

    def q(self, w):
        return ternary(w, self.cfg.ternary_group)

    def local_mix(self, u, mask):
        """Centred depthwise conv (as shifted sums) over the masked input, then 8-bit activations."""
        u = u * mask.unsqueeze(-1)
        k, pad, length = self.cfg.d_conv, self.cfg.d_conv // 2, u.shape[1]
        up = F.pad(u, (0, 0, pad, k - 1 - pad))
        w = self.conv1d.weight.squeeze(1)  # (d, k)
        x = self.conv1d.bias + sum(up[:, j : j + length] * w[:, j] for j in range(k))
        return quantize_activations(x, self.cfg.act_bits)

    def add_clocks(self, n, H):
        """The SSM's parameters: one projection to [B, C_fwd, C_bwd, dt per head] and a decay rate per head."""
        self.ctx_state = n
        self.ctx_proj = nn.Linear(self.cfg.d_model, 3 * n + H, bias=False)
        dt = torch.exp(torch.rand(H) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus, as Mamba-2
        self.A_log = nn.Parameter(torch.log(torch.empty(H).uniform_(1, 16)))
        self.dt_bias._no_weight_decay = self.A_log._no_weight_decay = True

    def signals(self, x, mask):
        n = self.ctx_state
        p = F.linear(x, self.q(self.ctx_proj.weight))
        B, C_fwd, C_bwd = p[..., :n], p[..., n : 2 * n], p[..., 2 * n : 3 * n]
        dt = F.softplus(p[..., 3 * n :] + self.dt_bias)  # (b, l, heads)
        a = dt * -torch.exp(self.A_log.float()) * mask.unsqueeze(-1)  # per-token log-decay; padding stops no clock
        return B, C_fwd, C_bwd, dt, a


def bi_context(C_fwd, C_bwd, B, src, a):
    """The bidirectional context, the token's own term excluded. C_fwd, C_bwd, B: (b, l, n); src: (b, l, heads, p),
    what each token writes; a: (b, l, heads), the log-decays. Returns (b, l, heads, p), float32 at least:
        ctx[t] = sum_{s<t} exp(a_{s+1}+..+a_t) <C_fwd[t], B[s]> src[s] + sum_{s>t} exp(a_t+..+a_{s-1}) <C_bwd[t], B[s]> src[s]"""
    if use_kernels(src):  # the chunked scan, linear in length, both directions in the same launches
        return gts_scan_bi(C_fwd, C_bwd, B, src, a)
    length = src.shape[1]
    # weights[t, s, h] = <C[t], B[s]> * decay between s and t on head h's clock, both directions, zero at s = t
    Bf, a = _f32(B), _f32(a)
    cs = torch.cumsum(a, 1)
    cx = cs - a
    lower = torch.ones(length, length, dtype=torch.bool, device=src.device).tril(-1)[None, :, :, None]
    fwd = torch.exp((cs[:, :, None] - cs[:, None, :]).masked_fill(~lower, -torch.inf))
    bwd = torch.exp((cx[:, None, :] - cx[:, :, None]).masked_fill(~lower.transpose(1, 2), -torch.inf))
    w = (_f32(C_fwd) @ Bf.transpose(1, 2)).unsqueeze(-1) * fwd + (_f32(C_bwd) @ Bf.transpose(1, 2)).unsqueeze(-1) * bwd
    return torch.einsum("btsh,bshp->bthp", w, _f32(src))


class Bank(Trees):
    """Depth-0 trees with bidirectional SSM context, one clock per head."""

    def __init__(self, cfg):
        super().__init__(cfg, cfg.bank_trees, 0)
        self.add_clocks(cfg.bank_state, cfg.bank_heads)

    def forward(self, u, mask):
        b, length, _ = u.shape
        H, T = self.cfg.bank_heads, self.n_trees
        x = self.local_mix(u, mask)
        logit = F.linear(x, self.q(self.node_in), self.node_bias)  # (b, l, trees)
        B, C_fwd, C_bwd, dt, a = self.signals(x, mask)
        src = dt.repeat_interleave(T // H, -1) * mask.unsqueeze(-1) * logit  # what each token writes to each tree
        ctx = bi_context(C_fwd, C_bwd, B, src.view(b, length, H, T // H), a).reshape(b, length, T)
        coef = F.gelu(logit) + ctx.to(logit.dtype)
        return (coef @ self.q(self.node_out)) * mask.unsqueeze(-1)


def _dgelu(x):
    return 0.5 * (1.0 + torch.erf(x * 0.7071067811865476)) + x * torch.exp(-0.5 * x * x) * 0.3989422804014327


def _walk(L, depth):
    """L: (tokens, trees, nodes). The node index along each token's path in each tree, (tokens, trees, depth + 1)."""
    cur = torch.zeros(L.shape[:2], dtype=torch.long, device=L.device)
    path = []
    for _ in range(depth + 1):
        path.append(cur)
        cur = 2 * cur + 1 + (L.gather(2, cur.unsqueeze(-1)).squeeze(-1) > 0).long()
    return torch.stack(path, -1)


def _node_coef(L3, E3, idx):
    """A node's coefficient at the node indices idx: gelu(logit), plus its context when the tree is stateful."""
    c = F.gelu(_f32(L3.gather(2, idx)))
    return c if E3 is None else c + _f32(E3.gather(2, idx))


class RouteSTE(torch.autograd.Function):
    """out = sum over each token's path nodes of coef[node] * W[node], with the straight-through branch gradient,
    without forming the path weights (``Deep.reference`` is the definition). coef = gelu(logit), plus E, the context,
    at the top E.shape[-1] nodes of each tree (stateful trees; E: (tokens, trees, nodes with context) or None).
    Backward, with g = dout @ W^T, the only nonzero logit and context gradients are at path nodes; at path node a on
    level k:
        dL[a] = gelu'(L[a]) g[a] + sign * (on[a] - alt[a]) * sigmoid'(L[a] / temp) / temp,   dE[a] = g[a]
    on[a]: coef * g summed over the path below a; alt[a]: the same over the chain from a's other child following the
    token's own decisions; sign +1 if the token went right at a."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, L, W, n_trees, depth, temp, E=None):
        N = L.shape[0]
        L3 = L.view(N, n_trees, -1)
        E3 = None if E is None else F.pad(E, (0, L3.shape[2] - E.shape[2]))
        path = _walk(L3, depth)
        A = torch.zeros_like(L3).scatter(2, path, _node_coef(L3, E3, path).to(L.dtype)).view(N, -1)
        ctx.save_for_backward(L, W, E)
        ctx.cfg = n_trees, depth, temp
        return A @ W

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout):
        L, W, E = ctx.saved_tensors
        n_trees, depth, temp = ctx.cfg
        N = L.shape[0]
        L3 = L.view(N, n_trees, -1)
        E3 = None if E is None else F.pad(E, (0, L3.shape[2] - E.shape[2]))
        path = _walk(L3, depth)
        Lp = _f32(L3.gather(2, path))
        cp = _node_coef(L3, E3, path)
        A = torch.zeros_like(L3).scatter(2, path, cp.to(L.dtype)).view(N, -1)
        dW = A.t() @ dout
        G = (dout @ W.t()).view(N, n_trees, -1)
        Gp = _f32(G.gather(2, path))
        c = cp * Gp
        on = c.sum(-1, keepdim=True) - c.cumsum(-1)  # the path strictly below each node
        d = _dgelu(Lp) * Gp
        for k in range(depth):
            right = Lp[..., k] > 0
            s = 2 * path[..., k] + 2 - right.long()  # the other child
            alt = torch.zeros_like(on[..., k])
            for _ in range(depth - k):
                ls = _f32(L3.gather(2, s.unsqueeze(-1)).squeeze(-1))
                alt = alt + _node_coef(L3, E3, s.unsqueeze(-1)).squeeze(-1) * _f32(G.gather(2, s.unsqueeze(-1)).squeeze(-1))
                s = 2 * s + 1 + (ls > 0).long()
            p = torch.sigmoid(Lp[..., k] / temp)
            d[..., k] += torch.where(right, on[..., k] - alt, alt - on[..., k]) * p * (1 - p) / temp
        dL = torch.zeros(L3.shape, dtype=d.dtype, device=L.device).scatter(2, path, d).view(N, -1).to(L.dtype)
        dE = None
        if E is not None:
            dE = torch.zeros(L3.shape, dtype=Gp.dtype, device=L.device).scatter(2, path, Gp)[..., : E.shape[2]].to(E.dtype)
        return dL, dW, None, None, None, dE


class Deep(Trees):
    """Deep trees, hard routing with the straight-through branch gradient. Stateless (GTS3), or with
    deep_ctx_levels = K, stateful: each node on the top K levels of each tree carries the bidirectional context of the
    tokens that visited it. One clock per tree; B, C_fwd and C_bwd shared by all of a layer's trees."""

    def __init__(self, cfg):
        super().__init__(cfg, cfg.deep_trees, cfg.deep_depth)
        self.ctx_levels = min(cfg.deep_ctx_levels, cfg.deep_depth + 1)
        if self.ctx_levels:
            self.add_clocks(cfg.deep_ctx_state, cfg.deep_trees)

    def context(self, x, al, mask, visit=None):
        """al: (b, l, trees, nodes), every node's logit. Context at the top 2^K - 1 nodes of each tree, for every token,
        on its path or not (off it, what the token would read had it gone there; the branch gradient needs it):
            ctx[t, a] = sum_{s != t, s visited a} decay(s -> t) <C[t], B[s]> dt_s logit_s(a)
        visit (b, l, trees, 2^K - 1): which tokens visited which nodes, by default the walk on al. Going forward a
        token's route depends only on the logits, so the context changes coefficients, not paths."""
        b, length, T, _ = al.shape
        m = 2**self.ctx_levels - 1
        top = al[..., :m]
        if visit is None:
            with torch.no_grad():
                path = _walk(top.reshape(b * length, T, m), self.ctx_levels - 1).view(b, length, T, -1)
                visit = torch.zeros_like(top).scatter(-1, path, 1.0)
        B, C_fwd, C_bwd, dt, a = self.signals(x, mask)
        src = (dt * mask.unsqueeze(-1)).unsqueeze(-1) * visit * top  # what each token writes to each node it visits
        return bi_context(C_fwd, C_bwd, B, src, a)

    def forward(self, u, mask):
        b, length, d = u.shape
        x = self.local_mix(u, mask)
        w_in, w_out, bias = self.q(self.node_in), self.q(self.node_out), self.node_bias
        if self.ctx_levels:  # the context through bi_context (the bank's scan on a GPU), the walk in PyTorch
            L = F.linear(x, w_in, bias)
            E = self.context(x, L.view(b, length, self.n_trees, -1), mask).to(L.dtype)
            E = E.reshape(b * length, self.n_trees, -1)
            out = RouteSTE.apply(L.reshape(b * length, -1), w_out, self.n_trees, self.depth, self.cfg.route_ste_temp, E)
            return out.view(b, length, d).to(x.dtype) * mask.unsqueeze(-1)
        if use_kernels(x):  # Triton walk; node rows padded to a multiple of 64 (4 x 1,023 misaligns every GEMM)
            pad = -w_in.shape[0] % 64
            w_in, w_out, bias = F.pad(w_in, (0, 0, 0, pad)), F.pad(w_out, (0, 0, 0, pad)), F.pad(bias, (0, pad))
            L = F.linear(x, w_in, bias).reshape(b * length, -1)
            out, _ = route_ste_out(L, w_out, self.n_trees, self.depth, "gelu", self.cfg.route_ste_temp, n_nodes=self.n_nodes)
            return out.view(b, length, d).to(x.dtype) * mask.unsqueeze(-1)
        L = F.linear(x, w_in, bias).reshape(b * length, -1)  # every node's logit
        out = RouteSTE.apply(L, w_out, self.n_trees, self.depth, self.cfg.route_ste_temp)
        return out.view(b, length, d).to(x.dtype) * mask.unsqueeze(-1)

    def reference(self, u, mask):
        """The definition: every node's coefficient times its path weight, the product of the branch values above
        it (the hard step going forward, sigmoid(logit / temp) going backward). In a stateful tree a node's
        coefficient adds its context, written by the tokens whose (hard) path weight there is 1."""
        b, length, _ = u.shape
        x = self.local_mix(u, mask)
        al = F.linear(x, self.q(self.node_in), self.node_bias).view(b, length, self.n_trees, self.n_nodes)
        p = torch.sigmoid(al / self.cfg.route_ste_temp)
        right = (al > 0).to(al.dtype) + (p - p.detach())
        levels = [al.new_ones(b, length, self.n_trees, 1)]
        for k in range(self.depth):
            g = right[..., 2**k - 1 : 2 ** (k + 1) - 1]
            levels.append(torch.stack([levels[-1] * (1 - g), levels[-1] * g], -1).flatten(3))  # children 2i+1, 2i+2
        pi = torch.cat(levels, -1)  # (b, l, trees, nodes)
        coef = F.gelu(al)
        if self.ctx_levels:
            m = 2**self.ctx_levels - 1
            ctx = self.context(x, al, mask, visit=pi.detach()[..., :m])
            coef = coef + F.pad(ctx.to(al.dtype), (0, self.n_nodes - m))
        return ((pi * coef).flatten(2) @ self.q(self.node_out)) * mask.unsqueeze(-1)


class Mixer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bank, self.deep = Bank(cfg), Deep(cfg)

    def forward(self, u, mask):
        return self.bank(u, mask) + self.deep(u, mask)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mixer = Mixer(cfg)

    def forward(self, x, mask):
        return x + self.mixer(self.norm(x), mask)


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[cfg.pad_token_id].zero_()
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.compiled = None  # compile_blocks(): torch.compile'd blocks, used for gradient steps only

    def compile_blocks(self):
        self.compiled = [torch.compile(layer, dynamic=False) for layer in self.layers]

    def forward(self, ids):
        mask = (ids != self.cfg.pad_token_id).float()
        x = self.embedding(ids)
        for layer in self.compiled if self.compiled and torch.is_grad_enabled() else self.layers:
            x = layer(x, mask.to(x.dtype))
        return self.norm_f(x)


class GTS(nn.Module):
    """Masked LM: the encoder and a tied output layer with a bias."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = Encoder(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=True)
        self.lm_head.weight = self.backbone.embedding.weight
        nn.init.zeros_(self.lm_head.bias)

    def forward(self, ids, labels=None):
        """Without labels: logits (b, l, vocab). With labels (-100 = unscored): the summed cross-entropy and the logits
        at the labelled positions only, (n, vocab), in row-major order."""
        h = self.backbone(ids)
        if labels is None:
            return self.head(h)
        sel = labels != -100
        logits = self.head(h[sel])
        return F.cross_entropy(logits.float(), labels[sel], reduction="sum"), logits

    def head(self, h):
        """The output layer; on a GPU with the vocabulary padded to a multiple of 64 rows (30,522 misaligns the GEMM)."""
        w, bias = self.lm_head.weight, self.lm_head.bias
        pad = -w.shape[0] % 64
        if pad and h.is_cuda:
            return F.linear(h, F.pad(w, (0, 0, 0, pad)), F.pad(bias, (0, pad)))[..., : w.shape[0]]
        return F.linear(h, w, bias)


def config_from(rc):
    return GTSConfig(**{f.name: rc[f.name] for f in fields(GTSConfig) if f.name in rc})


def load(path, device="cpu"):
    """A nanoGTS ckpt.pt, or the repository's checkpoint.pt (float) or binarized.pt (2-bit ternary codes and scales)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    rc = blob["config"]
    assert rc.get("mixer", "mixed") == "mixed" and not rc.get("causal") and rc.get("loops", 1) == 1, "not a GTS3-style model"
    model = GTS(config_from(rc))
    if "model" in blob:
        model.load_state_dict(blob["model"])
        return model.to(device)
    state = dict(blob["float"])  # binarized: rebuild latent weights whose absmean quantisation gives back the codes
    for name, e in blob["ternary"].items():
        u = torch.stack([(e["packed"] >> s) & 3 for s in (0, 2, 4, 6)], 1).flatten()
        codes = (u[: math.prod(e["shape"])].float() - 1).reshape(e["shape"])
        cg = codes.reshape(*codes.shape[:-1], -1, e["group"])
        frac = (cg != 0).float().mean(-1, keepdim=True).clamp(min=1.0 / e["group"])
        state[name] = (cg * e["scales"].unsqueeze(-1) / frac).reshape(codes.shape)
    state["lm_head.weight"] = state["backbone.embedding.weight"]
    model.load_state_dict(state)
    return model.to(device)


# ----------------------------------------------------------------------------------------------- Triton kernels
# The repository's training kernels, verbatim (mamba_ssm/ops/ternary_fused.py, gts_scan.py, gts_route.py). On a GPU
# with Triton they replace three PyTorch paths that compute the same values and gradients:
#   absmean_ternary_fused  the ternary quantiser in one read and one write
#   gts_scan_bi            the bank's bidirectional context as a chunked scan, linear in length (Mamba-2's SSD with the
#                          token's own term excluded), instead of the quadratic form
#   route_ste_out          the deep trees' walk and straight-through gradient, in registers, instead of gathers
# Without a GPU they run in Triton's interpreter if TRITON_INTERPRET=1 (slow; how ``selftest --kernels`` checks them).

try:
    import triton
    import triton.language as tl
except ImportError:  # the PyTorch paths need nothing else
    triton = None

HAVE_TRITON = triton is not None
USE_KERNELS = True  # False forces the PyTorch paths everywhere


def use_kernels(t):
    return USE_KERNELS and HAVE_TRITON and (t.is_cuda or os.environ.get("TRITON_INTERPRET") == "1")


if HAVE_TRITON:

    @triton.jit
    def _absmean_kernel(Wp, Op, n_groups, eps, G: tl.constexpr, BLOCK_G: tl.constexpr, ROWS: tl.constexpr):
        r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        j = tl.arange(0, BLOCK_G)
        ok = (r < n_groups)[:, None] & (j < G)[None, :]
        w = tl.load(Wp + r[:, None] * G + j[None, :], mask=ok, other=0.0).to(tl.float32)
        scale = tl.maximum(tl.sum(tl.abs(w), axis=1) / G, eps)[:, None]
        q = w / scale
        q = tl.minimum(tl.maximum(q, -1.0), 1.0)
        code = tl.where(q > 0.5, 1.0, tl.where(q < -0.5, -1.0, 0.0))  # torch.round: halves go to even, so 0.5 -> 0
        tl.store(Op + r[:, None] * G + j[None, :], code * scale, mask=ok)


class _AbsmeanSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, g, eps, out_dtype):
        wc = w.contiguous()
        out = torch.empty(w.shape, device=w.device, dtype=out_dtype)
        n_groups = wc.numel() // g
        rows = 16
        _absmean_kernel[(triton.cdiv(n_groups, rows),)](wc, out, n_groups, eps, G=g, BLOCK_G=triton.next_power_of_2(g), ROWS=rows)
        return out

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None, None


def absmean_ternary_fused(w, g, eps=1e-8):
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else w.dtype
    return _AbsmeanSTE.apply(w, g, eps, dtype)


# --- the bank's scan (gts_scan.py)

if HAVE_TRITON:

    @triton.jit
    def _chunk_clock(Ap, c, i, L, s_al, rev, CHUNK: tl.constexpr, EXCL: tl.constexpr):
        """Positions of chunk c's rows in processing order (reversed if rev), which are real, the clock relative to
        the chunk start (inclusive or exclusive running sum of a), and the chunk's total."""
        k = c * CHUNK + i
        ok = k < L
        pos = tl.where(rev != 0, L - 1 - k, k)
        a = tl.load(Ap + pos * s_al, mask=ok, other=0.0)
        incl = tl.cumsum(a, axis=0)
        total = tl.sum(a, axis=0)
        clock = incl - a if EXCL else incl
        return ok, pos, clock, total

    @triton.jit
    def _state_kernel(
        Kp, Vp, Ap, STp, TOTp, L, H, BH, n_chunks, rev_base,
        s_kd, s_kb, s_kl, s_vd, s_vb, s_vl, s_vh, s_ab, s_al,
        N: tl.constexpr, P: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr, CHUNK: tl.constexpr,
        EXCL: tl.constexpr, PREC: tl.constexpr,
    ):
        c = tl.program_id(0)
        bh = tl.program_id(1)
        d = tl.program_id(2)
        b = bh // H
        h = bh % H
        i = tl.arange(0, CHUNK)
        n = tl.arange(0, BLOCK_N)
        p = tl.arange(0, BLOCK_P)
        ok, pos, clock, total = _chunk_clock(Ap + b * s_ab + h, c, i, L, s_al, rev_base ^ d, CHUNK, EXCL)
        k = tl.load(Kp + d * s_kd + b * s_kb + pos[:, None] * s_kl + n[None, :], mask=ok[:, None] & (n < N)[None, :], other=0.0)
        v = tl.load(Vp + d * s_vd + b * s_vb + h * s_vh + pos[:, None] * s_vl + p[None, :], mask=ok[:, None] & (p < P)[None, :], other=0.0)
        wk = k * tl.exp(total - clock)[:, None]  # reference: the inclusive clock at the chunk's last token
        S = tl.dot(tl.trans(wk), v, input_precision=PREC)
        row = (d * BH + bh) * n_chunks + c
        tl.store(STp + row * BLOCK_N * BLOCK_P + n[:, None] * BLOCK_P + p[None, :], S)
        tl.store(TOTp + row, total)

    @triton.jit
    def _pass_kernel(STp, TOTp, n_chunks, BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr):
        """in[c] = exp(T[c-1]) * in[c-1] + own[c-1], in[0] = 0: the state entering each chunk, in place.
        One program per (direction, batch * head) row of chunks."""
        r = tl.program_id(0)
        tile = STp + r * n_chunks * BLOCK_N * BLOCK_P + tl.arange(0, BLOCK_N)[:, None] * BLOCK_P + tl.arange(0, BLOCK_P)[None, :]
        carry = tl.zeros((BLOCK_N, BLOCK_P), dtype=tl.float32)
        for c in range(0, n_chunks):
            own = tl.load(tile + c * BLOCK_N * BLOCK_P)
            tl.store(tile + c * BLOCK_N * BLOCK_P, carry)
            carry = carry * tl.exp(tl.load(TOTp + r * n_chunks + c)) + own

    @triton.jit
    def _out_kernel(
        Qp, Kp, Vp, Up, Ap, STp, O1p, O2p, Yp, DCp, L, H, BH, n_chunks, rev_base,
        s_qd, s_qb, s_ql, s_kd, s_kb, s_kl, s_vd, s_vb, s_vl, s_vh, s_ud, s_ub, s_ul, s_uh, s_ab, s_al,
        s_o1d, s_o1b, s_o1l, s_o1h, s_o2d, s_o2b, s_o2l, s_o2h,
        N: tl.constexpr, P: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr, CHUNK: tl.constexpr,
        HAS_O1: tl.constexpr, HAS_O2: tl.constexpr, HAS_DCLOCK: tl.constexpr, EXCL: tl.constexpr, PREC: tl.constexpr,
    ):
        """HAS_DCLOCK (the transposed pass, where V = dY, U = X and O1 = dX): also write <dY, Y> - <X, dX> per row,
        with Y laid out like O1 and the result in DCp, (directions, b, l, h)."""
        c = tl.program_id(0)
        bh = tl.program_id(1)
        d = tl.program_id(2)
        b = bh // H
        h = bh % H
        i = tl.arange(0, CHUNK)
        n = tl.arange(0, BLOCK_N)
        p = tl.arange(0, BLOCK_P)
        n_ok = n < N
        p_ok = p < P
        ok, pos, clock, total = _chunk_clock(Ap + b * s_ab + h, c, i, L, s_al, rev_base ^ d, CHUNK, EXCL)
        row = (d * BH + bh) * n_chunks + c
        S = tl.load(STp + row * BLOCK_N * BLOCK_P + n[:, None] * BLOCK_P + p[None, :])  # zero for c = 0
        dec = tl.exp(clock)  # from the end of the previous chunk (clock 0) to each row
        D = tl.exp(tl.where(i[:, None] > i[None, :], clock[:, None] - clock[None, :], -float("inf")))
        k = tl.load(Kp + d * s_kd + b * s_kb + pos[:, None] * s_kl + n[None, :], mask=ok[:, None] & n_ok[None, :], other=0.0)
        v = tl.load(Vp + d * s_vd + b * s_vb + h * s_vh + pos[:, None] * s_vl + p[None, :], mask=ok[:, None] & p_ok[None, :], other=0.0)
        if HAS_O1:
            q = tl.load(Qp + d * s_qd + b * s_qb + pos[:, None] * s_ql + n[None, :], mask=ok[:, None] & n_ok[None, :], other=0.0)
            M = tl.dot(q, tl.trans(k), input_precision=PREC) * D
            o1 = tl.dot(M, v, input_precision=PREC) + dec[:, None] * tl.dot(q, S, input_precision=PREC)
            tl.store(O1p + d * s_o1d + b * s_o1b + h * s_o1h + pos[:, None] * s_o1l + p[None, :], o1, mask=ok[:, None] & p_ok[None, :])
        if HAS_O2:
            u = tl.load(Up + d * s_ud + b * s_ub + h * s_uh + pos[:, None] * s_ul + p[None, :], mask=ok[:, None] & p_ok[None, :], other=0.0)
            M2 = tl.dot(u, tl.trans(v), input_precision=PREC) * D
            o2 = tl.dot(M2, k, input_precision=PREC) + dec[:, None] * tl.dot(u, tl.trans(S), input_precision=PREC)
            tl.store(O2p + d * s_o2d + b * s_o2b + h * s_o2h + pos[:, None] * s_o2l + n[None, :], o2, mask=ok[:, None] & n_ok[None, :])
        if HAS_DCLOCK:
            y = tl.load(Yp + d * s_o1d + b * s_o1b + h * s_o1h + pos[:, None] * s_o1l + p[None, :], mask=ok[:, None] & p_ok[None, :], other=0.0)
            dc = tl.sum(v * y, axis=1) - tl.sum(u * o1, axis=1)
            tl.store(DCp + ((d * (BH // H) + b) * L + pos) * H + h, dc, mask=ok)

    @triton.jit
    def _suffix_kernel(Dp, Op, L, H, BH, s_d, s_b, s_l, rev_base, EXCL: tl.constexpr, BLOCK: tl.constexpr):
        """Op[k] = sum of Dp over the rows at or after k in processing order (strictly after if EXCL); one program
        per (batch * head, direction)."""
        bh = tl.program_id(0)
        d = tl.program_id(1)
        b = bh // H
        h = bh % H
        rev = rev_base ^ d
        Dp += d * s_d + b * s_b + h
        Op += d * s_d + b * s_b + h
        i = tl.arange(0, BLOCK)
        carry = 0.0
        for blk in range(0, tl.cdiv(L, BLOCK)):
            k = L - 1 - (blk * BLOCK + i)  # processing index, walked from the end
            ok = k >= 0
            pos = tl.where(rev != 0, L - 1 - k, k)
            dd = tl.load(Dp + pos * s_l, mask=ok, other=0.0)
            run = carry + tl.cumsum(dd, axis=0)
            tl.store(Op + pos * s_l, run - dd if EXCL else run, mask=ok)
            carry += tl.sum(dd, axis=0)


def _blocks(n, p):
    return max(16, triton.next_power_of_2(n)), max(16, triton.next_power_of_2(p))


def _ds(t, nd):
    """Direction stride: 0 for a tensor both directions share (a plain (b, l, ...) tensor), else its first stride."""
    return t.stride(0) if t.dim() == nd + 1 else 0


def _bl(t, nd):
    """The (b, l, ...) strides of a tensor that may carry a leading direction axis."""
    return t.stride()[1:] if t.dim() == nd + 1 else t.stride()


def _states(K, V, a, rev_base, n_dir, excl, chunk, prec):
    """K: (b, l, n) or (dirs, b, l, n); V: (b, l, h, p) or (dirs, b, l, h, p). Returns the entering states,
    (dirs, b * h, chunks, BN, BP)."""
    b, length, h = a.shape
    p, n = V.shape[-1], K.shape[-1]
    bn, bp = _blocks(n, p)
    nc = triton.cdiv(length, chunk)
    st = torch.empty(n_dir, b * h, nc, bn, bp, device=V.device, dtype=torch.float32)
    tot = torch.empty(n_dir, b * h, nc, device=V.device, dtype=torch.float32)
    kb, vb = _bl(K, 3), _bl(V, 4)
    _state_kernel[(nc, b * h, n_dir)](
        K, V, a, st, tot, length, h, b * h, nc, int(rev_base),
        _ds(K, 3), kb[0], kb[1], _ds(V, 4), vb[0], vb[1], vb[2], a.stride(0), a.stride(1),
        N=n, P=p, BLOCK_N=bn, BLOCK_P=bp, CHUNK=chunk, EXCL=excl, PREC=prec,
    )
    _pass_kernel[(n_dir * b * h,)](st, tot, nc, BLOCK_N=bn, BLOCK_P=bp)
    return st


def _outputs(Q, K, V, U, a, st, rev_base, n_dir, excl, want_o1, want_o2, chunk, prec, Y=None):
    """Outputs per direction: O1 (dirs, b, l, h, p), O2 (dirs, b, l, h, n), and with Y the clock gradient
    (dirs, b, l, h)."""
    b, length, h = a.shape
    p, n = V.shape[-1], K.shape[-1]
    dev = V.device
    bn, bp = _blocks(n, p)
    nc = triton.cdiv(length, chunk)
    o1 = torch.empty(n_dir, b, length, h, p, device=dev, dtype=torch.float32) if want_o1 else torch.empty(1, 1, 1, 1, 1, device=dev)
    o2 = torch.empty(n_dir, b, length, h, n, device=dev, dtype=torch.float32) if want_o2 else torch.empty(1, 1, 1, 1, 1, device=dev)
    Q = Q if want_o1 else K
    U = U if want_o2 else V
    dclock = torch.empty(n_dir, b, length, h, device=dev, dtype=torch.float32) if Y is not None else o1
    qb, kb, vb, ub = _bl(Q, 3), _bl(K, 3), _bl(V, 4), _bl(U, 4)
    _out_kernel[(nc, b * h, n_dir)](
        Q, K, V, U, a, st, o1, o2, Y if Y is not None else o1, dclock, length, h, b * h, nc, int(rev_base),
        _ds(Q, 3), qb[0], qb[1], _ds(K, 3), kb[0], kb[1], _ds(V, 4), vb[0], vb[1], vb[2], _ds(U, 4), ub[0], ub[1], ub[2],
        a.stride(0), a.stride(1),
        o1.stride(0), o1.stride(1), o1.stride(2), o1.stride(3), o2.stride(0), o2.stride(1), o2.stride(2), o2.stride(3),
        N=n, P=p, BLOCK_N=bn, BLOCK_P=bp, CHUNK=chunk,
        HAS_O1=want_o1, HAS_O2=want_o2, HAS_DCLOCK=Y is not None, EXCL=excl, PREC=prec,
    )
    return (o1 if want_o1 else None), (o2 if want_o2 else None), (dclock if Y is not None else None)


def _suffix(d, rev_base, excl):
    """d: (dirs, b, l, h) -> the same shape, each direction summed onwards in its own processing order."""
    n_dir, b, length, h = d.shape
    out = torch.empty_like(d)
    _suffix_kernel[(b * h, n_dir)](d, out, length, h, b * h, d.stride(0), d.stride(1), d.stride(2), int(rev_base),
                                   EXCL=excl, BLOCK=1024)
    return out


def _unit_last(t):
    """float32 with a unit stride along the last dimension (the kernels take every other stride as given); GTS's B
    and C are column slices of one projection and need no copy."""
    t = t.float()
    return t if t.stride(-1) == 1 else t.contiguous()


class _GTSScan(torch.autograd.Function):
    """One direction (n_dir = 1, C: (b, l, n)) or both (n_dir = 2, C: (2, b, l, n), direction 1 reversed relative to
    direction 0). The output is the sum over directions."""

    @staticmethod
    def forward(ctx, C, B, X, a, reverse, excl, chunk, prec):
        n_dir = 2 if C.dim() == 4 else 1
        C, B, X, a = (_unit_last(t) for t in (C, B, X, a))
        st = _states(B, X, a, reverse, n_dir, excl, chunk, prec)
        Yd, _, _ = _outputs(C, B, X, None, a, st, reverse, n_dir, excl, True, False, chunk, prec)
        ctx.save_for_backward(C, B, X, a, Yd, st)
        ctx.cfg = reverse, excl, chunk, prec, n_dir
        return Yd.sum(0) if n_dir > 1 else Yd[0]

    @staticmethod
    def backward(ctx, dY):
        C, B, X, a, Yd, st = ctx.saved_tensors
        r, e, chunk, prec, n_dir = ctx.cfg
        dY = _unit_last(dY)
        # The transposed scan runs the other way on the other clock: dX[j] = sum_i w <B[j], C[i]> dY[i] and
        # dB[j] = sum_i w <X[j], dY[i]> C[i], one pass. dC[i] = sum_j w <dY[i], X[j]> B[j] reuses the forward's states.
        tst = _states(C, dY, a, not r, n_dir, not e, chunk, prec)
        dX, dB, dclock = _outputs(B, C, dY, X, a, tst, not r, n_dir, not e, True, True, chunk, prec, Yd)
        _, dC, _ = _outputs(None, B, X, dY, a, st, r, n_dir, e, False, True, chunk, prec)
        da = _suffix(dclock, r, e).sum(0)
        dC = dC.sum(3)  # over heads: (dirs, b, l, n)
        return (dC if n_dir > 1 else dC[0]), dB.sum((0, 3)), dX.sum(0), da, None, None, None, None


def _precision(C, precision):
    if precision is None:
        precision = "tf32" if (C.is_cuda and torch.backends.cuda.matmul.allow_tf32) else "ieee"
    return precision


def gts_scan(C, B, X, a, reverse=False, excl=False, chunk=64, precision=None):
    """C, B: (b, l, n); X: (b, l, h, p); a: (b, l, h) per-token log-decays <= 0. Returns (b, l, h, p) float32.

    ``precision`` is "ieee" or "tf32" for the chunk matmuls; by default it follows
    ``torch.backends.cuda.matmul.allow_tf32``. ``excl`` uses the exclusive running sum (the transposed scan)."""
    if not HAVE_TRITON:
        raise RuntimeError("gts_scan needs triton; use gts_scan_reference")
    return _GTSScan.apply(C, B, X, a, reverse, excl, chunk, _precision(C, precision))


def gts_scan_bi(C_fwd, C_bwd, B, X, a, chunk=64, precision=None):
    """gts_scan(C_fwd, B, X, a) + gts_scan(C_bwd, B, X, a, reverse=True), both directions in the same launches."""
    if not HAVE_TRITON:
        raise RuntimeError("gts_scan_bi needs triton; use gts_scan_reference")
    return _GTSScan.apply(torch.stack([C_fwd, C_bwd]), B, X, a, False, False, chunk, _precision(C_fwd, precision))


# --- the deep trees' walk (gts_route.py)

if HAVE_TRITON:

    @triton.jit
    def _coef(x, ACT: tl.constexpr):
        """ACT 0: gelu (also split with no context), 1: linear."""
        if ACT == 0:
            return 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
        return x

    @triton.jit
    def _dcoef(x, ACT: tl.constexpr):
        if ACT == 0:
            return 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476)) + x * tl.exp(-0.5 * x * x) * 0.3989422804014327
        return tl.full(x.shape, 1.0, tl.float32)

    @triton.jit
    def _route_fwd(Lp, Ap, NODESp, n_tok, s_l, s_a, n_trees,
                   N_NODES: tl.constexpr, DEPTH: tl.constexpr, ACT: tl.constexpr, BLOCK: tl.constexpr, STORE_NODES: tl.constexpr):
        tok = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tree = tl.program_id(1)
        ok = tok < n_tok
        base = tree * N_NODES
        cur = tl.zeros((BLOCK,), dtype=tl.int32)
        for k in tl.static_range(DEPTH + 1):
            node = base + cur
            lg = tl.load(Lp + tok * s_l + node, mask=ok, other=0.0).to(tl.float32)
            tl.store(Ap + tok * s_a + node, _coef(lg, ACT), mask=ok)
            if STORE_NODES:
                tl.store(NODESp + (tok * n_trees + tree) * (DEPTH + 1) + k, node, mask=ok)
            cur = 2 * cur + 1 + (lg > 0).to(tl.int32)

    @triton.jit
    def _route_bwd(Lp, Gp, DLp, n_tok, s_l, s_g, s_d, inv_temp,
                   N_NODES: tl.constexpr, DEPTH: tl.constexpr, ACT: tl.constexpr, BLOCK: tl.constexpr, STE: tl.constexpr):
        tok = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tree = tl.program_id(1)
        ok = tok < n_tok
        base = tree * N_NODES
        # pass 1: sum of coef * g over the whole path
        cur = tl.zeros((BLOCK,), dtype=tl.int32)
        total = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in tl.static_range(DEPTH + 1):
            lg = tl.load(Lp + tok * s_l + base + cur, mask=ok, other=0.0).to(tl.float32)
            total += _coef(lg, ACT) * tl.load(Gp + tok * s_g + base + cur, mask=ok, other=0.0).to(tl.float32)
            cur = 2 * cur + 1 + (lg > 0).to(tl.int32)
        # pass 2: each path node's gradient
        cur = tl.zeros((BLOCK,), dtype=tl.int32)
        done = tl.zeros((BLOCK,), dtype=tl.float32)  # coef * g summed over the path down to this node
        for k in tl.static_range(DEPTH + 1):
            lg = tl.load(Lp + tok * s_l + base + cur, mask=ok, other=0.0).to(tl.float32)
            gk = tl.load(Gp + tok * s_g + base + cur, mask=ok, other=0.0).to(tl.float32)
            done += _coef(lg, ACT) * gk
            d = _dcoef(lg, ACT) * gk
            right = lg > 0
            if STE and k < DEPTH:  # without route_ste, hard branches get no gradient (FFF): only coef' * g
                on = total - done  # the path below this node
                s = 2 * cur + 2 - right.to(tl.int32)  # the other child
                alt = tl.zeros((BLOCK,), dtype=tl.float32)
                for j in tl.static_range(DEPTH - k):
                    ls = tl.load(Lp + tok * s_l + base + s, mask=ok, other=0.0).to(tl.float32)
                    alt += _coef(ls, ACT) * tl.load(Gp + tok * s_g + base + s, mask=ok, other=0.0).to(tl.float32)
                    s = 2 * s + 1 + (ls > 0).to(tl.int32)
                p = 1.0 / (1.0 + tl.exp(-lg * inv_temp))
                d += tl.where(right, on - alt, alt - on) * p * (1.0 - p) * inv_temp
            tl.store(DLp + tok * s_d + base + cur, d, mask=ok)
            cur = 2 * cur + 1 + right.to(tl.int32)


def _grid(n_tok, n_trees, block):
    return (triton.cdiv(n_tok, block), n_trees)


class _RouteSTE(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, L, W, n_trees, n_nodes, depth, act, temp, want_nodes, ste):
        n_tok = L.shape[0]
        L = L.contiguous()
        A = torch.zeros_like(L)
        nodes = torch.empty(n_tok, n_trees, depth + 1, device=L.device, dtype=torch.int32) if want_nodes else A
        block = 128
        _route_fwd[_grid(n_tok, n_trees, block)](L, A, nodes, n_tok, L.stride(0), A.stride(0), n_trees,
                                                 N_NODES=n_nodes, DEPTH=depth, ACT=act, BLOCK=block, STORE_NODES=want_nodes)
        ctx.save_for_backward(L, W)
        ctx.cfg = n_trees, n_nodes, depth, act, temp, ste
        out = A @ W
        if want_nodes:
            ctx.mark_non_differentiable(nodes)
        return out, (nodes if want_nodes else None)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout, _dnodes):
        L, W = ctx.saved_tensors
        n_trees, n_nodes, depth, act, temp, ste = ctx.cfg
        n_tok = L.shape[0]
        dout = dout.contiguous()
        block = 128
        dW = None
        if ctx.needs_input_grad[1]:
            A = torch.zeros_like(L)
            _route_fwd[_grid(n_tok, n_trees, block)](L, A, A, n_tok, L.stride(0), A.stride(0), n_trees,
                                                     N_NODES=n_nodes, DEPTH=depth, ACT=act, BLOCK=block, STORE_NODES=False)
            dW = A.t() @ dout
        dL = None
        if ctx.needs_input_grad[0]:
            G = dout @ W.t()
            dL = torch.zeros_like(L)
            _route_bwd[_grid(n_tok, n_trees, block)](L, G, dL, n_tok, L.stride(0), G.stride(0), dL.stride(0), 1.0 / temp,
                                                     N_NODES=n_nodes, DEPTH=depth, ACT=act, BLOCK=block, STE=ste)
        return dL, dW, None, None, None, None, None, None, None


def route_ste_out(L, W, n_trees, depth, act="gelu", temp=1.0, want_nodes=False, n_nodes=None, ste=True):
    """L: (tokens, trees * nodes) every node's logit, in any float dtype (the kernels work in float32; under bf16
    autocast the (tokens x nodes) buffers stay bf16 and the matmuls run in bf16); W: (trees * nodes, d) output rows. Returns (out, nodes): out is
    (tokens, d), the stateless trees' output with the straight-through routing gradient; nodes is (tokens, trees,
    depth + 1) int32 global node ids along each path if ``want_nodes``, else None. ``act`` is "gelu", "split"
    (the same with no context) or "linear". ``ste=False`` is plain FFF routing: the same forward pass, and hard
    branches get no gradient."""
    if not HAVE_TRITON:
        raise RuntimeError("route_ste_out needs triton")
    code = {"gelu": 0, "split": 0, "linear": 1}[act]
    n_nodes = n_nodes or L.shape[1] // n_trees  # L and W may carry padding columns/rows after the trees
    return _RouteSTE.apply(L, W, n_trees, n_nodes, depth, code, float(temp), want_nodes, bool(ste))


# -------------------------------------------------------------------------------------------------------------- data

TOKENIZER = "google-bert/bert-base-uncased"
CLS, SEP, MASK, PAD = 101, 102, 103, 0


def tokenizer():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download(TOKENIZER, "tokenizer.json"))
    tok.no_padding()
    tok.no_truncation()
    return tok


def prep(a):
    """Wikipedia parquet shards -> one uint16 stream of articles, each followed by [SEP]; validation: the first
    --val-tokens of the last shard."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    tok = tokenizer()
    files = sorted(f for f in list_repo_files("wikimedia/wikipedia", repo_type="dataset") if f.startswith("20231101.en/"))
    lo, hi = (int(v) for v in a.shards.split("-"))
    assert hi < len(files) - 1, "the last shard is the validation shard"
    os.makedirs(a.out, exist_ok=True)

    def stream(name, f, limit):
        pf, n = pq.ParquetFile(hf_hub_download("wikimedia/wikipedia", name, repo_type="dataset")), 0
        for g in range(pf.num_row_groups):
            texts = pf.read_row_group(g, columns=["text"]).column("text").to_pylist()
            for enc in tok.encode_batch(texts, add_special_tokens=False):
                ids = np.asarray(enc.ids + [SEP], dtype=np.uint16)[: limit - n]
                ids.tofile(f)
                n += len(ids)
                if n >= limit:
                    return n
        return n

    with open(os.path.join(a.out, "val.bin"), "wb") as f:
        n_val = stream(files[-1], f, a.val_tokens)
    n_train = 0
    with open(os.path.join(a.out, "train.bin"), "wb") as f:
        for i in range(lo, hi + 1):
            n_train += stream(files[i], f, 1 << 62)
            print(f"  shard {i}: {n_train:,} training tokens", flush=True)
    json.dump({"shards": a.shards, "train_tokens": n_train, "val_tokens": n_val}, open(os.path.join(a.out, "meta.json"), "w"))


def get_batch(data, batch, seq_len, gen, mask_prob=0.15, vocab=30522):
    """Windows of the stream starting with [CLS]; BERT's 80/10/10 masking (random tokens from 999 on: real word pieces)."""
    starts = torch.randint(0, len(data) - seq_len, (batch,), generator=gen).tolist()
    ids = torch.from_numpy(np.stack([data[s : s + seq_len - 1] for s in starts]).astype(np.int64))
    ids = torch.cat([torch.full((batch, 1), CLS), ids], 1)
    chosen = (torch.rand(ids.shape, generator=gen) < mask_prob) & (ids != CLS) & (ids != SEP) & (ids != PAD)
    labels = torch.where(chosen, ids, torch.full_like(ids, -100))
    r = torch.rand(ids.shape, generator=gen)
    inputs = ids.clone()
    inputs[chosen & (r < 0.8)] = MASK
    rand = chosen & (r >= 0.8) & (r < 0.9)
    inputs[rand] = torch.randint(999, vocab, (int(rand.sum()),), generator=gen)
    return inputs, labels


# ------------------------------------------------------------------------------------------------------------- train


@torch.no_grad()
def evaluate(model, data, a, device):
    """Masked-LM loss and accuracy on the same 20 validation batches every time."""
    model.eval()
    g = torch.Generator().manual_seed(1234)
    loss, correct, total = 0.0, 0, 0
    for _ in range(a.eval_batches):
        x, y = get_batch(data, a.batch_size, a.seq_len, g)
        for xm, ym in zip(x.split(a.micro_batch), y.split(a.micro_batch)):
            xm, ym = xm.to(device), ym.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                l, logits = model(xm, ym)
            loss += l.item()
            correct += (logits.argmax(-1) == ym[ym != -100]).sum().item()
            total += int((ym != -100).sum())
    model.train()
    return loss / total, correct / total


def train(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed)
    train_data = np.memmap(os.path.join(a.data, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(a.data, "val.bin"), dtype=np.uint16, mode="r")
    ck = torch.load(a.resume, map_location="cpu", weights_only=False) if a.resume else None
    cfg = config_from(ck["config"]) if ck else GTSConfig(deep_ctx_levels=a.deep_ctx_levels, deep_ctx_state=a.deep_ctx_state)
    model = GTS(cfg).to(device)
    if a.init:  # another model's weights, e.g. GTS3's; parameters it lacks (the trees' context) keep their init
        missing, unexpected = model.load_state_dict(load(a.init).state_dict(), strict=False)
        assert not unexpected, f"--init has parameters this model lacks: {unexpected}"
        print(f"initialised from {a.init}; {len(missing)} parameter tensors new", flush=True)
    if device == "cuda" and a.compile:
        model.backbone.compile_blocks()
    decay = [p for p in model.parameters() if p.ndim >= 2 and not getattr(p, "_no_weight_decay", False)]
    rest = [p for p in model.parameters() if not (p.ndim >= 2 and not getattr(p, "_no_weight_decay", False))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": rest, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.98), eps=1e-6, fused=device == "cuda")
    gen = torch.Generator().manual_seed(a.seed)
    step, lr0, curve = 0, 0.0, []
    if ck:  # a new phase: config, weights, AdamW state, step and sampler carry over; warm from the last rate
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        gen.set_state(ck["generator"])
        step, curve = ck["step"], ck["curve"]
        lr0 = ck["optimizer"]["param_groups"][0]["lr"]
    s0 = step
    print(f"nanoGTS: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, steps {s0} -> {a.steps}", flush=True)

    def lr_at(k):  # linear from lr0 over the warmup, then cosine to 10% of --lr at --steps
        k -= s0
        if k < a.warmup:
            return lr0 + (a.lr - lr0) * (k + 1) / a.warmup
        frac = min(1.0, (k - a.warmup) / max(1, a.steps - s0 - a.warmup))
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))

    def save():
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "config": asdict(model.cfg),
                    "curve": curve, "generator": gen.get_state()}, os.path.join(a.out, "ckpt.pt"))

    os.makedirs(a.out, exist_ok=True)
    t0, run = time.time(), []
    while step < a.steps:
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        x, y = get_batch(train_data, a.batch_size, a.seq_len, gen)
        n = int((y != -100).sum())
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for xm, ym in zip(x.split(a.micro_batch), y.split(a.micro_batch)):  # the mean over the whole batch's labels
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss, _ = model(xm.to(device), ym.to(device))
            (loss / n).backward()
            total += loss.item() / n
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        run.append(total)
        if step % a.log_every == 0:
            print(f"  step {step:6d}  loss {sum(run[-a.log_every:]) / a.log_every:.4f}  lr {lr_at(step):.2e}  "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            vl, acc = evaluate(model, val_data, a, device)
            curve.append({"step": step, "tokens": step * a.batch_size * a.seq_len, "val_loss": vl, "val_masked_acc": acc})
            print(f"step {step:6d}  tokens {step * a.batch_size * a.seq_len / 1e6:8.1f}M  val loss {vl:.4f}  masked acc {acc:.4f}", flush=True)
            save()


EXAMPLES = ["The capital of France is [MASK].", "The [MASK] Ocean is the largest ocean on Earth.",
            "He played the [MASK] in the orchestra for twenty years.", "The film was directed by Steven [MASK]."]


@torch.no_grad()
def sample(a):
    model, tok = load(a.ckpt).eval(), tokenizer()
    for text in EXAMPLES:
        ids = tok.encode(text).ids  # [CLS] ... [SEP]
        top = model(torch.tensor([ids]))[0, ids.index(MASK)].topk(5).indices.tolist()
        print(f"{text}  ->  {', '.join(tok.id_to_token(i) for i in top)}")


# ---------------------------------------------------------------------------------------------------------- selftest


def selftest(kernels=False):
    """The PyTorch fast paths against their definitions, in float64 on a small model."""
    global USE_KERNELS
    USE_KERNELS = False
    torch.manual_seed(0)
    cfg = GTSConfig(d_model=64, n_layer=1, vocab_size=100, bank_trees=8, bank_heads=2, bank_state=4, deep_trees=2,
                    deep_depth=4, ternary_group=32)
    m = Mixer(cfg).double()
    for t in (m.bank.node_in, m.deep.node_in):  # spread the logits so both branches get traffic
        t.data *= 8
    u = torch.randn(2, 12, 64, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(2, 12, dtype=torch.float64)
    mask[1, 9:] = 0

    # deep trees, stateless and stateful (top 3 of 5 levels): the walk Function against the path-weight definition,
    # values and every gradient
    for levels in (0, 3):
        torch.manual_seed(1)
        deep = Deep(GTSConfig(**{**asdict(cfg), "deep_ctx_levels": levels})).double()
        deep.node_in.data *= 8
        out, ref = deep(u, mask), deep.reference(u, mask)
        g = torch.randn_like(out)
        ps = [u] + [p for p in deep.parameters()]
        g1, g2 = torch.autograd.grad(out, ps, g), torch.autograd.grad(ref, ps, g)
        assert torch.allclose(out, ref, atol=1e-10), f"deep trees ({levels} stateful levels): forward"
        assert all(torch.allclose(x, y, atol=1e-9) for x, y in zip(g1, g2)), f"deep trees ({levels} stateful levels): gradient"
    # stateful deep trees: the context against token-at-a-time states, one per node, written only by its visitors
    with torch.no_grad():
        x = deep.local_mix(u, mask)
        al = F.linear(x, deep.q(deep.node_in), deep.node_bias).view(2, 12, deep.n_trees, -1)
        B, C_fwd, C_bwd, dt, a = deep.signals(x, mask)
        n_top = 2**levels - 1
        want = torch.zeros(2, 12, deep.n_trees, n_top, dtype=torch.float64)
        for b in range(2):
            for order, C in ((range(12), C_fwd), (range(11, -1, -1), C_bwd)):
                state = torch.zeros(deep.n_trees, n_top, cfg.deep_ctx_state, dtype=torch.float64)
                for t in order:
                    state = state * torch.exp(a[b, t]).view(-1, 1, 1)
                    want[b, t] += state @ C[b, t]
                    for tree in range(deep.n_trees):
                        node = 0
                        for _ in range(levels):  # the token's own route through the stateful levels
                            state[tree, node] += dt[b, t, tree] * mask[b, t] * al[b, t, tree, node] * B[b, t]
                            node = 2 * node + 1 + int(al[b, t, tree, node] > 0)
        assert torch.allclose(deep.context(x, al, mask), want, atol=1e-9), "stateful deep trees: context"

    # bank: the quadratic form against token-at-a-time states, read then written, forward and backward
    with torch.no_grad():
        bank = m.bank
        x = bank.local_mix(u, mask)
        logit = F.linear(x, bank.q(bank.node_in), bank.node_bias)
        B, C_fwd, C_bwd, dt, a = bank.signals(x, mask)
        H, T = cfg.bank_heads, cfg.bank_trees
        head = torch.arange(T) * H // T
        ctx = torch.zeros_like(logit)
        for b in range(2):
            for order, C in ((range(12), C_fwd), (range(11, -1, -1), C_bwd)):
                state = torch.zeros(T, cfg.bank_state, dtype=torch.float64)
                for t in order:  # arriving at t decays the state by exp(a[t]) in either direction, then read, then write
                    state = state * torch.exp(a[b, t, head]).unsqueeze(-1)
                    ctx[b, t] += (state @ C[b, t].unsqueeze(-1)).squeeze(-1)
                    state = state + (dt[b, t, head] * mask[b, t] * logit[b, t]).unsqueeze(-1) * B[b, t]
        want = ((F.gelu(logit) + ctx) @ bank.q(bank.node_out)) * mask.unsqueeze(-1)
        assert torch.allclose(bank(u, mask), want, atol=1e-9), "bank: context"
    print("selftest passed: deep-tree straight-through walk == path-weight definition (stateless and stateful); "
          "node context == per-node recurrence; bank quadratic form == recurrence")
    USE_KERNELS = True
    if kernels:
        selftest_kernels()


def selftest_kernels():
    """The Triton paths against the PyTorch ones, values and gradients, in float32 (the kernels' precision). Without a
    GPU run as ``TRITON_INTERPRET=1 python nanogts.py selftest --kernels``."""
    global USE_KERNELS
    assert HAVE_TRITON, "needs triton"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda" or os.environ.get("TRITON_INTERPRET") == "1", "no GPU: set TRITON_INTERPRET=1"
    for levels in (0, 2):  # stateless deep trees (the Triton walk), stateful ones (the bank's scan for their context)
        torch.manual_seed(0)
        cfg = GTSConfig(d_model=64, n_layer=2, vocab_size=1100, bank_trees=8, bank_heads=2, bank_state=16, deep_trees=2,
                        deep_depth=4, ternary_group=32, deep_ctx_levels=levels)
        model = GTS(cfg).to(dev)
        ids = torch.randint(999, 1100, (2, 150), device=dev)  # 150 tokens: three chunks of the scan
        ids[1, 120:] = 0
        labels = torch.where(torch.rand(ids.shape, device=dev) < 0.3, ids, torch.full_like(ids, -100))
        labels[ids == 0] = -100
        results = []
        for k in (False, True):
            USE_KERNELS = k
            model.zero_grad()
            loss, logits = model(ids, labels)
            loss.backward()
            results.append((logits.detach(), {n: p.grad.clone() for n, p in model.named_parameters()}))
        USE_KERNELS = True
        (l0, g0), (l1, g1) = results
        rel = lambda x, y: ((x - y).norm() / (y.norm() + 1e-30)).item()  # noqa: E731
        worst = max((rel(g1[n], g0[n]), n) for n in g0)
        print(f"{levels} stateful levels: kernels vs PyTorch: logits {rel(l1, l0):.1e}, worst gradient {worst[0]:.1e} ({worst[1]}) relative")
        assert rel(l1, l0) < 1e-4 and worst[0] < 1e-3, "the Triton paths disagree with the PyTorch ones"
    print("selftest --kernels passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prep")
    q.add_argument("--out", required=True)
    q.add_argument("--shards", required=True, help="training shards, e.g. 0-6")
    q.add_argument("--val-tokens", type=int, default=2_000_000)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, required=True, help="the absolute step to end this phase at")
    t.add_argument("--lr", type=float, required=True)
    t.add_argument("--warmup", type=int, default=1000)
    t.add_argument("--resume", help="continue a run (its config, weights, optimizer and sampler)")
    t.add_argument("--init", help="start from another checkpoint's weights only, e.g. GTS3's, with a fresh optimizer")
    t.add_argument("--deep-ctx-levels", type=int, default=0, help="stateful deep trees: top levels with context")
    t.add_argument("--deep-ctx-state", type=int, default=16)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--micro-batch", type=int, default=64, help="sequences per forward pass; gradients accumulate exactly")
    t.add_argument("--seq-len", type=int, default=512)
    t.add_argument("--eval-every", type=int, default=1000)
    t.add_argument("--eval-batches", type=int, default=20)
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--no-compile", dest="compile", action="store_false")
    s = sub.add_parser("sample")
    s.add_argument("--ckpt", required=True)
    st = sub.add_parser("selftest")
    st.add_argument("--kernels", action="store_true", help="also check the Triton paths against the PyTorch ones")
    a = p.parse_args()
    {"prep": prep, "train": train, "sample": sample, "selftest": lambda a: selftest(a.kernels)}[a.cmd](a)


if __name__ == "__main__":
    main()
