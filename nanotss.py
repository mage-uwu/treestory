"""nanoTSS: the tree state space in one file. A causal language model whose only mixer is a forest of binary trees
whose nodes are its state; its training form, its training kernels (Triton), its CPU inference step (C), and the checks
that tie them together.

The model
---------
token embedding (tied with the output layer, no positions) -> n_layer x [x + TreeSSM(RMSNorm(x))] -> RMSNorm -> logits.
No attention, no MLP, no SSM bank. A TreeSSM layer: a causal depthwise conv (width 3, the identity at init), the
result as per-token 8-bit integers; one ternary projection to [B, C, V, Z, dt]; a forest of T binary trees of depth D.
Every node is a state cell, a small key -> value matrix h (state x value), with its own routing row w_in, bias, output
row w_out and decay rate. A token walks one root-to-leaf path per tree; at each node it visits it

    decays the node's state        h <- exp(-lambda_node * dt) h       (on a visit only: no clock)
    reads it by content            R = C^T h                            (earlier tokens routed here, weighted by <C, B>)
    computes the node's activation s = <x, w_in> + bias + <Z, R>        (memory moves the branch)
    branches                       right if s > 0
    outputs                        gelu(s) w_out (+ U R, U a ternary readout per tree)
    writes                         h <- h + dt B V^T

and touches nothing else: a selective SSM taken to the extreme, the routing as the selection. The route is a coarse
address and <C, B> matches within it; the root sees every token (one Mamba-2 head), a level-k node the tokens that
made the same k decisions. The state grows exponentially with depth, the work per token linearly.

What makes it cheap, carried over from GTS: ternary weights (w_in, w_out, proj, U: absmean codes in {-1, 0, 1} and one
scale per group of 128, recomputed every step from float latent weights, straight-through) on 8-bit activations, so a
visit is an integer sum of +-x over a 2-bit row; and GTS's branch gradient: a branch learns from what the subtree the
token took output below it minus what the other one would have, following the token's own decisions and reading the
states it would have found there. A network when it trains, a ternary tree when it runs.

Three forms of one function
---------------------------
* ``TreeSSM.forward``, training: levels in turn (routing at level k depends only on level-k states, written by
  decisions above it), each level parallel over time. Within a level the tokens are sorted by node, so every node's
  state over time is one segmented linear scan (``segscan``, Mamba-2's chunked scan with the segments' resets as
  masks), linear in length; the branch gradient's counterfactual reads are read-only entries of the same scan. On a
  GPU (Triton) the whole level runs in sorted order: a chunk is one node's tokens, one weight row and one decay rate
  (no divergence) over contiguous memory (coalesced), the per-node products grouped (``group_dot``, ``group_outer``,
  blocks aligned to nodes, as a mixture of experts' dispatch). On a CPU the weight rows are gathered in token order.
  The scan's backward is the same kernels: two more scans and a segmented suffix sum.
* ``TreeSSM.step``: the recurrence a token at a time (PyTorch), the definition inference must match.
* the C step (``C_SOURCE``, ``c_step``): inference on a CPU on the trained representation. Rows as two bitmasks
  (2 bits a weight, a 256-wide row in one cache line), dot products as masked integer add/subtract (AVX-512BW; an
  AVX2 path on int8 codes; a plain path), the trees walked in lockstep, both children's rows prefetched, the chosen
  child's state; optional bfloat16 states; several sequences walked together and split across cores (OpenMP).

* the C training step (``C_TRAIN_SOURCE``, ``CTrainer``): training on a CPU, sparse. The tree forward pass records a
  tape; the backward pass touches only what the forward touched: the visited node rows, the visited nodes' states (an
  adjoint state per node, run backward in time), and for the branch gradient the other subtree's nodes as scalars
  <g, out> only; dense backprop for the rest (proj, U, conv, norms, the output layer). AdamW, lazy on node rows (a row
  moves in the steps that touch it), the touched rows' ternary codes recomputed; sequences across threads. Its
  gradients are autograd's (float64, ``selftest --c``); it trains the model's own tensors in place. On AVX-512 in
  float the rows also live as 2-bit planes (the inference step's representation, refreshed for the touched rows), so
  dot products and row updates are masked adds and subtracts, and a node state's row is one register. The tape of
  previous states (the largest thing the forward writes) is kept in bfloat16 there (``tape_bf16``; gradients within
  ~1e-3 of the float tape's, 10% faster on one thread, 6% on four).
* the branch gradient's depth (``branch_depth`` r, 0 = all the way): a branch at level k compares levels k + 1 .. k + r
  of the subtree the token took and of the other one; in the definition, a node's path weight carries the gradient of
  the branches at most r levels above it. The training form, ``reference`` and the C step agree on it (selftest,
  selftest --c). The other subtrees' walks are most of what training costs beyond inference; capping them is the
  largest lever left (Shakespeare model on 4 threads: 29.4K tokens/s full, 37.2K at r = 2, 40.8K at r = 1). It does
  not pay: on Tiny Shakespeare (5,000 steps, one seed) held-out 1.594 full in 12.3 min, 1.633 at r = 2 in 10.0 min,
  1.653 at r = 1 in 9.1 min, and at equal wall time the full gradient is ahead too (~1.62 at 9 min). Default: full.

``selftest`` checks, in float64: the training form against the recurrence (routes included), its gradients against
``reference`` (the path-weight definition over every node, the counterfactual reads included), and the scan against
a loop. ``selftest --kernels`` checks the Triton kernels against PyTorch through the whole model. ``bench`` checks
the C steps against the recurrence and times them.

    python nanotss.py selftest [--kernels | --c]  # --kernels without a GPU: TRITON_INTERPRET=1 (slow)
    python nanotss.py mqar [--engine torch|c] [--float] [--no-memory] [--depth 0] [--readout level|tree|layer]
    python nanotss.py lm --data input.txt [--engine c|torch]   # a byte-level language model, sampled at the end
    python nanotss.py bench                   # C inference vs the recurrence, then tokens/s

Results so far (one seed each)
------------------------------
Multi-query associative recall (8 pairs of 32 keys; 2 layers, d 64, 4 trees, node state 16 x 16; 2,000 steps of 64
sequences on 2 CPU threads), recall at steps 600 / 800 / 2,000:
    stateless trees                                    0.03 (chance)
    depth 0 (one state per tree: a Mamba-2 head)       - / - / 0.53
    depth 5, float, a gradient on the taken path only  0.20 / 0.83 / 0.991
    depth 5, float, the branch gradient                0.93 / 0.98 / 0.995
    depth 5, ternary + 8-bit + the branch gradient     0.50 / 0.80 / 0.983
Nodes holding a decayed sum of values (no <C, B> matching within a node) reached only 0.20.
Inference (d 256, 6 layers, 4 trees of depth 9, 12.9M parameters, vocab 256; a 2.1 GHz Xeon): plain C 120 us/token,
AVX2 78, 2-bit AVX-512 56 on one core; 64K tokens/s for 4 sequences on 4 cores; 48K tokens/s for 64 sequences on 4
cores with bfloat16 states (11K with float32: 25 MB of state a sequence leaves the cache). The node state, not the
weights, is what a stateful tree moves: 1 KB a visit in float32 against 128 bytes of weights.
Training cost per token is flat in length (PyTorch on 4 CPU threads, 2 layers, d 64: 240 - 300 us/token from 64 to
4,096 tokens). The Triton kernels match PyTorch in the interpreter (logits 2e-7, gradients <= 1.5e-4); their speed on a
GPU is unmeasured. Not yet trained on language.
Training on a CPU, the sparse C step against PyTorch (forward, backward, AdamW; same model, same CPU): 14 - 16x on one
thread, 19 - 23x on four (recall-size model: 97K tokens/s on 4 threads; 12.9M parameters: 347 us/token on one thread,
10.2K tokens/s on four). The rows as 2-bit planes (masked integer and float add/subtract), the node states' rows as
AVX-512 registers forward and backward, the trees and the branch gradient's chains through the other subtrees walked
in lockstep with prefetch (36 independent walks a token at 4 trees of depth 9, their cache misses overlapping), the
node rows' weight gradients summed in row order, the optimizer parallel over rows (at a batch of 2,048 tokens 89% of the node
rows are touched: per token the update is sparse, per batch it is not). MQAR with --engine c on 4 threads: recall
0.968 at 2,000 steps in 30 s (PyTorch: 0.983 in 645 s on 2).
Tiny Shakespeare, byte level (lm, defaults: 1.16M parameters, d 128, 4 layers x 4 trees of depth 7, batch 16 x 256,
5,000 steps = 20.5M tokens), trained by the sparse C step on 4 CPU threads in 12.3 min (27.8K tokens/s): held-out
1.594 nats a byte (2.30 bits), from 2.09 at step 250; train 1.34. (For scale, not a matched comparison: nanoGPT's
character-level Shakespeare, a 10.7M-parameter transformer on a GPU, reaches 1.47.)
Hard routing turns float32 rounding into an occasional different branch (an activation within rounding of zero), as in
GTS: the C kernels agree with each other to rounding, and with PyTorch but for such tokens.
"""

import argparse
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------------------------------- model


@dataclass
class Config:
    vocab_size: int = 256
    d_model: int = 128
    n_layer: int = 2
    trees: int = 4
    depth: int = 5  # levels 0..depth: 2^(depth+1) - 1 nodes per tree, depth + 1 visited
    state: int = 16  # key size of a node's state (B, C)
    value: int = 16  # value size (V, Z, the readout)
    d_conv: int = 3  # causal depthwise, over the current and previous tokens
    temp: float = 1.0
    branch_depth: int = 0  # the branch gradient compares the two subtrees this many levels down (0: all the way)
    memory: bool = True  # False: the same trees, stateless (the ablation)
    readout: str = "tree"  # the reads' readout U: per tree and level, per tree (summed over the path), or "layer"
    ternary: bool = True  # w_in, w_out, proj and U ternary (absmean per group), straight-through to latent weights
    ternary_group: int = 128
    act_bits: int = 8  # the trees' input as per-token integers (0: float)


def group_size(n, g):
    """Largest group size <= g that divides n."""
    if g is None or g >= n:
        return n
    while n % g:
        g -= 1
    return g


def ternary(w, group):
    """BitNet b1.58's absmean quantiser per group of weights along each row, straight-through to the latent weights:
    codes in {-1, 0, 1} times one scale per group, recomputed every step."""
    wg = w.reshape(*w.shape[:-1], -1, group_size(w.shape[-1], group))
    scale = wg.abs().mean(-1, keepdim=True).clamp(min=1e-8)
    wq = ((wg / scale).clamp(-1, 1).round() * scale).reshape(w.shape)
    return wq.detach() + (w - w.detach())


def quantize_activations(x, bits):
    """Per-token absmax integer activations (rounded in float32 at least), straight-through."""
    if not bits:
        return x
    qmax = 2 ** (bits - 1) - 1
    xf = x.to(torch.promote_types(x.dtype, torch.float32))
    scale = qmax / xf.abs().amax(-1, keepdim=True).clamp(min=1e-5)
    xq = ((xf * scale).round().clamp(-qmax - 1, qmax) / scale).to(x.dtype)
    return xq.detach() + (x - x.detach())


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class TreeSSM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, T, D, n, p = cfg.d_model, cfg.trees, cfg.depth, cfg.state, cfg.value
        self.cfg = cfg
        N = 2 ** (D + 1) - 1
        k_in, k_out = 1 / math.sqrt(d), 1 / math.sqrt(T * (D + 1))
        self.w_in = nn.Parameter(torch.empty(T, N, d).uniform_(-k_in, k_in))
        self.bias = nn.Parameter(torch.empty(T, N).uniform_(-k_in, k_in))
        self.w_out = nn.Parameter(torch.empty(T, N, d).uniform_(-k_out, k_out))
        self.conv = nn.Parameter(torch.zeros(cfg.d_conv, d))  # tap j looks j tokens back; the identity at init
        with torch.no_grad():
            self.conv[0] = 1.0
        if cfg.memory:
            self.proj = nn.Linear(d, 2 * n + 2 * p + T, bias=False)  # [B, C, V, Z, dt per tree]
            dt = torch.exp(torch.rand(T) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
            self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
            self.A_log = nn.Parameter(torch.log(torch.empty(T, N).uniform_(1, 16)))  # a decay rate per node
            shape = {"level": (T, D + 1), "tree": (T, 1), "layer": (1, 1)}[cfg.readout]
            self.U = nn.Parameter(torch.empty(*shape, p, d).uniform_(-k_out / math.sqrt(p), k_out / math.sqrt(p)))

    def q(self, w):
        return ternary(w, self.cfg.ternary_group) if self.cfg.ternary else w

    def weights(self):
        """The weights as the forward pass uses them (ternary values when cfg.ternary): w_in, w_out, proj, U."""
        mem = self.cfg.memory
        return (self.q(self.w_in), self.q(self.w_out), self.q(self.proj.weight) if mem else None, self.q(self.U) if mem else None)

    def U_at(self, U, k):
        """The readout of level k's reads, (trees, value, d); shared by levels and trees as cfg.readout says."""
        return U[:, k if self.cfg.readout == "level" else 0].expand(self.cfg.trees, -1, -1)

    def mix(self, u):
        """Causal depthwise conv, x[t] = sum_j conv[j] * u[t - j], then integer activations."""
        L, K = u.shape[1], self.cfg.d_conv
        up = F.pad(u, (0, 0, K - 1, 0))
        x = sum(self.conv[j] * up[:, K - 1 - j : K - 1 - j + L] for j in range(K))
        return quantize_activations(x, self.cfg.act_bits)

    def signals(self, x, proj):
        """B, C (.., state); V, Z (.., value); dt (.., trees)."""
        n, p = self.cfg.state, self.cfg.value
        B, C, V, Z, dt = F.linear(x, proj).split([n, n, p, p, self.cfg.trees], -1)
        return B, C, V, Z, F.softplus(dt + self.dt_bias)

    def forward(self, u):
        """The training form, u: (b, l, d). Levels in turn; each level is one pass over its tokens sorted by node
        (tree, node, sequence, time): grouped products with the node's weight rows (``group_dot``, ``group_outer``),
        and every node's state over time as one segmented scan (``segscan``), linear in length. Going backward each
        branch learns from GTS's straight-through gradient: sign * (on - alt) * sigmoid'(s / temp) / temp, where on is
        what the subtree the token took output below the branch and alt what the other subtree would have, following
        the token's own decisions and reading the states it would have found there (``reference`` is the definition).
        Those are read-only entries of the same pass."""
        cfg = self.cfg
        b, L, _ = u.shape
        T, D, N = cfg.trees, cfg.depth, self.w_in.shape[1]
        dev = u.device
        x = self.mix(u)
        w_in3, w_out3, proj, U = self.weights()
        w_in, w_out, bias = w_in3.reshape(T * N, -1), w_out3.reshape(T * N, -1), self.bias.reshape(-1)
        tree = torch.arange(T, device=dev)
        if cfg.memory:
            B, C, V, Z, dt = self.signals(x, proj)
            A = torch.exp(self.A_log.reshape(-1))
        M = b * L * T
        bi = torch.arange(b, device=dev).view(b, 1, 1).expand(b, L, T).reshape(-1)
        li = torch.arange(L, device=dev).view(1, L, 1).expand(b, L, T).reshape(-1)
        ti = torch.arange(T, device=dev).view(1, 1, T).expand(b, L, T).reshape(-1)

        def level(k, node, queries=()):
            """The tokens at node (b, l, trees) of level k read their node's state, then write it; the tokens at each
            node tensor in queries only read there. Returns, per tensor, the activation s (b, l, trees) and the node
            output (b, l, trees, d). Without the Triton kernels (a CPU): the weight rows gathered in token order and
            only the scan's inputs sorted, which a CPU does faster than the sorted layout a GPU wants."""
            if not use_triton(x):
                return level_gathered(k, node, queries)
            nq = len(queries) + 1
            nodes = torch.cat([node.reshape(-1)] + [q.reshape(-1) for q in queries])
            grp = ti.repeat(nq) * N + nodes  # the node's weight row
            BI, LI = bi.repeat(nq), li.repeat(nq)
            order = torch.argsort((grp * b + BI) * L + LI)  # by node, then sequence, then time
            g, BI, LI = grp[order], BI[order], LI[order]
            s = group_dot(x[BI, LI], w_in, g) + bias[g]
            out = 0
            if cfg.memory:
                member = order < M
                dt_e = dt.reshape(b * L, T)[BI * L + LI, g // N]
                log_decay = -A[g] * dt_e
                zero = torch.zeros((), dtype=dt.dtype, device=dev)
                R = segscan(C[BI, LI], B[BI, LI], torch.where(member[:, None], dt_e[:, None] * V[BI, LI], zero),
                            torch.where(member, log_decay, zero), g * b + BI)  # a segment: a node of one sequence
                R = torch.where(member[:, None], R, R * torch.exp(log_decay)[:, None])  # a query's own arrival decay
                s = s + (R * Z[BI, LI]).sum(-1)
                Uk = self.U_at(U, k)  # entries are tree-major: one matrix product per tree
                counts = torch.bincount(g // N, minlength=T).tolist()
                out = torch.cat([Rt @ Uk[t] for t, Rt in enumerate(torch.split(R, counts))])
            out = out + group_outer(F.gelu(s), w_out, g)
            inv = torch.empty_like(order)
            inv[order] = torch.arange(len(order), device=dev)
            s, out = s[inv], out[inv]  # back to token order
            return [(s[i * M : (i + 1) * M].view(b, L, T), out[i * M : (i + 1) * M].view(b, L, T, -1)) for i in range(nq)]

        def level_gathered(k, node, queries):
            nq = len(queries) + 1
            nodes = torch.cat([node.reshape(-1)] + [q.reshape(-1) for q in queries])
            reads = [None] * nq
            if cfg.memory:
                BI, LI, TI = bi.repeat(nq), li.repeat(nq), ti.repeat(nq)
                member = torch.arange(nq * M, device=dev) < M
                seg = (BI * T + TI) * N + nodes  # a segment: one node of one tree of one sequence
                order = torch.argsort(seg * L + LI)  # contiguous segments, in time within each
                dt_e = dt[BI, LI, TI]
                log_decay = -A[TI * N + nodes] * dt_e
                zero = torch.zeros((), dtype=dt.dtype, device=dev)
                a_e = torch.where(member, log_decay, zero)
                X_e = torch.where(member[:, None], dt_e[:, None] * V[BI, LI], zero)
                Y = segscan(C[BI, LI][order], B[BI, LI][order], X_e[order], a_e[order], seg[order])
                Y = torch.zeros_like(Y).index_copy(0, order, Y)
                Y = torch.where(member[:, None], Y, Y * torch.exp(log_decay)[:, None])  # a query's own arrival decay
                reads = [Y[i * M : (i + 1) * M].view(b, L, T, -1) for i in range(nq)]
            res = []
            for nd, R in zip([node] + list(queries), reads):
                s = torch.einsum("bld,bltd->blt", x, w_in3[tree, nd]) + self.bias[tree, nd]
                out = 0
                if R is not None:
                    s = s + (R * Z[:, :, None]).sum(-1)
                    out = torch.einsum("blhp,hpd->blhd", R, self.U_at(U, k))
                res.append((s, out + F.gelu(s)[..., None] * w_out3[tree, nd]))
            return res

        node = torch.zeros(b, L, T, dtype=torch.long, device=dev)
        path = []  # per level: node, s, output
        for k in range(D + 1):
            s, out = level(k, node)[0]
            path.append((node, s, out))
            node = 2 * node + 1 + (s > 0).long()
        y = sum(out for _, _, out in path).sum(2)
        if not torch.is_grad_enabled() or D == 0:
            return y
        with torch.no_grad():  # the branch gradient's two sides: on (taken, below the branch) and alt (the other child down)
            on = torch.stack([out for _, _, out in path]).flip(0).cumsum(0).flip(0)  # on[k] = sum of levels >= k
            r = cfg.branch_depth or D  # a branch at level k compares levels k + 1 .. k + r of both subtrees
            chains = []  # per branch level k: the other subtree's node on the current level, and its output so far
            for j in range(1, D + 1):
                a, s_prev = path[j - 1][0], path[j - 1][1]
                chains.append([2 * a + 2 - (s_prev > 0).long(), 0])  # the other child of level j - 1's branch
                live = [c for k, c in enumerate(chains) if j - k <= r]
                for c, (s_alt, out_alt) in zip(live, level(j, path[j][0], [c[0] for c in live])[1:]):
                    c[1] = c[1] + out_alt
                    c[0] = 2 * c[0] + 1 + (s_alt > 0).long()
        for k in range(D):
            s_k = path[k][1]
            g = torch.sigmoid(torch.where(s_k > 0, s_k, -s_k) / cfg.temp)
            taken = on[k + 1] - (on[k + r + 1] if k + r + 1 <= D else 0)  # the taken subtree's levels k + 1 .. k + r
            y = y + torch.einsum("blt,bltd->bld", g - g.detach(), taken - chains[k][1])  # zero going forward
        return y

    def reference(self, u):
        """The definition, for selftest: every node of every tree for every token, with its read (what the token would
        find there: the states its visitors wrote before it), times its path weight, the product of the branch values
        above it (the hard step going forward, sigmoid(s / temp) going backward), as GTS's path-weight definition.
        With branch_depth r, a node's path weight carries the gradient of the branches at most r levels above it only."""
        cfg = self.cfg
        b, L, _ = u.shape
        T, D = cfg.trees, cfg.depth
        x = self.mix(u)
        w_in, w_out, proj, U = self.weights()
        if cfg.memory:
            B, C, V, Z, dt = self.signals(x, proj)
            CB = torch.einsum("btn,bsn->bts", C, B)
        node = torch.zeros(b, L, T, dtype=torch.long)
        r = cfg.branch_depth or D
        factors = []  # per level m: its nodes' children's branch values, (b, l, trees, 2^(m+1))
        y = torch.zeros_like(u)
        for k in range(D + 1):
            pi = torch.ones(b, L, T, 2**k, dtype=u.dtype)
            for m, f in enumerate(factors):  # the branches above level k, each expanded to level k's nodes
                f = f.repeat_interleave(2 ** (k - m - 1), dim=3)
                pi = pi * (f if k - m <= r else f.detach())
            ids = torch.arange(2**k - 1, 2 ** (k + 1) - 1)  # level k's nodes
            s = torch.einsum("bld,tnd->bltn", x, w_in[:, ids]) + self.bias[:, ids]
            out = F.gelu(s)[..., None] * w_out[:, ids]
            if cfg.memory:
                R = torch.zeros(b, L, T, len(ids), cfg.value, dtype=u.dtype)
                for t in range(L):  # token t at each node c of the level: the visitors of c before it
                    for c in range(len(ids)):
                        here = (node[:, :t] == ids[c]).to(u.dtype)  # (b, s < t, trees)
                        arrivals = torch.cat([(here * dt[:, :t]).flip(1).cumsum(1).flip(1), torch.zeros_like(dt[:, :1])], 1)
                        lam = torch.exp(self.A_log[:, ids[c]])
                        decay = torch.exp(-lam * (arrivals[:, 1:] + dt[:, t : t + 1]))  # arrivals after s, then t's own
                        w = here * decay * dt[:, :t] * CB[:, t, :t, None]
                        R[:, t, :, c] = torch.einsum("bsh,bsp->bhp", w, V[:, :t])
                s = s + torch.einsum("blhnp,blp->blhn", R, Z)
                out = F.gelu(s)[..., None] * w_out[:, ids] + torch.einsum("blhnp,hpd->blhnd", R, self.U_at(U, k))
            y = y + torch.einsum("blhn,blhnd->bld", pi, out)
            g = (s > 0).to(u.dtype)
            p = torch.sigmoid(s / cfg.temp)
            g = g + (p - p.detach())
            factors.append(torch.stack([1 - g, g], -1).flatten(3))  # children 2i+1, 2i+2
            here = node - (2**k - 1)
            node = 2 * node + 1 + (s.gather(3, here[..., None]).squeeze(-1) > 0).long()
        return y

    def init_state(self, b, device=None, dtype=torch.float32):
        cfg = self.cfg
        N = 2 ** (cfg.depth + 1) - 1
        h = torch.zeros(b, cfg.trees, N, cfg.state, cfg.value, device=device, dtype=dtype) if cfg.memory else None
        return {"h": h, "conv": torch.zeros(b, cfg.d_conv - 1, cfg.d_model, device=device, dtype=dtype)}

    @torch.no_grad()
    def step(self, u, st):
        """The recurrence: one token, u (b, d); st from init_state, updated in place. Returns (b, d)."""
        cfg = self.cfg
        b, T = u.shape[0], cfg.trees
        hist = torch.cat([u[:, None], st["conv"]], 1)  # hist[:, j] = u[t - j]
        x = quantize_activations((self.conv * hist).sum(1), cfg.act_bits)
        st["conv"] = hist[:, :-1]
        w_in, w_out, proj, U = self.weights()
        bi, ti = torch.arange(b)[:, None], torch.arange(T)[None, :]
        node = torch.zeros(b, T, dtype=torch.long)
        y = torch.zeros_like(u)
        if cfg.memory:
            B, C, V, Z, dt = self.signals(x, proj)
            BV = B[:, None, :, None] * V[:, None, None, :]  # (b, 1, state, value)
            h = st["h"]
        for k in range(cfg.depth + 1):
            s = (x[:, None] * w_in[ti, node]).sum(-1) + self.bias[ti, node]
            if cfg.memory:
                ha = h[bi, ti, node] * torch.exp(-torch.exp(self.A_log[ti, node]) * dt)[..., None, None]  # decay on arrival
                R = torch.einsum("bn,bhnp->bhp", C, ha)  # read
                s = s + (R * Z[:, None]).sum(-1)
                y = y + torch.einsum("bhp,hpd->bd", R, self.U_at(U, k))
                h[bi, ti, node] = ha + dt[..., None, None] * BV  # then write
            y = y + torch.einsum("bh,bhd->bd", F.gelu(s), w_out[ti, node])
            node = 2 * node + 1 + (s > 0).long()
        return y


class TSS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.norms = nn.ModuleList([RMSNorm(cfg.d_model) for _ in range(cfg.n_layer)])
        self.layers = nn.ModuleList([TreeSSM(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)

    def forward(self, ids, reference=False):
        x = self.embed(ids)
        for norm, layer in zip(self.norms, self.layers):
            x = x + (layer.reference if reference else layer)(norm(x))
        return self.norm_f(x) @ self.embed.weight.t()

    def init_state(self, b, dtype=torch.float32):
        return [layer.init_state(b, dtype=dtype) for layer in self.layers]

    @torch.no_grad()
    def step(self, ids, states):
        """ids: (b,). One token through every layer's recurrence; returns the next-token logits (b, vocab)."""
        x = self.embed(ids)
        for norm, layer, st in zip(self.norms, self.layers, states):
            x = x + layer.step(norm(x), st)
        return self.norm_f(x) @ self.embed.weight.t()


# ---------------------------------------------------------------------------------------------------------- selftest

# ----------------------------------------------------------------------------------------------------- training kernels


try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

USE_TRITON = True  # False forces PyTorch everywhere


def use_triton(t):
    return USE_TRITON and triton is not None and (t.is_cuda or os.environ.get("TRITON_INTERPRET") == "1")


# ----------------------------------------------------------------------------------------------- the scan, PyTorch


def scan2_torch(Q, K, V, a, seg, V2=None, chunk=64):
    """(Y or None, Y2 or None), as the module docstring defines them. Q, K (M, n) (Q may be None); V (M, p); a (M,);
    seg (M,) ids, each segment contiguous; V2 (M, p) or None. Differentiable by autograd."""
    M, Qn = V.shape[0], chunk
    pad = -M % Qn
    if pad:
        z = lambda t: None if t is None else torch.cat([t, t.new_zeros((pad,) + t.shape[1:])])  # noqa: E731
        Q, K, V, V2, a = z(Q), z(K), z(V), z(V2), z(a)
        seg = torch.cat([seg, -1 - torch.arange(pad, device=seg.device)])  # padding: segments of its own
    nc = (M + pad) // Qn
    K, V, a, seg = K.view(nc, Qn, -1), V.view(nc, Qn, -1), a.view(nc, Qn), seg.view(nc, Qn)
    cs = a.cumsum(1)
    strict = torch.ones(Qn, Qn, dtype=torch.bool, device=a.device).tril(-1)
    same = (seg[:, :, None] == seg[:, None, :]) & strict
    Wd = torch.exp((cs[:, :, None] - cs[:, None, :]).masked_fill(~same, -torch.inf))
    last = seg[:, -1]
    S = torch.einsum("cj,cjn,cjp->cnp", torch.exp(cs[:, -1:] - cs) * (seg == last[:, None]), K, V)
    A = cs[:, -1]
    Sf = S.reshape(nc, -1)
    if nc > Qn:  # the state entering each chunk: the same scan over the chunks
        one = Sf.new_ones(nc, 1)
        G = scan2_torch(one, one, Sf, A, last, chunk=chunk)[0] + Sf
    else:
        same_c = (last[:, None] == last[None, :]) & torch.ones(nc, nc, dtype=torch.bool, device=a.device).tril()
        Gc = A.cumsum(0)
        G = torch.exp((Gc[:, None] - Gc[None, :]).masked_fill(~same_c, -torch.inf)) @ Sf
    H = torch.cat([G.new_zeros(1, G.shape[1]), G[:-1]]).view_as(S)
    carried = (seg == torch.cat([last.new_full((1,), -2), last[:-1]])[:, None]) * torch.exp(cs)  # (nc, Qn)
    Y = Y2 = None
    if Q is not None:
        Qc = Q.view(nc, Qn, -1)
        Y = ((Qc @ K.transpose(1, 2)) * Wd) @ V + carried[..., None] * (Qc @ H)
        Y = Y.reshape(nc * Qn, -1)[:M]
    if V2 is not None:
        V2c = V2.view(nc, Qn, -1)
        Y2 = ((V2c @ V.transpose(1, 2)) * Wd) @ K + carried[..., None] * (V2c @ H.transpose(1, 2))
        Y2 = Y2.reshape(nc * Qn, -1)[:M]
    return Y, Y2


def _scan2(Q, K, V, a, seg, V2=None, chunk=64):
    if use_triton(V):
        return scan2_triton(Q, K, V, a, seg, V2, chunk=chunk)
    return scan2_torch(Q, K, V, a, seg, V2, chunk=chunk)


def _seg_suffix_sum(v, seg):
    """Within each contiguous segment, v summed from each entry to the segment's end."""
    total = v.flip(0).cumsum(0).flip(0)
    end = torch.ones_like(seg, dtype=torch.bool)
    end[:-1] = seg[1:] != seg[:-1]  # the last entry of each segment
    idx = torch.arange(len(seg), device=seg.device)
    nxt = torch.where(end, idx + 1, torch.full_like(idx, len(seg)))
    nxt = nxt.flip(0).cummin(0).values.flip(0)  # the first index past each entry's segment
    after = torch.cat([total, total.new_zeros(1)])[nxt]
    return total - after


class SegScan(torch.autograd.Function):
    """Y = scan(Q=C, K=B, V=X), with the backward pass as two more scans (the module docstring)."""

    @staticmethod
    def forward(ctx, C, B, X, a, seg, chunk):
        Y = _scan2(C, B, X, a, seg, chunk=chunk)[0].to(X.dtype)
        ctx.save_for_backward(C, B, X, a, seg, Y)
        ctx.chunk = chunk
        return Y

    @staticmethod
    def backward(ctx, dY):
        C, B, X, a, seg, Y = ctx.saved_tensors
        dY, q = dY.contiguous(), ctx.chunk
        dC = _scan2(None, B, X, a, seg, V2=dY, chunk=q)[1].to(C.dtype)  # the forward state, read with dY
        # the reverse scan: entry i in reverse order is entry M - 1 - i; its log-decay is the next entry's
        ar = torch.cat([a.new_zeros(1), a.flip(0)[:-1]])
        dXr, dBr = _scan2(B.flip(0), C.flip(0), dY.flip(0), ar, seg.flip(0), V2=X.flip(0), chunk=q)
        dX, dB = dXr.flip(0).to(X.dtype), dBr.flip(0).to(B.dtype)
        da = _seg_suffix_sum((dY * Y).sum(-1) - (X * dX).sum(-1), seg)
        return dC, dB, dX, da.to(a.dtype), None, None


def segscan(C, B, X, a, seg, chunk=64):
    """Y_i = sum over j < i in i's segment of exp(a_{j+1} + .. + a_i) <C_i, B_j> X_j (the module docstring)."""
    return SegScan.apply(C, B, X, a, seg, chunk)


# ----------------------------------------------------------------------------------------------- the scan, Triton

if triton is not None:

    @triton.jit
    def _seg_state_kernel(Kp, Vp, Ap, SEGp, Sp, ATOTp, LASTp, M, n, p,
                          CHUNK: tl.constexpr, BN: tl.constexpr, BP: tl.constexpr, PREC: tl.constexpr):
        """Per chunk: the state its closing segment hands on (BN x BP), its total log-decay and that segment's id."""
        c = tl.program_id(0)
        i = tl.arange(0, CHUNK)
        idx = c * CHUNK + i
        ok = idx < M
        a = tl.load(Ap + idx, mask=ok, other=0.0)
        seg = tl.load(SEGp + idx, mask=ok, other=-1)
        cs = tl.cumsum(a, axis=0)
        total = tl.sum(a, axis=0)
        last = tl.sum(tl.where(i == CHUNK - 1, seg, 0), axis=0)
        w = tl.where(seg == last, tl.exp(total - cs), 0.0)
        kn, pp = tl.arange(0, BN), tl.arange(0, BP)
        Kt = tl.load(Kp + idx[:, None] * n + kn[None, :], mask=ok[:, None] & (kn < n)[None, :], other=0.0)
        Vt = tl.load(Vp + idx[:, None] * p + pp[None, :], mask=ok[:, None] & (pp < p)[None, :], other=0.0)
        S = tl.dot(tl.trans(Kt * w[:, None]), Vt, input_precision=PREC)
        tl.store(Sp + c * BN * BP + kn[:, None] * BP + pp[None, :], S)
        tl.store(ATOTp + c, total)
        tl.store(LASTp + c, last)

    @triton.jit
    def _seg_out_kernel(Qp, Kp, Vp, V2p, Ap, SEGp, Hp, PLASTp, Yp, Y2p, M, n, p,
                        CHUNK: tl.constexpr, BN: tl.constexpr, BP: tl.constexpr,
                        HAS_Q: tl.constexpr, HAS_V2: tl.constexpr, PREC: tl.constexpr):
        """Per chunk: Y = (Q K^T * W) V + carried Q H and/or Y2 = (V2 V^T * W) K + carried V2 H^T, W the masked decays."""
        c = tl.program_id(0)
        i = tl.arange(0, CHUNK)
        idx = c * CHUNK + i
        ok = idx < M
        a = tl.load(Ap + idx, mask=ok, other=0.0)
        seg = tl.load(SEGp + idx, mask=ok, other=-1)
        cs = tl.cumsum(a, axis=0)
        segj = tl.load(SEGp + idx, mask=ok, other=-1)
        keep = (i[:, None] > i[None, :]) & (seg[:, None] == segj[None, :])
        W = tl.exp(tl.where(keep, cs[:, None] - cs[None, :], -float("inf")))
        plast = tl.load(PLASTp + c)
        carried = tl.where(seg == plast, tl.exp(cs), 0.0)
        kn, pp = tl.arange(0, BN), tl.arange(0, BP)
        Kt = tl.load(Kp + idx[:, None] * n + kn[None, :], mask=ok[:, None] & (kn < n)[None, :], other=0.0)
        Vt = tl.load(Vp + idx[:, None] * p + pp[None, :], mask=ok[:, None] & (pp < p)[None, :], other=0.0)
        H = tl.load(Hp + c * BN * BP + kn[:, None] * BP + pp[None, :])
        if HAS_Q:
            Qt = tl.load(Qp + idx[:, None] * n + kn[None, :], mask=ok[:, None] & (kn < n)[None, :], other=0.0)
            Wq = tl.dot(Qt, tl.trans(Kt), input_precision=PREC) * W
            Y = tl.dot(Wq, Vt, input_precision=PREC) + carried[:, None] * tl.dot(Qt, H, input_precision=PREC)
            tl.store(Yp + idx[:, None] * p + pp[None, :], Y, mask=ok[:, None] & (pp < p)[None, :])
        if HAS_V2:
            V2t = tl.load(V2p + idx[:, None] * p + pp[None, :], mask=ok[:, None] & (pp < p)[None, :], other=0.0)
            Wv = tl.dot(V2t, tl.trans(Vt), input_precision=PREC) * W
            Y2 = tl.dot(Wv, Kt, input_precision=PREC) + carried[:, None] * tl.dot(V2t, tl.trans(H), input_precision=PREC)
            tl.store(Y2p + idx[:, None] * n + kn[None, :], Y2, mask=ok[:, None] & (kn < n)[None, :])


def _blk(k):
    return max(16, triton.next_power_of_2(k))


def scan2_triton(Q, K, V, a, seg, V2=None, chunk=64, precision=None):
    """scan2_torch in two Triton kernels (float32), the chunk-to-chunk carry by the same function one level up."""
    dev, f32 = V.device, torch.float32
    M, n, p = V.shape[0], K.shape[1], V.shape[1]
    K, V, a = K.contiguous().to(f32), V.contiguous().to(f32), a.contiguous().to(f32)
    seg = seg.contiguous().to(torch.int64)
    BN, BP = _blk(n), _blk(p)
    prec = precision or ("tf32" if dev.type == "cuda" and torch.backends.cuda.matmul.allow_tf32 else "ieee")
    nc = triton.cdiv(M, chunk)
    S = torch.empty(nc, BN, BP, device=dev, dtype=f32)
    atot = torch.empty(nc, device=dev, dtype=f32)
    last = torch.empty(nc, device=dev, dtype=torch.int64)
    _seg_state_kernel[(nc,)](K, V, a, seg, S, atot, last, M, n, p, CHUNK=chunk, BN=BN, BP=BP, PREC=prec)
    Sf = S[:, :n, :p].reshape(nc, n * p)
    if nc > chunk:
        one = Sf.new_ones(nc, 1)
        G = scan2_triton(one, one, Sf, atot, last, chunk=chunk, precision=precision)[0] + Sf
    else:
        same_c = (last[:, None] == last[None, :]) & torch.ones(nc, nc, dtype=torch.bool, device=dev).tril()
        Gc = atot.cumsum(0)
        G = torch.exp((Gc[:, None] - Gc[None, :]).masked_fill(~same_c, -torch.inf)) @ Sf
    H = torch.zeros(nc, BN, BP, device=dev, dtype=f32)
    H[1:, :n, :p] = G[:-1].view(nc - 1, n, p)
    plast = torch.cat([last.new_full((1,), -2), last[:-1]])
    Y = torch.empty(M, p, device=dev, dtype=f32) if Q is not None else None
    Y2 = torch.empty(M, n, device=dev, dtype=f32) if V2 is not None else None
    Qc = Q.contiguous().to(f32) if Q is not None else K
    V2c = V2.contiguous().to(f32) if V2 is not None else V
    _seg_out_kernel[(nc,)](Qc, K, V, V2c, a, seg, H, plast, Y if Y is not None else V, Y2 if Y2 is not None else K,
                           M, n, p, CHUNK=chunk, BN=BN, BP=BP, HAS_Q=Q is not None, HAS_V2=V2 is not None, PREC=prec)
    return Y, Y2


# ------------------------------------------------------------------------------------------ the grouped products


def group_blocks(groups, block):
    """Blocks of at most ``block`` consecutive entries that never straddle two groups (groups: (E,), ascending).
    Returns (group, start, length) per block."""
    ids, counts = torch.unique_consecutive(groups, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    nblk = (counts + block - 1) // block
    blk_group = torch.repeat_interleave(ids, nblk)
    first = torch.repeat_interleave(torch.cumsum(nblk, 0) - nblk, nblk)
    k = torch.arange(len(blk_group), device=groups.device) - first  # the block's index within its group
    blk_start = torch.repeat_interleave(starts, nblk) + k * block
    blk_len = torch.minimum(torch.repeat_interleave(counts, nblk) - k * block, torch.full_like(k, block))
    return blk_group.contiguous(), blk_start.contiguous(), blk_len.contiguous()


if triton is not None:

    @triton.jit
    def _gdot_kernel(ROWSp, Wp, BGp, BSp, BLp, OUTp, d, BLOCK: tl.constexpr, BD: tl.constexpr):
        """out[r] = <rows[r], W[g]> for the block's rows, its group's weight row loaded once per tile."""
        b = tl.program_id(0)
        g, start, length = tl.load(BGp + b), tl.load(BSp + b), tl.load(BLp + b)
        r = tl.arange(0, BLOCK)
        ok = r < length
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for j in range(0, d, BD):
            col = j + tl.arange(0, BD)
            w = tl.load(Wp + g * d + col, mask=col < d, other=0.0)
            x = tl.load(ROWSp + (start + r)[:, None] * d + col[None, :], mask=ok[:, None] & (col < d)[None, :], other=0.0)
            acc += tl.sum(x * w[None, :], axis=1)
        tl.store(OUTp + start + r, acc, mask=ok)

    @triton.jit
    def _gouter_kernel(COEFp, Wp, BGp, BSp, BLp, OUTp, d, BLOCK: tl.constexpr, BD: tl.constexpr):
        """out[r] = coef[r] * W[g] for the block's rows."""
        b = tl.program_id(0)
        g, start, length = tl.load(BGp + b), tl.load(BSp + b), tl.load(BLp + b)
        r = tl.arange(0, BLOCK)
        ok = r < length
        c = tl.load(COEFp + start + r, mask=ok, other=0.0)
        for j in range(0, d, BD):
            col = j + tl.arange(0, BD)
            w = tl.load(Wp + g * d + col, mask=col < d, other=0.0)
            tl.store(OUTp + (start + r)[:, None] * d + col[None, :], c[:, None] * w[None, :],
                     mask=ok[:, None] & (col < d)[None, :])

    @triton.jit
    def _gacc_kernel(ROWSp, COEFp, BGp, BSp, BLp, DWp, d, BLOCK: tl.constexpr, BD: tl.constexpr):
        """DW[g] += sum over the block's rows of coef[r] * rows[r] (blocks of one group add atomically)."""
        b = tl.program_id(0)
        g, start, length = tl.load(BGp + b), tl.load(BSp + b), tl.load(BLp + b)
        r = tl.arange(0, BLOCK)
        ok = r < length
        c = tl.load(COEFp + start + r, mask=ok, other=0.0)
        for j in range(0, d, BD):
            col = j + tl.arange(0, BD)
            x = tl.load(ROWSp + (start + r)[:, None] * d + col[None, :], mask=ok[:, None] & (col < d)[None, :], other=0.0)
            tl.atomic_add(DWp + g * d + col, tl.sum(c[:, None] * x, axis=0), mask=col < d)


GROUP_BLOCK = 32


def _gdot(rows, W, blocks):
    out = torch.empty(rows.shape[0], device=rows.device, dtype=torch.float32)
    _gdot_kernel[(len(blocks[0]),)](rows, W, *blocks, out, W.shape[1], BLOCK=GROUP_BLOCK, BD=min(128, _blk(W.shape[1])))
    return out


def _gouter(coef, W, blocks):
    out = torch.empty(coef.shape[0], W.shape[1], device=coef.device, dtype=torch.float32)
    _gouter_kernel[(len(blocks[0]),)](coef, W, *blocks, out, W.shape[1], BLOCK=GROUP_BLOCK, BD=min(128, _blk(W.shape[1])))
    return out


def _gacc(rows, coef, blocks, n_groups):
    dW = torch.zeros(n_groups, rows.shape[1], device=rows.device, dtype=torch.float32)
    _gacc_kernel[(len(blocks[0]),)](rows, coef, *blocks, dW, rows.shape[1], BLOCK=GROUP_BLOCK, BD=min(128, _blk(rows.shape[1])))
    return dW


class GroupDot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, W, groups):
        rows, W32 = rows.contiguous().float(), W.contiguous().float()
        blocks = group_blocks(groups, GROUP_BLOCK)
        ctx.save_for_backward(rows, W32, *blocks)
        ctx.w_dtype = W.dtype
        return _gdot(rows, W32, blocks)

    @staticmethod
    def backward(ctx, ds):
        rows, W, *blocks = ctx.saved_tensors
        ds = ds.contiguous().float()
        return _gouter(ds, W, blocks), _gacc(rows, ds, blocks, W.shape[0]).to(ctx.w_dtype), None


class GroupOuter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coef, W, groups):
        coef, W32 = coef.contiguous().float(), W.contiguous().float()
        blocks = group_blocks(groups, GROUP_BLOCK)
        ctx.save_for_backward(coef, W32, *blocks)
        ctx.w_dtype = W.dtype
        return _gouter(coef, W32, blocks)

    @staticmethod
    def backward(ctx, dout):
        coef, W, *blocks = ctx.saved_tensors
        dout = dout.contiguous().float()
        return _gdot(dout, W, blocks), _gacc(dout, coef, blocks, W.shape[0]).to(ctx.w_dtype), None


def group_dot(rows, W, groups):
    """s[e] = <rows[e], W[groups[e]]>, groups ascending (each group's entries contiguous)."""
    if use_triton(rows):
        return GroupDot.apply(rows, W, groups).to(rows.dtype)
    return (rows * W[groups]).sum(-1)


def group_outer(coef, W, groups):
    """out[e] = coef[e] * W[groups[e]], groups ascending."""
    if use_triton(coef):
        return GroupOuter.apply(coef, W, groups).to(coef.dtype)
    return coef[:, None] * W[groups]

# ------------------------------------------------------------------------------------------------------------- selftest


def selftest():
    """In float64: the training form against the token-at-a-time recurrence (logits, routes included), and its
    gradients, the branch gradient's counterfactual side included, against the path-weight definition."""
    torch.manual_seed(0)
    for memory, readout, quant, bd in ((True, "level", True, 0), (True, "tree", True, 0), (True, "layer", False, 0),
                                       (False, "tree", True, 0), (True, "tree", True, 1), (True, "tree", True, 2)):
        cfg = Config(vocab_size=50, d_model=32, n_layer=2, trees=3, depth=3, state=4, value=6, memory=memory,
                     readout=readout, ternary=quant, act_bits=8 if quant else 0, ternary_group=16, branch_depth=bd)
        model = TSS(cfg).double()
        for layer in model.layers:  # spread the logits so both branches get traffic
            layer.w_in.data *= 4
        ids = torch.randint(0, 50, (2, 14))
        par = model(ids)
        states = model.init_state(2, dtype=torch.float64)
        seq = torch.stack([model.step(ids[:, t], states) for t in range(14)], 1)
        err = (par - seq).abs().max().item()
        assert err < 1e-10, f"{cfg}: training form != recurrence ({err:.1e})"
        ref = model(ids, reference=True)
        assert (par - ref).abs().max().item() < 1e-10, f"{cfg}: training form != definition"
        g = torch.randn_like(par)
        ps = list(model.parameters())
        g1, g2 = torch.autograd.grad(par, ps, g), torch.autograd.grad(ref, ps, g)
        names = [n for n, _ in model.named_parameters()]
        bad = [(n, (a - c).abs().max().item()) for n, a, c in zip(names, g1, g2) if not torch.allclose(a, c, atol=1e-9)]
        assert not bad, f"{cfg}: gradients != definition: {bad}"
        missing = [n for n, a in zip(names, g1) if not a.abs().sum() > 0]
        assert not missing, f"{cfg}: no gradient: {missing}"
    # the segmented scan against a loop, chunks small enough that segments span several, gradients included
    torch.manual_seed(1)
    M, n, p = 57, 3, 4
    C, B, X = (torch.randn(M, k, dtype=torch.float64, requires_grad=True) for k in (n, n, p))
    a = (-torch.rand(M, dtype=torch.float64)).requires_grad_()
    seg = torch.repeat_interleave(torch.arange(9), torch.tensor([1, 13, 2, 9, 1, 1, 17, 6, 7]))
    want, h = [], None
    for i in range(M):
        h = torch.zeros(n, p, dtype=torch.float64) if i == 0 or seg[i] != seg[i - 1] else h * torch.exp(a[i])
        want.append(C[i] @ h)  # read the state the earlier entries left, decayed by this entry's arrival
        h = h + B[i][:, None] * X[i][None, :]
    want = torch.stack(want)
    for q in (4, 5, 64):
        got = segscan(C, B, X, a, seg, chunk=q)
        g = torch.randn_like(got)
        assert torch.allclose(got, want, atol=1e-12), f"segscan, chunk {q}: values"
        gw, gg = torch.autograd.grad(want, (C, B, X, a), g, retain_graph=True), torch.autograd.grad(got, (C, B, X, a), g)
        assert all(torch.allclose(x, y, atol=1e-10) for x, y in zip(gw, gg)), f"segscan, chunk {q}: gradients"
    print("selftest passed: training form == recurrence (routes included) and its gradients == the path-weight definition "
          "(ternary and float; stateful and stateless; every readout); the segmented scan == its loop")


def selftest_kernels():
    """The Triton kernels against the PyTorch paths, through the whole model: logits and every parameter's gradient,
    float32. Without a GPU: TRITON_INTERPRET=1 python nanotss.py selftest --kernels (slow)."""
    global USE_TRITON
    assert triton is not None, "needs triton"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    assert dev == "cuda" or os.environ.get("TRITON_INTERPRET") == "1", "no GPU: set TRITON_INTERPRET=1"
    torch.manual_seed(0)
    for readout in ("tree", "level"):
        cfg = Config(vocab_size=64, d_model=64, n_layer=2, trees=3, depth=3, state=8, value=8, readout=readout, ternary_group=32)
        model = TSS(cfg).to(dev)
        for layer in model.layers:
            layer.w_in.data *= 4
        ids = torch.randint(0, 64, (2, 90), device=dev)  # 2 x 90 x 3 = 540 entries a level: several chunks of the scan
        results = []
        for k in (False, True):
            USE_TRITON = k
            model.zero_grad()
            logits = model(ids)
            F.cross_entropy(logits.flatten(0, 1), ids.flatten()).backward()
            results.append((logits.detach(), {n: p.grad.clone() for n, p in model.named_parameters()}))
        USE_TRITON = True
        (l0, g0), (l1, g1) = results
        rel = lambda x, y: ((x - y).norm() / (y.norm() + 1e-30)).item()  # noqa: E731
        worst = max((rel(g1[n], g0[n]), n) for n in g0)
        print(f"readout {readout}: kernels vs PyTorch: logits {rel(l1, l0):.1e}, worst gradient {worst[0]:.1e} ({worst[1]})")
        assert rel(l1, l0) < 1e-4 and worst[0] < 1e-3, "the Triton kernels disagree with the PyTorch paths"
    print("selftest --kernels passed")

# ----------------------------------------------------------------------------------------------------------------- mqar


def mqar_batch(b, n_kv, n_keys, gen):
    """Multi-query associative recall: k1 v1 ... kn vn, then the keys again in a random order; at each query the
    target is its value. Keys are tokens 0..n_keys-1, values n_keys..2 n_keys-1."""
    keys = torch.stack([torch.randperm(n_keys, generator=gen)[:n_kv] for _ in range(b)])
    vals = torch.randint(n_keys, 2 * n_keys, (b, n_kv), generator=gen)
    order = torch.stack([torch.randperm(n_kv, generator=gen) for _ in range(b)])
    ctx = torch.stack([keys, vals], -1).flatten(1)
    queries = keys.gather(1, order)
    ids = torch.cat([ctx, queries], 1)
    labels = torch.full_like(ids, -100)
    labels[:, 2 * n_kv :] = vals.gather(1, order)  # predicted at the query token itself
    return ids, labels


def mqar(a):
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    cfg = Config(vocab_size=2 * a.keys, d_model=a.d_model, n_layer=a.layers, trees=a.trees, depth=a.depth,
                 state=a.state, value=a.value, memory=a.memory, readout=a.readout, ternary=a.quant,
                 act_bits=8 if a.quant else 0)
    model = TSS(cfg)
    trainer = CTrainer(model, threads=a.threads, lr=a.lr) if a.engine == "c" else None
    print(f"tss mqar ({a.engine}): {sum(p.numel() for p in model.parameters()) / 1e3:.0f}K parameters, memory={a.memory}, "
          f"depth {a.depth}, readout {a.readout}, {'ternary' if a.quant else 'float'}, {a.kv} pairs of {a.keys} keys, length {3 * a.kv}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.01)
    gen, t0 = torch.Generator().manual_seed(a.seed), time.time()
    val = mqar_batch(512, a.kv, a.keys, torch.Generator().manual_seed(1234))
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        ids, labels = mqar_batch(a.batch, a.kv, a.keys, gen)
        if trainer:  # the C sparse step: forward, backward and AdamW on the CPU, in place in the model's tensors
            loss = torch.tensor(trainer.step(ids.numpy(), labels.numpy(), lr))
        else:
            for g in opt.param_groups:
                g["lr"] = lr
            loss = F.cross_entropy(model(ids).flatten(0, 1), labels.flatten())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            with torch.no_grad():
                ids, labels = val
                pred = model(ids).argmax(-1)
                sel = labels != -100
                acc = (pred[sel] == labels[sel]).float().mean().item()
            print(f"  step {step:5d}  loss {loss.item():.4f}  recall acc {acc:.3f}  {time.time() - t0:.0f}s", flush=True)
    return acc

# ---------------------------------------------------------------------------------------------------------- C inference


C_SOURCE = r"""// Tree state space: one token through the recurrence, in plain C (what nanotss.py's ``TreeSSM.step`` computes), on the trained
// representation: ternary weights as int8 codes {-1, 0, 1} with one float scale per group, the trees' input as
// per-token int8 activations, so every dot product with a weight row is an integer sum of +-x, one scale per group.
// Per layer and tree it touches depth + 1 rows of w_in and w_out and depth + 1 node states, nothing else.
// Built by c_step: cc -O3 -march=native -ffast-math -fopenmp -shared -fPIC (cached in ~/.cache/nanotss)
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {  // a ternary matrix: rows of d codes, rows x (d / group) scales
    const int8_t *codes;
    const float *scales;
} Ternary;

typedef struct {  // a ternary matrix in 2 bits a weight: per row, per 64 weights, a plus and a minus bitmask
    const uint64_t *bits;
    const float *scales;
} Packed;

typedef struct {
    int vocab, d, n_layer, trees, depth, state, value, d_conv, memory, readout, group;  // readout: 0 level, 1 tree, 2 layer
    const float *embed;   // (vocab, d), tied with the output layer
    const float *norm_f;  // (d)
    // per layer, concatenated over layers:
    const float *norm;    // (d)
    const float *conv;    // (d_conv, d)
    Ternary w_in;         // (trees * nodes, d)
    const float *bias;    // (trees * nodes)
    Ternary w_out;        // (trees * nodes, d)
    Ternary proj;         // (2 * state + 2 * value + trees, d)  [B, C, V, Z, dt]
    const float *dt_bias; // (trees)
    const float *neg_A;   // (trees * nodes): -exp(A_log)
    Ternary U;            // (readouts * value, d), readouts = trees * (depth + 1), trees or 1 by readout
    Packed w_in2, w_out2, proj2, U2;  // the same four matrices in 2 bits (tss_step_batch)
    int state_bf16;       // tss_step_batch: the node states in bfloat16 (half the bytes; rounded at every write)
} Model;

static float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678f)); }
static float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

static float dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static void rmsnorm(float *out, const float *x, const float *w, int d) {
    const float r = 1.0f / sqrtf(dot(x, x, d) / d + 1e-5f);
    for (int i = 0; i < d; i++) out[i] = x[i] * r * w[i];
}

// x -> int8 codes and the factor back to x (quantize_activations, 8 bits)
static float quantize(const float *x, int8_t *xq, int d) {
    float m = 1e-5f;
    for (int i = 0; i < d; i++) m = fmaxf(m, fabsf(x[i]));
    const float scale = 127.0f / m;
    for (int i = 0; i < d; i++) {
        float v = nearbyintf(x[i] * scale);
        xq[i] = (int8_t)fminf(fmaxf(v, -128.0f), 127.0f);
    }
    return 1.0f / scale;
}

// <row r of a ternary matrix, x>: integer sums of +-x per group, one scale each
static float tdot(const Ternary *w, size_t r, const int8_t *xq, float xs, int d, int g) {
    const int8_t *c = w->codes + r * d;
    const float *sc = w->scales + r * (d / g);
    float acc = 0.0f;
    for (int j = 0; j < d; j += g) {
        int32_t s = 0;
        for (int i = j; i < j + g; i++) s += c[i] * xq[i];
        acc += sc[j / g] * (float)s;
    }
    return acc * xs;
}

// y += a * (row r of a ternary matrix)
static void taxpy(float *y, float a, const Ternary *w, size_t r, int d, int g) {
    const int8_t *c = w->codes + r * d;
    const float *sc = w->scales + r * (d / g);
    for (int j = 0; j < d; j += g) {
        const float as = a * sc[j / g];
        for (int i = j; i < j + g; i++) y[i] += as * (float)c[i];
    }
}

static Ternary at(Ternary w, size_t rows, int d, int g) {  // the matrix rows further on (the next layer's)
    w.codes += rows * d;
    w.scales += rows * (d / g);
    return w;
}

// h: (n_layer, trees, nodes, state, value); hist: (n_layer, d_conv - 1, d);
// scratch: >= 5 d + 2 state + (3 + trees) value + trees floats. Writes the next-token logits (vocab).
void tss_step_naive(const Model *m, float *h, float *hist, int token, float *logits, float *scratch) {
    const int d = m->d, T = m->trees, D = m->depth, n = m->state, pv = m->value, K = m->d_conv, g = m->group;
    const int N = (1 << (D + 1)) - 1, P = 2 * n + 2 * pv + T;
    const int nU = m->readout == 0 ? T * (D + 1) : m->readout == 1 ? T : 1;
    float *x = scratch, *u = x + d, *mx = u + d, *y = mx + d, *p = y + d;  // p: B, C, V, Z, dt
    float *R = p + P, *Racc = R + pv;  // Racc: the reads summed, per tree or for the layer
    int8_t *xq = (int8_t *)(Racc + T * pv);
    memcpy(x, m->embed + (size_t)token * d, d * sizeof(float));
    for (int l = 0; l < m->n_layer; l++) {
        const float *conv = m->conv + (size_t)l * K * d;
        const Ternary w_in = at(m->w_in, (size_t)l * T * N, d, g), w_out = at(m->w_out, (size_t)l * T * N, d, g);
        const Ternary proj = at(m->proj, (size_t)l * P, d, g), U = at(m->U, (size_t)l * nU * pv, d, g);
        const float *bias = m->bias + (size_t)l * T * N, *neg_A = m->neg_A + (size_t)l * T * N;
        float *hl = h + (size_t)l * T * N * n * pv, *hb = hist + (size_t)l * (K - 1) * d;
        rmsnorm(u, x, m->norm + (size_t)l * d, d);
        for (int i = 0; i < d; i++) {  // causal depthwise conv, then shift the history
            float s = conv[i] * u[i];
            for (int j = 1; j < K; j++) s += conv[j * d + i] * hb[(j - 1) * d + i];
            mx[i] = s;
        }
        if (K > 1) {
            memmove(hb + d, hb, (size_t)(K - 2) * d * sizeof(float));
            memcpy(hb, u, d * sizeof(float));
        }
        const float xs = quantize(mx, xq, d);
        memset(y, 0, d * sizeof(float));
        float *B = p, *C = p + n, *V = p + 2 * n, *Z = p + 2 * n + pv, *dt = p + 2 * n + 2 * pv;
        if (m->memory) {
            for (int r = 0; r < P; r++) p[r] = tdot(&proj, r, xq, xs, d, g);
            for (int t = 0; t < T; t++) dt[t] = softplus(dt[t] + m->dt_bias[l * T + t]);
            if (m->readout) memset(Racc, 0, (size_t)T * pv * sizeof(float));
        }
        for (int t = 0; t < T; t++) {
            int a = 0;
            for (int k = 0; k <= D; k++) {
                const size_t row = (size_t)t * N + a;
                float s = tdot(&w_in, row, xq, xs, d, g) + bias[row];
                if (m->memory) {
                    float *ha = hl + row * n * pv;
                    const float decay = expf(neg_A[row] * dt[t]), w = dt[t];
                    memset(R, 0, pv * sizeof(float));
                    for (int j = 0; j < n; j++) {  // decay on arrival, read C^T h, then write B V^T
                        float *hj = ha + j * pv;
                        const float c = C[j], bw = B[j] * w;
                        for (int q = 0; q < pv; q++) {
                            const float v = hj[q] * decay;
                            R[q] += c * v;
                            hj[q] = v + bw * V[q];
                        }
                    }
                    s += dot(R, Z, pv);
                    if (m->readout == 0) {
                        for (int q = 0; q < pv; q++) taxpy(y, R[q], &U, ((size_t)t * (D + 1) + k) * pv + q, d, g);
                    } else {
                        float *acc = Racc + (m->readout == 1 ? t * pv : 0);
                        for (int q = 0; q < pv; q++) acc[q] += R[q];
                    }
                }
                taxpy(y, gelu(s), &w_out, row, d, g);
                a = 2 * a + 1 + (s > 0.0f);  // the branch
            }
        }
        if (m->memory && m->readout)  // one readout per tree, or one for the layer
            for (int t = 0; t < (m->readout == 1 ? T : 1); t++)
                for (int q = 0; q < pv; q++) taxpy(y, Racc[t * pv + q], &U, (size_t)t * pv + q, d, g);
        for (int i = 0; i < d; i++) x[i] += y[i];
    }
    rmsnorm(u, x, m->norm_f, d);
    for (int v = 0; v < m->vocab; v++) logits[v] = dot(m->embed + (size_t)v * d, u, d);
}


// ------------------------------------------------------------------------------------------------- the fast step
// The same function, arranged for a CPU: trees walked in lockstep, level by level, so their dependent fetches overlap;
// both children of every node prefetched while it is being computed; ternary dot products as sign-and-add on int8
// lanes (_mm256_sign_epi8 is x * code exactly for codes in {-1, 0, 1}); the node state's decay, read and write fused in
// one pass of 8-float vectors. Needs value % 8 == 0 and group % 32 == 0 (else tss_step_naive).
#ifdef __AVX2__
#include <immintrin.h>

static inline int32_t hsum_epi32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4e));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xb1));
    return _mm_cvtsi128_si32(s);
}

static inline float hsum_ps(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

// <ternary row, int8 x>: per group, sum of sign(code) * x in int32, times the group's scale
static inline float tdot_fast(const int8_t *restrict c, const float *restrict sc, const int8_t *restrict xq, float xs,
                              int d, int g) {
    const __m256i ones8 = _mm256_set1_epi8(1), ones16 = _mm256_set1_epi16(1);
    float acc = 0.0f;
    for (int j = 0; j < d; j += g) {
        __m256i s32 = _mm256_setzero_si256();
        for (int i = j; i < j + g; i += 32) {
            const __m256i x = _mm256_load_si256((const __m256i *)(xq + i));
            const __m256i w = _mm256_loadu_si256((const __m256i *)(c + i));
            const __m256i p16 = _mm256_maddubs_epi16(ones8, _mm256_sign_epi8(x, w));  // pairs, |.| <= 254
            s32 = _mm256_add_epi32(s32, _mm256_madd_epi16(p16, ones16));
        }
        acc += sc[j / g] * (float)hsum_epi32(s32);
    }
    return acc * xs;
}

// y += a * ternary row
static inline void taxpy_fast(float *restrict y, float a, const int8_t *restrict c, const float *restrict sc, int d, int g) {
    for (int j = 0; j < d; j += g) {
        const __m256 as = _mm256_set1_ps(a * sc[j / g]);
        for (int i = j; i < j + g; i += 8) {
            const __m256 w = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(c + i))));
            _mm256_store_ps(y + i, _mm256_fmadd_ps(as, w, _mm256_load_ps(y + i)));
        }
    }
}

static inline void prefetch_row(const Ternary *w, size_t r, int d, int g) {
    const char *c = (const char *)(w->codes + r * d);
    for (int i = 0; i < d; i += 64) __builtin_prefetch(c + i);
    __builtin_prefetch(w->scales + r * (d / g));
}

#define MAX_TREES 64
void tss_step(const Model *m, float *h, float *hist, int token, float *logits, float *scratch) {
    const int d = m->d, T = m->trees, D = m->depth, n = m->state, pv = m->value, K = m->d_conv, g = m->group;
    const int N = (1 << (D + 1)) - 1, P = 2 * n + 2 * pv + T;
    const int nU = m->readout == 0 ? T * (D + 1) : m->readout == 1 ? T : 1;
    if (pv % 8 || g % 32 || d % g || T > MAX_TREES) {
        tss_step_naive(m, h, hist, token, logits, scratch);
        return;
    }
    // scratch, 32-byte aligned pieces: x, u, mx, y (d each), p (P), R (T * pv), Racc (T * pv), xq (d bytes)
    float *x = (float *)(((uintptr_t)scratch + 31) & ~(uintptr_t)31);
    const int dA = (d + 7) & ~7, PA = (P + 7) & ~7, RA = (T * pv + 7) & ~7;
    float *u = x + dA, *mx = u + dA, *y = mx + dA, *p = y + dA, *R = p + PA, *Racc = R + RA;
    int8_t *xq = (int8_t *)(Racc + RA);
    float *B = p, *C = p + n, *V = p + 2 * n, *Z = p + 2 * n + pv, *dt = p + 2 * n + 2 * pv;
    int node[MAX_TREES];
    float s[MAX_TREES];
    memcpy(x, m->embed + (size_t)token * d, d * sizeof(float));
    for (int l = 0; l < m->n_layer; l++) {
        const float *conv = m->conv + (size_t)l * K * d;
        const Ternary w_in = at(m->w_in, (size_t)l * T * N, d, g), w_out = at(m->w_out, (size_t)l * T * N, d, g);
        const Ternary proj = at(m->proj, (size_t)l * P, d, g), U = at(m->U, (size_t)l * nU * pv, d, g);
        const float *bias = m->bias + (size_t)l * T * N, *neg_A = m->neg_A + (size_t)l * T * N;
        float *hl = h + (size_t)l * T * N * n * pv, *hb = hist + (size_t)l * (K - 1) * d;
        for (int t = 0; t < T; t++) {  // the roots: needed first, fetched while the dense part runs
            prefetch_row(&w_in, (size_t)t * N, d, g);
            __builtin_prefetch(hl + (size_t)t * N * n * pv);
        }
        rmsnorm(u, x, m->norm + (size_t)l * d, d);
        for (int i = 0; i < d; i++) {
            float acc = conv[i] * u[i];
            for (int j = 1; j < K; j++) acc += conv[j * d + i] * hb[(j - 1) * d + i];
            mx[i] = acc;
        }
        if (K > 1) {
            memmove(hb + d, hb, (size_t)(K - 2) * d * sizeof(float));
            memcpy(hb, u, d * sizeof(float));
        }
        const float xs = quantize(mx, xq, d);
        memset(y, 0, d * sizeof(float));
        if (m->memory) {
            for (int r = 0; r < P; r++) p[r] = tdot_fast(proj.codes + (size_t)r * d, proj.scales + (size_t)r * (d / g), xq, xs, d, g);
            for (int t = 0; t < T; t++) dt[t] = softplus(dt[t] + m->dt_bias[l * T + t]);
            memset(Racc, 0, (size_t)T * pv * sizeof(float));
        }
        for (int t = 0; t < T; t++) node[t] = 0;
        for (int k = 0; k <= D; k++) {
            for (int t = 0; t < T; t++) {  // every tree's node on this level: independent, so their latencies overlap
                const size_t row = (size_t)t * N + node[t];
                if (k < D) {  // both children, before knowing which
                    const size_t c0 = (size_t)t * N + 2 * node[t] + 1;
                    prefetch_row(&w_in, c0, d, g);
                    prefetch_row(&w_in, c0 + 1, d, g);
                    prefetch_row(&w_out, c0, d, g);
                    prefetch_row(&w_out, c0 + 1, d, g);
                    if (m->memory) {
                        const char *h0 = (const char *)(hl + c0 * n * pv);
                        for (int i = 0; i < 2 * n * pv * 4; i += 64) __builtin_prefetch(h0 + i);
                    }
                }
                float st = tdot_fast(w_in.codes + row * d, w_in.scales + row * (d / g), xq, xs, d, g) + bias[row];
                if (m->memory) {
                    float *restrict ha = hl + row * n * pv;
                    const __m256 decay = _mm256_set1_ps(expf(neg_A[row] * dt[t]));
                    float *Rt = R + t * pv;
                    for (int q = 0; q < pv; q += 8) {  // decay on arrival, read C^T h, write B V^T, 8 values at a time
                        __m256 r = _mm256_setzero_ps();
                        const __m256 v8 = _mm256_loadu_ps(V + q);
                        for (int j = 0; j < n; j++) {
                            float *hj = ha + j * pv + q;
                            const __m256 hv = _mm256_mul_ps(_mm256_loadu_ps(hj), decay);
                            r = _mm256_fmadd_ps(_mm256_set1_ps(C[j]), hv, r);
                            _mm256_storeu_ps(hj, _mm256_fmadd_ps(_mm256_set1_ps(B[j] * dt[t]), v8, hv));
                        }
                        _mm256_storeu_ps(Rt + q, r);
                    }
                    st += dot(Rt, Z, pv);
                    if (m->readout == 0) {
                        for (int q = 0; q < pv; q++) {
                            const size_t ur = ((size_t)t * (D + 1) + k) * pv + q;
                            taxpy_fast(y, Rt[q], U.codes + ur * d, U.scales + ur * (d / g), d, g);
                        }
                    } else {
                        float *acc = Racc + (m->readout == 1 ? t * pv : 0);
                        for (int q = 0; q < pv; q++) acc[q] += Rt[q];
                    }
                }
                s[t] = st;
            }
            for (int t = 0; t < T; t++) {
                const size_t row = (size_t)t * N + node[t];
                taxpy_fast(y, gelu(s[t]), w_out.codes + row * d, w_out.scales + row * (d / g), d, g);
                node[t] = 2 * node[t] + 1 + (s[t] > 0.0f);  // the branch
            }
        }
        if (m->memory && m->readout)
            for (int t = 0; t < (m->readout == 1 ? T : 1); t++)
                for (int q = 0; q < pv; q++) {
                    const size_t ur = (size_t)t * pv + q;
                    taxpy_fast(y, Racc[t * pv + q], U.codes + ur * d, U.scales + ur * (d / g), d, g);
                }
        for (int i = 0; i < d; i++) x[i] += y[i];
    }
    rmsnorm(u, x, m->norm_f, d);
    for (int v = 0; v < m->vocab; v++) logits[v] = dot(m->embed + (size_t)v * d, u, d);
}
#else
void tss_step(const Model *m, float *h, float *hist, int token, float *logits, float *scratch) {
    tss_step_naive(m, h, hist, token, logits, scratch);
}
#endif

// n tokens in a row (teacher-forced), the logits of the last; what a benchmark should time.
void tss_run(const Model *m, float *h, float *hist, const int *tokens, int n, float *logits, float *scratch) {
    for (int i = 0; i < n; i++) tss_step(m, h, hist, tokens[i], logits, scratch);
}

void tss_run_naive(const Model *m, float *h, float *hist, const int *tokens, int n, float *logits, float *scratch) {
    for (int i = 0; i < n; i++) tss_step_naive(m, h, hist, tokens[i], logits, scratch);
}

// --------------------------------------------------------------------------------- the 2-bit step (AVX-512BW)
// Bytes, not arithmetic, set the speed of a visit: a node is a w_in row, a w_out row and its state, fetched from
// wherever the walk lands. Here a 256-wide row is 64 bytes, one cache line (two bitmasks per 64 weights), and its
// dot product with int8 x is two masked byte operations per 64 weights; y += a * row is a masked add and a masked
// subtract per 16 floats. The node states may be bfloat16. tss_step_batch walks several sequences at once, so that
// more independent fetches are in flight. Needs d % 64 == 0, group % 64 == 0, value % 16 == 0.
#if defined(__AVX512BW__) && defined(__AVX512F__)

static inline Packed pat(Packed w, size_t rows, int d, int g) {
    w.bits += rows * (size_t)(d / 32);
    w.scales += rows * (size_t)(d / g);
    return w;
}

static inline float pdot(const uint64_t *restrict b, const float *restrict sc, const int8_t *restrict xq, float xs, int d, int g) {
    const __m512i ones8 = _mm512_set1_epi8(1), ones16 = _mm512_set1_epi16(1);
    float acc = 0.0f;
    for (int j = 0; j < d; j += g) {
        __m512i s32 = _mm512_setzero_si512();
        for (int i = j; i < j + g; i += 64) {
            const __m512i x = _mm512_load_si512((const void *)(xq + i));
            const uint64_t *bb = b + (i / 64) * 2;
            __m512i v = _mm512_maskz_mov_epi8(bb[0], x);  // +x where the code is 1
            v = _mm512_mask_sub_epi8(v, bb[1], v, x);     // -x where it is -1
            s32 = _mm512_add_epi32(s32, _mm512_madd_epi16(_mm512_maddubs_epi16(ones8, v), ones16));
        }
        acc += sc[j / g] * (float)_mm512_reduce_add_epi32(s32);
    }
    return acc * xs;
}

static inline void paxpy(float *restrict y, float a, const uint64_t *restrict b, const float *restrict sc, int d, int g) {
    for (int j = 0; j < d; j += g) {
        const __m512 as = _mm512_set1_ps(a * sc[j / g]);
        for (int i = j; i < j + g; i += 64) {
            const uint64_t P = b[(i / 64) * 2], Nm = b[(i / 64) * 2 + 1];
            for (int q = 0; q < 4; q++) {
                float *yq = y + i + 16 * q;
                __m512 yv = _mm512_load_ps(yq);
                yv = _mm512_mask_add_ps(yv, (__mmask16)(P >> (16 * q)), yv, as);
                yv = _mm512_mask_sub_ps(yv, (__mmask16)(Nm >> (16 * q)), yv, as);
                _mm512_store_ps(yq, yv);
            }
        }
    }
}

// quantize_activations on 16 lanes: absmax, x * 127 / max rounded half to even (the default rounding mode)
static inline float quantize16(const float *restrict x, int8_t *restrict xq, int d) {
    __m512 mx = _mm512_set1_ps(1e-5f);
    for (int i = 0; i < d; i += 16) mx = _mm512_max_ps(mx, _mm512_abs_ps(_mm512_loadu_ps(x + i)));
    const float scale = 127.0f / _mm512_reduce_max_ps(mx);
    const __m512 sv = _mm512_set1_ps(scale);
    for (int i = 0; i < d; i += 16)
        _mm_storeu_si128((__m128i *)(xq + i), _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(x + i), sv))));
    return 1.0f / scale;
}

// y += sum_r a[r] * (ternary row r), rows r0.., 16 floats of y at a time held in a register across all rows
static inline void paxpy_rows(float *restrict y, const float *restrict a, int rows, const uint64_t *restrict b,
                              const float *restrict sc, int d, int g) {
    const int rw = d / 32, gs = d / g;
    for (int i = 0; i < d; i += 16) {
        __m512 yv = _mm512_load_ps(y + i);
        const int chunk = i / 64, shift = i % 64;
        for (int r = 0; r < rows; r++) {
            const __m512 as = _mm512_set1_ps(a[r] * sc[(size_t)r * gs + i / g]);
            const uint64_t *bb = b + (size_t)r * rw + chunk * 2;
            yv = _mm512_mask_add_ps(yv, (__mmask16)(bb[0] >> shift), yv, as);
            yv = _mm512_mask_sub_ps(yv, (__mmask16)(bb[1] >> shift), yv, as);
        }
        _mm512_store_ps(y + i, yv);
    }
}

// out[v] = <E[v], u> for every row of a float matrix, 16 rows at a time (the output layer)
static inline void matvec16(float *restrict out, const float *restrict E, const float *restrict u, int rows, int d) {
    int v = 0;
    for (; v + 4 <= rows; v += 4) {
        __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        const float *e = E + (size_t)v * d;
        for (int i = 0; i < d; i += 16) {
            const __m512 uv = _mm512_loadu_ps(u + i);
            a0 = _mm512_fmadd_ps(_mm512_loadu_ps(e + i), uv, a0);
            a1 = _mm512_fmadd_ps(_mm512_loadu_ps(e + d + i), uv, a1);
            a2 = _mm512_fmadd_ps(_mm512_loadu_ps(e + 2 * d + i), uv, a2);
            a3 = _mm512_fmadd_ps(_mm512_loadu_ps(e + 3 * d + i), uv, a3);
        }
        out[v] = _mm512_reduce_add_ps(a0), out[v + 1] = _mm512_reduce_add_ps(a1);
        out[v + 2] = _mm512_reduce_add_ps(a2), out[v + 3] = _mm512_reduce_add_ps(a3);
    }
    for (; v < rows; v++) out[v] = dot(E + (size_t)v * d, u, d);
}

static inline void prefetch_lines(const void *p, size_t bytes) {
    for (size_t i = 0; i < bytes; i += 64) __builtin_prefetch((const char *)p + i);
}

static inline __m512 load16(const void *h, int bf16) {
    if (!bf16) return _mm512_loadu_ps(h);
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)h)), 16));
}

static inline void store16(void *h, __m512 v, int bf16) {
    if (!bf16) {
        _mm512_storeu_ps(h, v);
        return;
    }
    const __m512i u = _mm512_castps_si512(v);  // round to nearest even
    const __m512i r = _mm512_add_epi32(u, _mm512_add_epi32(_mm512_set1_epi32(0x7fff), _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1))));
    _mm256_storeu_si256((__m256i *)h, _mm512_cvtepi32_epi16(_mm512_srli_epi32(r, 16)));
}

// B sequences, one token each. h: B sequences' states one after another (n_layer, trees, nodes, state, value, in
// float32 or bfloat16), hist: B x (n_layer, d_conv - 1, d), logits: B x vocab, scratch: >= B * (5 d + 2 state +
// 2 (trees + 1) value + trees + 64) + 16 floats. Returns 0, or -1 if the shapes do not fit (or B > 64).
#define MAX_SEQ_TREES 512
int tss_step_batch(const Model *m, int nb, void *h, float *hist, const int *tokens, float *logits, float *scratch) {
    const int d = m->d, T = m->trees, D = m->depth, n = m->state, pv = m->value, K = m->d_conv, g = m->group;
    const int N = (1 << (D + 1)) - 1, P = 2 * n + 2 * pv + T, bf = m->state_bf16;
    const int nU = m->readout == 0 ? T * (D + 1) : m->readout == 1 ? T : 1;
    if (d % 64 || g % 64 || d % g || pv % 16 || (m->memory == 0) || nb * T > MAX_SEQ_TREES) return -1;
    const size_t esz = bf ? 2 : 4, seq_h = (size_t)m->n_layer * T * N * n * pv * esz;
    const int dA = (d + 15) & ~15, PA = (P + 15) & ~15, RA = (T * pv + 15) & ~15;
    const size_t stride = (4 * dA + PA + 2 * RA + dA / 4 + 16 + 15) & ~(size_t)15;  // whole cache lines per sequence
    float *base = (float *)(((uintptr_t)scratch + 63) & ~(uintptr_t)63);
    int node[MAX_SEQ_TREES];
    float sv[MAX_SEQ_TREES], xsv[64];
    if (nb > 64) return -1;
#define SEQ(b) (base + (size_t)(b) * stride)
#define X(b) SEQ(b)
#define U_(b) (SEQ(b) + dA)
#define MX(b) (SEQ(b) + 2 * dA)
#define Y(b) (SEQ(b) + 3 * dA)
#define PP(b) (SEQ(b) + 4 * dA)
#define RR(b) (PP(b) + PA)
#define RACC(b) (RR(b) + RA)
#define XQ(b) ((int8_t *)(RACC(b) + RA))
    for (int b = 0; b < nb; b++) memcpy(X(b), m->embed + (size_t)tokens[b] * d, d * sizeof(float));
    for (int l = 0; l < m->n_layer; l++) {
        const float *conv = m->conv + (size_t)l * K * d;
        const Packed w_in = pat(m->w_in2, (size_t)l * T * N, d, g), w_out = pat(m->w_out2, (size_t)l * T * N, d, g);
        const Packed proj = pat(m->proj2, (size_t)l * P, d, g), U = pat(m->U2, (size_t)l * nU * pv, d, g);
        const float *bias = m->bias + (size_t)l * T * N, *neg_A = m->neg_A + (size_t)l * T * N;
        const size_t row_u64 = d / 32, layer_h = (size_t)l * T * N * n * pv * esz;
        for (int b = 0; b < nb; b++)
            for (int t = 0; t < T; t++) {
                prefetch_lines(w_in.bits + (size_t)t * N * row_u64, d / 4);
                prefetch_lines((char *)h + b * seq_h + layer_h + (size_t)t * N * n * pv * esz, (size_t)n * pv * esz);
            }
        for (int b = 0; b < nb; b++) {  // the dense part: norm, conv, int8, the signals
            float *x = X(b), *u = U_(b), *mx = MX(b), *p = PP(b), *hb = hist + ((size_t)b * m->n_layer + l) * (K - 1) * d;
            rmsnorm(u, x, m->norm + (size_t)l * d, d);
            for (int i = 0; i < d; i++) mx[i] = conv[i] * u[i];
            for (int j = 1; j < K; j++)
                for (int i = 0; i < d; i++) mx[i] += conv[j * d + i] * hb[(j - 1) * d + i];
            if (K > 1) {
                memmove(hb + d, hb, (size_t)(K - 2) * d * sizeof(float));
                memcpy(hb, u, d * sizeof(float));
            }
            xsv[b] = quantize16(mx, XQ(b), d);
            memset(Y(b), 0, d * sizeof(float));
            for (int r = 0; r < P; r++) p[r] = pdot(proj.bits + (size_t)r * row_u64, proj.scales + (size_t)r * (d / g), XQ(b), xsv[b], d, g);
            for (int t = 0; t < T; t++) p[2 * n + 2 * pv + t] = softplus(p[2 * n + 2 * pv + t] + m->dt_bias[l * T + t]);
            memset(RACC(b), 0, (size_t)T * pv * sizeof(float));
        }
        for (int i = 0; i < nb * T; i++) node[i] = 0;
        for (int k = 0; k <= D; k++) {
            for (int b = 0; b < nb; b++) {  // every sequence's every tree on this level: independent fetches in flight
                const float *p = PP(b), *B = p, *C = p + n, *V = p + 2 * n, *Z = p + 2 * n + pv, *dt = p + 2 * n + 2 * pv;
                char *hl = (char *)h + b * seq_h + layer_h;
                for (int t = 0; t < T; t++) {
                    const size_t row = (size_t)t * N + node[b * T + t];
                    if (k < D) {  // both children's rows (a cache line each), before knowing which
                        const size_t c0 = (size_t)t * N + 2 * node[b * T + t] + 1;
                        prefetch_lines(w_in.bits + c0 * row_u64, d / 2);
                        prefetch_lines(w_out.bits + c0 * row_u64, d / 2);
                    }
                    float st = pdot(w_in.bits + row * row_u64, w_in.scales + row * (d / g), XQ(b), xsv[b], d, g) + bias[row];
                    char *ha = hl + row * n * pv * esz;
                    const __m512 decay = _mm512_set1_ps(expf(neg_A[row] * dt[t]));
                    float *Rt = RR(b) + t * pv;
                    for (int q = 0; q < pv; q += 16) {  // decay on arrival, read C^T h, write B V^T
                        __m512 r = _mm512_setzero_ps();
                        const __m512 v16 = _mm512_loadu_ps(V + q);
                        for (int j = 0; j < n; j++) {
                            char *hj = ha + ((size_t)j * pv + q) * esz;
                            const __m512 hv = _mm512_mul_ps(load16(hj, bf), decay);
                            r = _mm512_fmadd_ps(_mm512_set1_ps(C[j]), hv, r);
                            store16(hj, _mm512_fmadd_ps(_mm512_set1_ps(B[j] * dt[t]), v16, hv), bf);
                        }
                        _mm512_storeu_ps(Rt + q, r);
                    }
                    st += dot(Rt, Z, pv);
                    if (m->readout == 0) {
                        for (int q = 0; q < pv; q++) {
                            const size_t ur = ((size_t)t * (D + 1) + k) * pv + q;
                            paxpy(Y(b), Rt[q], U.bits + ur * row_u64, U.scales + ur * (d / g), d, g);
                        }
                    } else {
                        float *acc = RACC(b) + (m->readout == 1 ? t * pv : 0);
                        for (int q = 0; q < pv; q++) acc[q] += Rt[q];
                    }
                    sv[b * T + t] = st;
                }
            }
            for (int b = 0; b < nb; b++)
                for (int t = 0; t < T; t++) {
                    const size_t row = (size_t)t * N + node[b * T + t];
                    const float st = sv[b * T + t];
                    node[b * T + t] = 2 * node[b * T + t] + 1 + (st > 0.0f);  // the branch
                    if (k < D)  // only the chosen child's state: states are most of the bytes, a guess would double them
                        prefetch_lines((char *)h + b * seq_h + layer_h + ((size_t)t * N + node[b * T + t]) * n * pv * esz,
                                       (size_t)n * pv * esz);
                    paxpy(Y(b), gelu(st), w_out.bits + row * row_u64, w_out.scales + row * (d / g), d, g);
                }
        }
        for (int b = 0; b < nb; b++) {
            if (m->readout)  // every tree's readout rows at once: (trees or 1) x value rows, y kept in registers
                paxpy_rows(Y(b), RACC(b), (m->readout == 1 ? T : 1) * pv, U.bits, U.scales, d, g);
            float *x = X(b), *y = Y(b);
            for (int i = 0; i < d; i++) x[i] += y[i];
        }
    }
    for (int b = 0; b < nb; b++) {
        rmsnorm(U_(b), X(b), m->norm_f, d);
        matvec16(logits + (size_t)b * m->vocab, m->embed, U_(b), m->vocab, d);
    }
    return 0;
}

// n tokens in a row for each of nb sequences: tokens (n, nb). With OpenMP the sequences are split across threads, each
// walking its share together (weights shared, states private). scratch: per sequence as tss_step_batch asks, + 16 per thread.
int tss_run_batch(const Model *m, int nb, void *h, float *hist, const int *tokens, int n, float *logits, float *scratch,
                  int threads) {
    const int T = m->trees, D = m->depth, N = (1 << (D + 1)) - 1, d = m->d, P = 2 * m->state + 2 * m->value + T;
    const size_t seq_h = (size_t)m->n_layer * T * N * m->state * m->value * (m->state_bf16 ? 2 : 4);
    const size_t seq_hist = (size_t)m->n_layer * (m->d_conv - 1) * d;
    const size_t seq_scratch = 5 * (size_t)d + 2 * P + 2 * (T + 1) * m->value + 128;
    int err = 0;
#ifdef _OPENMP
    int nt = threads > 0 ? threads : omp_get_max_threads();
#else
    int nt = 1;
    (void)threads;
#endif
    if (nt > nb) nt = nb;
#pragma omp parallel for num_threads(nt) reduction(| : err) schedule(static, 1)
    for (int w = 0; w < nt; w++) {
        const int b0 = nb * w / nt, b1 = nb * (w + 1) / nt, k = b1 - b0;
        int *tok = (int *)malloc(sizeof(int) * k);
        float *sc = scratch + (size_t)b0 * seq_scratch + 16 * w;
        for (int i = 0; i < n && !err; i++) {
            for (int j = 0; j < k; j++) tok[j] = tokens[(size_t)i * nb + b0 + j];
            err |= tss_step_batch(m, k, (char *)h + b0 * seq_h, hist + b0 * seq_hist, tok, logits + (size_t)b0 * m->vocab, sc);
        }
        free(tok);
    }
    return err ? -1 : 0;
}
#else
int tss_run_batch(const Model *m, int nb, void *h, float *hist, const int *tokens, int n, float *logits, float *scratch,
                  int threads) {
    (void)m, (void)nb, (void)h, (void)hist, (void)tokens, (void)n, (void)logits, (void)scratch, (void)threads;
    return -1;  // needs AVX-512BW
}
#endif
"""


def build_c(source, name, flags=()):
    """Compile an embedded C source once (per version and flags) into ~/.cache/nanotss; returns the library's path."""
    import hashlib
    import subprocess

    cache = os.path.join(os.path.expanduser("~"), ".cache", "nanotss")
    os.makedirs(cache, exist_ok=True)
    tag = hashlib.sha1((source + " ".join(flags)).encode()).hexdigest()[:12]
    so, src = os.path.join(cache, f"{name}_{tag}.so"), os.path.join(cache, f"{name}_{tag}.c")
    if not os.path.exists(so):
        with open(src, "w") as f:
            f.write(source)
        subprocess.check_call(["cc", "-O3", "-march=native", "-ffast-math", "-fopenmp", "-shared", "-fPIC", *flags, "-o", so, src, "-lm"])
    return so


def c_step(model, state_bf16=False):
    """The C step built and loaded, with the model packed for it: ternary weights as int8 codes and one scale per
    group, as the forward pass uses them (and as 2-bit planes). Returns step(token, state, logits), with .run,
    .new_state, .new_batch and .run_batch."""
    import ctypes

    import numpy as np

    cfg = model.cfg
    assert cfg.ternary and cfg.act_bits == 8 and cfg.memory, "the C step runs the ternary, 8-bit, stateful model"
    lib = ctypes.CDLL(build_c(C_SOURCE, "step"))
    f, i8 = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int8)
    g = group_size(cfg.d_model, cfg.ternary_group)

    class Ternary(ctypes.Structure):
        _fields_ = [("codes", i8), ("scales", f)]

    class Packed(ctypes.Structure):
        _fields_ = [("bits", ctypes.POINTER(ctypes.c_uint64)), ("scales", f)]

    class Model(ctypes.Structure):
        _fields_ = [(k, ctypes.c_int) for k in ("vocab", "d", "n_layer", "trees", "depth", "state", "value", "d_conv",
                                                "memory", "readout", "group")] + [
            ("embed", f), ("norm_f", f), ("norm", f), ("conv", f), ("w_in", Ternary), ("bias", f), ("w_out", Ternary),
            ("proj", Ternary), ("dt_bias", f), ("neg_A", f), ("U", Ternary),
            ("w_in2", Packed), ("w_out2", Packed), ("proj2", Packed), ("U2", Packed), ("state_bf16", ctypes.c_int)]

    keep = []

    def floats(fn):
        a = np.ascontiguousarray(torch.cat([fn(l).detach().float().flatten() for l in model.layers]).numpy())
        keep.append(a)
        return a.ctypes.data_as(f)

    packed = {}

    def tern(fn):  # the codes and scales ``ternary`` computes, over every layer; also kept as 2-bit planes
        codes, scales = [], []
        for l in model.layers:
            w = fn(l).detach().double().reshape(-1, cfg.d_model)
            wg = w.reshape(w.shape[0], -1, g)
            sc = wg.abs().mean(-1, keepdim=True).clamp(min=1e-8)
            codes.append((wg / sc).clamp(-1, 1).round().to(torch.int8).flatten())
            scales.append(sc.float().flatten())
        c, sc = np.ascontiguousarray(torch.cat(codes).numpy()), np.ascontiguousarray(torch.cat(scales).numpy())
        keep.extend([c, sc])
        if cfg.d_model % 64 == 0:  # per row, per 64 weights: a plus and a minus bitmask, bit i = weight i
            chunks = c.reshape(-1, cfg.d_model // 64, 64)
            weights = (np.uint64(1) << np.arange(64, dtype=np.uint64))
            planes = np.stack([((chunks == 1) * weights).sum(-1, dtype=np.uint64),
                               ((chunks == -1) * weights).sum(-1, dtype=np.uint64)], -1)
            planes = np.ascontiguousarray(planes.reshape(-1))
            keep.append(planes)
            packed[len(packed)] = Packed(planes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), sc.ctypes.data_as(f))
        return Ternary(c.ctypes.data_as(i8), sc.ctypes.data_as(f))

    embed = np.ascontiguousarray(model.embed.weight.detach().float().numpy())
    norm_f = np.ascontiguousarray(model.norm_f.weight.detach().float().numpy())
    keep.extend([embed, norm_f])
    m = Model(cfg.vocab_size, cfg.d_model, cfg.n_layer, cfg.trees, cfg.depth, cfg.state, cfg.value, cfg.d_conv, 1,
              ["level", "tree", "layer"].index(cfg.readout), g, embed.ctypes.data_as(f), norm_f.ctypes.data_as(f),
              floats(lambda l: model.norms[list(model.layers).index(l)].weight), floats(lambda l: l.conv),
              tern(lambda l: l.w_in), floats(lambda l: l.bias), tern(lambda l: l.w_out), tern(lambda l: l.proj.weight),
              floats(lambda l: l.dt_bias), floats(lambda l: -torch.exp(l.A_log)), tern(lambda l: l.U))
    if packed:  # w_in, w_out, proj, U in the order tern() saw them
        m.w_in2, m.w_out2, m.proj2, m.U2 = packed[0], packed[1], packed[2], packed[3]
    m.state_bf16 = int(state_bf16)
    N = 2 ** (cfg.depth + 1) - 1
    P = 2 * cfg.state + 2 * cfg.value + cfg.trees
    scratch = np.zeros(8 * cfg.d_model + 2 * P + 4 * cfg.trees * cfg.value + 64, dtype=np.float32)  # with room to align
    lib.tss_step.argtypes = [ctypes.POINTER(Model), f, f, ctypes.c_int, f, f]
    for fn in (lib.tss_run, lib.tss_run_naive):
        fn.argtypes = [ctypes.POINTER(Model), f, f, ctypes.POINTER(ctypes.c_int), ctypes.c_int, f, f]

    def new_state():
        return (np.zeros(cfg.n_layer * cfg.trees * N * cfg.state * cfg.value, dtype=np.float32),
                np.zeros(cfg.n_layer * max(cfg.d_conv - 1, 1) * cfg.d_model, dtype=np.float32))

    lib.tss_run_batch.argtypes = [ctypes.POINTER(Model), ctypes.c_int, ctypes.c_void_p, f, ctypes.POINTER(ctypes.c_int),
                                  ctypes.c_int, f, f, ctypes.c_int]
    lib.tss_run_batch.restype = ctypes.c_int
    state_elems = cfg.n_layer * cfg.trees * N * cfg.state * cfg.value
    hist_elems = cfg.n_layer * max(cfg.d_conv - 1, 1) * cfg.d_model

    def new_batch(nb):
        return (np.zeros(nb * state_elems, dtype=np.uint16 if state_bf16 else np.float32),
                np.zeros(nb * hist_elems, dtype=np.float32))

    batch_scratch = {}

    def run_batch(tokens, state, logits, threads=1):
        """tokens (n, nb): n steps of nb sequences at once, on 2-bit weights (AVX-512), the sequences split across
        threads. logits (nb, vocab): the last step's."""
        n, nb = tokens.shape
        sc = batch_scratch.setdefault(nb, np.zeros(nb * (5 * cfg.d_model + 2 * P + 2 * (cfg.trees + 1) * cfg.value + 128)
                                                  + 16 * nb + 64, dtype=np.float32))
        tok = np.ascontiguousarray(tokens, dtype=np.int32)
        err = lib.tss_run_batch(ctypes.byref(m), nb, state[0].ctypes.data, state[1].ctypes.data_as(f),
                                tok.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), n, logits.ctypes.data_as(f), sc.ctypes.data_as(f),
                                threads)
        assert err == 0, "tss_run_batch: needs AVX-512BW, d and the group multiples of 64, value of 16, nb <= 64"

    def step(token, state, logits):
        lib.tss_step(ctypes.byref(m), state[0].ctypes.data_as(f), state[1].ctypes.data_as(f), int(token),
                     logits.ctypes.data_as(f), scratch.ctypes.data_as(f))

    def run(tokens, state, logits, naive=False):
        tok = np.ascontiguousarray(tokens, dtype=np.int32)
        (lib.tss_run_naive if naive else lib.tss_run)(ctypes.byref(m), state[0].ctypes.data_as(f), state[1].ctypes.data_as(f),
                    tok.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(tok), logits.ctypes.data_as(f), scratch.ctypes.data_as(f))

    step.keep, step.new_state, step.run, step.new_batch, step.run_batch = (keep, m), new_state, run, new_batch, run_batch
    return step


@torch.no_grad()
def bench(a):
    """The C step against the PyTorch recurrence (same logits), then single-stream tokens per second on one core."""
    import numpy as np

    torch.manual_seed(0)
    torch.set_num_threads(1)
    cfg = Config(vocab_size=a.vocab, d_model=a.d_model, n_layer=a.layers, trees=a.trees, depth=a.depth, state=a.state,
                 value=a.value, readout=a.readout)
    model = TSS(cfg).eval()
    for layer in model.layers:
        layer.w_in.data *= 4
    n_params = sum(p.numel() for p in model.parameters())
    step = c_step(model)
    ids = torch.randint(0, cfg.vocab_size, (64,))
    states, cs, logits = model.init_state(1), step.new_state(), np.zeros(cfg.vocab_size, dtype=np.float32)
    worst = 0.0
    for t in range(len(ids)):
        ref = model.step(ids[t : t + 1], states)[0].numpy()
        step(ids[t], cs, logits)
        worst = max(worst, float(np.abs(logits - ref).max() / (np.abs(ref).max() + 1e-9)))
    print(f"tss bench: {n_params / 1e6:.2f}M parameters, d {cfg.d_model}, {cfg.n_layer} layers x {cfg.trees} trees of depth "
          f"{cfg.depth} ({2 ** (cfg.depth + 1) - 1} nodes, {cfg.depth + 1} visited), node state {cfg.state} x {cfg.value}, readout {cfg.readout}, "
          f"ternary weights, 8-bit activations")
    print(f"  C step vs PyTorch recurrence over {len(ids)} tokens: worst logit difference {worst:.1e} relative")
    naive_state, naive_logits = step.new_state(), np.zeros_like(logits)
    for t in range(len(ids)):  # the fast step against the plain C one, same tokens, same states
        step.run(ids[t : t + 1].numpy(), naive_state, naive_logits, naive=True)
    print(f"  fast C step vs plain C step: worst logit difference "
          f"{float(np.abs(logits - naive_logits).max() / (np.abs(naive_logits).max() + 1e-9)):.1e} relative")
    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, a.tokens)
    t0 = time.perf_counter()
    step.run(tokens, cs, logits)  # the loop in C: no Python per token
    c_rate = a.tokens / (time.perf_counter() - t0)
    t0 = time.perf_counter()
    step.run(tokens, naive_state, naive_logits, naive=True)
    naive_rate = a.tokens / (time.perf_counter() - t0)
    t0 = time.perf_counter()
    for t in range(200):
        model.step(torch.tensor([t % cfg.vocab_size]), states)
    py_rate = 200 / (time.perf_counter() - t0)
    print(f"  one core, batch 1: fast C {c_rate:,.0f} tokens/s ({1e6 / c_rate:.1f} us/token); plain C {naive_rate:,.0f} "
          f"({1e6 / naive_rate:.1f} us/token); PyTorch step {py_rate:,.0f} tokens/s")
    if cfg.d_model % 64 == 0 and cfg.value % 16 == 0:  # the 2-bit kernel (AVX-512): sequences at once, on cores
        cores = os.cpu_count()
        for bf in (False, True):
            st2 = c_step(model, state_bf16=bf)
            for nb, threads in ((1, 1), (4, 1), (16, 1), (cores, cores), (4 * cores, cores), (16 * cores, cores)):
                state, lg = st2.new_batch(nb), np.zeros((nb, cfg.vocab_size), dtype=np.float32)
                toks = np.random.default_rng(1).integers(0, cfg.vocab_size, (max(a.tokens // nb, 50), nb))
                try:
                    t0 = time.perf_counter()
                    st2.run_batch(toks, state, lg, threads)
                except AssertionError:
                    print("  2-bit kernel: not available on this CPU (needs AVX-512BW)")
                    return
                rate = toks.size / (time.perf_counter() - t0)
                print(f"  2-bit weights, {'bf16' if bf else 'f32 '} state, {nb:3d} sequences on {threads} core{'s' if threads > 1 else ' '}: "
                      f"{rate:9,.0f} tokens/s  ({1e6 / rate * threads:6.1f} core-us/token; state {state[0].nbytes / nb / 1e6:.1f} MB a sequence)")


# ------------------------------------------------------------------------------------------------- C training (sparse)


C_TRAIN_SOURCE = r"""// Tree state space: the training step on a CPU, sparse. One sequence's loss and gradients, the same function and the
// same gradients as the training form (TreeSSM.forward and autograd): a tree forward pass that records a tape,
// then a backward pass that touches only what the forward touched. Per token and layer: the depth + 1 visited rows of
// each tree (w_in, w_out, bias, the node's decay), the visited nodes' states (an adjoint state per node, run backward
// in time), and for the branch gradient the other subtree's nodes as scalars <g, out> only. The dense parts (proj, U,
// conv, norms, the output layer) are ordinary backprop. Gradients go to the latent float weights (the ternary codes and
// 8-bit activations straight-through), into buffers the caller owns; tss_train_rows reports which node rows were hit.
//
// Readout per tree (cfg readout "tree"), memory on, ternary weights, 8-bit activations. REAL is float, or double for
// checking against PyTorch's float64 autograd. Build: cc -O3 -march=native -fopenmp -shared -fPIC [-DREAL=double]
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef REAL
#define REAL float
#endif
typedef REAL real;

typedef struct {
    int vocab, d, n_layer, T, D, n, p, K, group;
    int bdepth;  // the branch gradient compares the two subtrees this many levels down (0: all the way)
    real temp;
} Cfg;

typedef struct {  // every tensor's layers one after another, in the layer's own shape
    real *embed, *norm_f;               // (vocab, d), (d)
    real *norm, *conv;                  // (d), (K, d)
    real *w_in, *bias, *w_out, *A_log;  // (T * N, d), (T * N), (T * N, d), (T * N)
    real *proj, *dt_bias, *U;           // (P, d), (T), (T * p, d)
} Params;

#define NODES(c) ((1 << ((c)->D + 1)) - 1)
#define PDIM(c) (2 * (c)->n + 2 * (c)->p + (c)->T)
#define BDEPTH(c) ((c)->bdepth > 0 && (c)->bdepth < (c)->D ? (c)->bdepth : (c)->D)

static real gelu(real x) { return (real)0.5 * x * (1 + erf(x * (real)0.70710678118654752440)); }
static real dgelu(real x) {
    return (real)0.5 * (1 + erf(x * (real)0.70710678118654752440)) + x * exp((real)-0.5 * x * x) * (real)0.39894228040143267794;
}
static real softplus(real x) { return x > 20 ? x : log1p(exp(x)); }
static real sigm(real x) { return 1 / (1 + exp(-x)); }

static real dotr(const real *a, const real *b, int n) {
    real s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

// absmean ternary per group of each row (nanotss.ternary): out = round(clamp(w / s, -1, 1)) * s, s = mean |w| of the group
void tss_ternary(const real *w, real *out, long rows, int d, int g) {
    for (long r = 0; r < rows; r++)
        for (int j = 0; j < d; j += g) {
            const real *wr = w + r * d + j;
            real s = 0;
            for (int i = 0; i < g; i++) s += fabs(wr[i]);
            s = s / g;
            if (s < (real)1e-8) s = (real)1e-8;
            for (int i = 0; i < g; i++) {
                real v = wr[i] / s;
                v = v > 1 ? 1 : (v < -1 ? -1 : v);
                out[r * d + j + i] = nearbyint(v) * s;
            }
        }
}

// The quantized weights of every layer, from the latent ones (the caller's buffers, Params-shaped; only w_in, w_out,
// proj, U are written). Rows: an optimizer that touched only some rows may requantize only those.
void tss_quantize(const Cfg *c, const Params *P, Params *Q) {
    const long N = NODES(c), L = c->n_layer;
    tss_ternary(P->w_in, Q->w_in, L * c->T * N, c->d, c->group);
    tss_ternary(P->w_out, Q->w_out, L * c->T * N, c->d, c->group);
    tss_ternary(P->proj, Q->proj, L * PDIM(c), c->d, c->group);
    tss_ternary(P->U, Q->U, L * c->T * c->p, c->d, c->group);
}

// per-token 8-bit activations (nanotss.quantize_activations): the values, their codes, and the codes' scale
static void quant8(const real *x, real *xq, int8_t *code, real *xs, int d) {
    real m = 0;
    for (int i = 0; i < d; i++) m = fabs(x[i]) > m ? fabs(x[i]) : m;
    if (m < (real)1e-5) m = (real)1e-5;
    const real scale = 127 / m;
    for (int i = 0; i < d; i++) {
        real v = nearbyint(x[i] * scale);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        xq[i] = v / scale;
        code[i] = (int8_t)v;
    }
    *xs = 1 / scale;
}

static void rmsnorm_fwd(const real *x, const real *w, real *out, int d, real *r_out) {
    const real r = 1 / sqrt(dotr(x, x, d) / d + (real)1e-5);
    for (int i = 0; i < d; i++) out[i] = x[i] * r * w[i];
    *r_out = r;
}

// out = x r w: dx += r (w du) - x r^3 / d sum(w du x); dw += du x r
static void rmsnorm_bwd(const real *x, const real *w, real r, const real *du, real *dx, real *dw, int d) {
    real s = 0;
    for (int i = 0; i < d; i++) s += w[i] * du[i] * x[i];
    const real k = r * r * r * s / d;
    for (int i = 0; i < d; i++) {
        dx[i] += r * w[i] * du[i] - x[i] * k;
        dw[i] += du[i] * x[i] * r;
    }
}

// ----------------------------------------------------------------------------- the rows: 2-bit codes, fast paths
// A ternary row as two bitmasks per 64 weights (plus, minus) and one scale per group, beside the quantized values.
// With AVX-512BW in float: a row's dot product with int8 x is masked byte add/subtract, with a float vector masked float
// add/subtract, and y += a row is a masked add and a masked subtract per 16 floats. Without them (or in the double build
// that checks against autograd) the same helpers run the plain loops on the quantized values.
#if defined(__AVX512BW__) && defined(__AVX512F__) && !defined(TSS_DOUBLE)
#define FAST 1
#include <immintrin.h>
#else
#define FAST 0
#endif

typedef struct {
    uint64_t *bits;  // per row: d / 64 pairs (plus, minus)
    real *sc;        // per row: d / group
} Plane;

typedef struct {
    int ok;  // the fast path applies (FAST, d and group multiples of 64, value a multiple of 16)
    Plane w_in, w_out, proj, U;
} Codes;

static void plane_rows(const real *q, Plane *pl, long r0, long r1, int d, int g) {
    for (long r = r0; r < r1; r++) {
        const real *qr = q + r * d;
        uint64_t *b = pl->bits + r * (d / 32);
        for (int j = 0; j < d; j += 64) {
            uint64_t P = 0, Nm = 0;
            for (int i = 0; i < 64; i++) {
                P |= (uint64_t)(qr[j + i] > 0) << i;
                Nm |= (uint64_t)(qr[j + i] < 0) << i;
            }
            b[(j / 64) * 2] = P, b[(j / 64) * 2 + 1] = Nm;
        }
        for (int j = 0; j < d; j += g) {
            real m = 0;
            for (int i = 0; i < g; i++) m = fabs(qr[j + i]) > m ? fabs(qr[j + i]) : m;
            pl->sc[r * (d / g) + j / g] = m;
        }
    }
}

static Plane plane_new(long rows, int d, int g) {
    Plane p = {calloc((size_t)rows * (d / 32), sizeof(uint64_t)), calloc((size_t)rows * (d / g), sizeof(real))};
    return p;
}

Codes *tss_codes_new(const Cfg *c) {
    Codes *k = calloc(1, sizeof(Codes));
    k->ok = FAST && c->d % 64 == 0 && c->group % 64 == 0 && c->p % 16 == 0;
    if (!k->ok) return k;
    const long N = NODES(c), L = c->n_layer;
    k->w_in = plane_new(L * c->T * N, c->d, c->group), k->w_out = plane_new(L * c->T * N, c->d, c->group);
    k->proj = plane_new(L * PDIM(c), c->d, c->group), k->U = plane_new(L * c->T * c->p, c->d, c->group);
    return k;
}

// the planes of every row from the quantized values (after tss_quantize)
void tss_codes_all(const Cfg *c, const Params *Q, Codes *k) {
    if (!k || !k->ok) return;
    const long N = NODES(c), L = c->n_layer;
    plane_rows(Q->w_in, &k->w_in, 0, L * c->T * N, c->d, c->group);
    plane_rows(Q->w_out, &k->w_out, 0, L * c->T * N, c->d, c->group);
    plane_rows(Q->proj, &k->proj, 0, L * PDIM(c), c->d, c->group);
    plane_rows(Q->U, &k->U, 0, L * c->T * c->p, c->d, c->group);
}

#if FAST
static inline float pdot8(const Plane *pl, size_t r, const int8_t *xq, float xs, int d, int g) {
    const uint64_t *b = pl->bits + r * (d / 32);
    const float *sc = pl->sc + r * (d / g);
    const __m512i ones8 = _mm512_set1_epi8(1), ones16 = _mm512_set1_epi16(1);
    float acc = 0;
    for (int j = 0; j < d; j += g) {
        __m512i s32 = _mm512_setzero_si512();
        for (int i = j; i < j + g; i += 64) {
            const __m512i x = _mm512_loadu_si512((const void *)(xq + i));
            __m512i v = _mm512_maskz_mov_epi8(b[(i / 64) * 2], x);
            v = _mm512_mask_sub_epi8(v, b[(i / 64) * 2 + 1], v, x);
            s32 = _mm512_add_epi32(s32, _mm512_madd_epi16(_mm512_maddubs_epi16(ones8, v), ones16));
        }
        acc += sc[j / g] * (float)_mm512_reduce_add_epi32(s32);
    }
    return acc * xs;
}

static inline float pdotf(const Plane *pl, size_t r, const float *v, int d, int g) {
    const uint64_t *b = pl->bits + r * (d / 32);
    const float *sc = pl->sc + r * (d / g);
    float acc = 0;
    for (int j = 0; j < d; j += g) {
        __m512 s = _mm512_setzero_ps();
        for (int i = j; i < j + g; i += 64) {
            const uint64_t P = b[(i / 64) * 2], Nm = b[(i / 64) * 2 + 1];
            for (int q = 0; q < 4; q++) {
                const __m512 x = _mm512_loadu_ps(v + i + 16 * q);
                s = _mm512_mask_add_ps(s, (__mmask16)(P >> (16 * q)), s, x);
                s = _mm512_mask_sub_ps(s, (__mmask16)(Nm >> (16 * q)), s, x);
            }
        }
        acc += sc[j / g] * _mm512_reduce_add_ps(s);
    }
    return acc;
}

static inline void paxpyf(float *y, float a, const Plane *pl, size_t r, int d, int g) {
    const uint64_t *b = pl->bits + r * (d / 32);
    const float *sc = pl->sc + r * (d / g);
    for (int j = 0; j < d; j += g) {
        const __m512 as = _mm512_set1_ps(a * sc[j / g]);
        for (int i = j; i < j + g; i += 64) {
            const uint64_t P = b[(i / 64) * 2], Nm = b[(i / 64) * 2 + 1];
            for (int q = 0; q < 4; q++) {
                __m512 yv = _mm512_loadu_ps(y + i + 16 * q);
                yv = _mm512_mask_add_ps(yv, (__mmask16)(P >> (16 * q)), yv, as);
                yv = _mm512_mask_sub_ps(yv, (__mmask16)(Nm >> (16 * q)), yv, as);
                _mm512_storeu_ps(y + i + 16 * q, yv);
            }
        }
    }
}
#endif

// <row r of a quantized matrix, x> (x as int8 codes xq times xs on the fast path)
static inline real qdotx(const Codes *k, const Plane *pl, const real *qm, size_t r, const real *x, const int8_t *xq, real xs,
                         int d, int g) {
#if FAST
    if (k && k->ok) return pdot8(pl, r, xq, xs, d, g);
#endif
    (void)k, (void)pl, (void)xq, (void)xs, (void)g;
    return dotr(qm + r * d, x, d);
}

// <row r, v>, v float
static inline real qdotf(const Codes *k, const Plane *pl, const real *qm, size_t r, const real *v, int d, int g) {
#if FAST
    if (k && k->ok) return pdotf(pl, r, v, d, g);
#endif
    (void)k, (void)pl, (void)g;
    return dotr(qm + r * d, v, d);
}

// y += a * row r
static inline void qaxpy(const Codes *k, const Plane *pl, const real *qm, size_t r, real *y, real a, int d, int g) {
#if FAST
    if (k && k->ok) {
        paxpyf(y, a, pl, r, d, g);
        return;
    }
#endif
    (void)k, (void)pl, (void)g;
    const real *row = qm + r * d;
    for (int i = 0; i < d; i++) y[i] += a * row[i];
}

static inline void axpy(real *y, real a, const real *x, int d) {
    for (int i = 0; i < d; i++) y[i] += a * x[i];
}

typedef struct {  // one layer's tape over a sequence of L tokens
    real *xin, *u, *rn, *x, *p, *dt;   // (L, d) x3, (L), (L, P), (L, T)
    real *xs;                           // (L): x = xq * xs
    int8_t *xq;                         // (L, d): the 8-bit activations' codes
    int *node;                          // (L, T, D + 1)
    real *s, *R, *hprev, *decay;        // (L, T, D + 1), (.., p), (.., n * p), (..)
    int *anode;                         // the other subtrees: (L, T, D (D + 1) / 2)
    real *as, *aR;                      // (.., ), (.., p)
} Tape;

static size_t tape_reals(const Cfg *c, int L) {
    const size_t V = (size_t)L * c->T * (c->D + 1), A = (size_t)L * c->T * c->D * (c->D + 1) / 2;
    return (size_t)L * (3 * c->d + 2 + PDIM(c) + c->T) + V * (1 + c->p + c->n * c->p + 1) + A * (1 + c->p);
}

static void tape_carve(const Cfg *c, int L, real *buf, int *ibuf, Tape *tp) {
    const size_t V = (size_t)L * c->T * (c->D + 1), A = (size_t)L * c->T * c->D * (c->D + 1) / 2;
    tp->xin = buf, buf += (size_t)L * c->d;
    tp->u = buf, buf += (size_t)L * c->d;
    tp->x = buf, buf += (size_t)L * c->d;
    tp->rn = buf, buf += L;
    tp->xs = buf, buf += L;
    tp->p = buf, buf += (size_t)L * PDIM(c);
    tp->dt = buf, buf += (size_t)L * c->T;
    tp->s = buf, buf += V;
    tp->R = buf, buf += V * c->p;
    tp->hprev = buf, buf += V * c->n * c->p;
    tp->decay = buf, buf += V;
    tp->as = buf, buf += A;
    tp->aR = buf, buf += A * c->p;
    tp->node = ibuf, ibuf += V;
    tp->anode = ibuf;
}

// One layer forward over the sequence: x_res (L, d) in place (x_res += y). h: the node states (T * N, n * p), zeroed.
typedef struct {
    int Lmax;
    real *tapes, *x, *h, *g, *xn, *dxn, *lg, *lam;
    real *ds, *dR, *dp, *dx, *du, *gUr, *on, *Racc, *mx, *y;
    int *itapes;
    int8_t *xq8;
    int *rcnt, *rord;
    int *walk;  // the walks' current nodes: a tree's path, or a tree's chains through its other subtrees (T * (D + 1))  // the visits sorted by node row: (T * N + 1) offsets, (Lmax * T * (D + 1)) visit indices
    uint16_t *hp16;  // the fast path's tape of previous states in bfloat16 (n_layer, Lmax, T, D + 1, n * p)
    int tape_bf16;   // use it (tss_work_bf16): half the tape's bytes, the states' gradients from rounded values
    Tape *tp;
    const Codes *k;  // the row codes (the fast path), or NULL
} Work;

static void layer_fwd(const Cfg *c, const Params *P, const Params *Q, int l, int L, real *xres, real *h, Tape *tp, Work *w) {
    const int d = c->d, T = c->T, D = c->D, n = c->n, pv = c->p, K = c->K, Pd = PDIM(c), N = NODES(c), g = c->group;
    const real *norm = P->norm + (size_t)l * d, *conv = P->conv + (size_t)l * K * d;
    const real *bias = P->bias + (size_t)l * T * N, *A_log = P->A_log + (size_t)l * T * N, *dt_bias = P->dt_bias + (size_t)l * T;
    const size_t r_node = (size_t)l * T * N, r_proj = (size_t)l * Pd, r_U = (size_t)l * T * pv;  // the layer's first rows
    const Codes *k = w->k;
    const Plane *pin = k ? &k->w_in : NULL, *pout = k ? &k->w_out : NULL, *pproj = k ? &k->proj : NULL, *pU = k ? &k->U : NULL;
    const int AL = D * (D + 1) / 2;
#if FAST
    const int vec = k && k->ok && pv == 16;  // a node state's row is one register
    uint16_t *hp16 = w->hp16 + (size_t)l * w->Lmax * T * (D + 1) * n * pv;
    const int bf = vec && w->tape_bf16;
#endif
    real *mx = w->mx, *y = w->y, *Racc = w->Racc;
    for (int t = 0; t < L; t++) {
        real *xt = tp->x + (size_t)t * d, *ut = tp->u + (size_t)t * d, *pt = tp->p + (size_t)t * Pd, *dtt = tp->dt + (size_t)t * T;
        int8_t *xq = tp->xq + (size_t)t * d;
        memcpy(tp->xin + (size_t)t * d, xres + (size_t)t * d, sizeof(real) * d);
        rmsnorm_fwd(xres + (size_t)t * d, norm, ut, d, tp->rn + t);
        for (int i = 0; i < d; i++) mx[i] = conv[i] * ut[i];
        for (int j = 1; j < K; j++)
            if (t - j >= 0)
                for (int i = 0; i < d; i++) mx[i] += conv[j * d + i] * tp->u[(size_t)(t - j) * d + i];
        quant8(mx, xt, xq, tp->xs + t, d);
        const real xs = tp->xs[t];
        for (int r = 0; r < Pd; r++) pt[r] = qdotx(k, pproj, Q->proj, r_proj + r, xt, xq, xs, d, g);
        const real *B = pt, *C = pt + n, *V = pt + 2 * n, *Z = pt + 2 * n + pv;
        for (int tr = 0; tr < T; tr++) dtt[tr] = softplus(pt[2 * n + 2 * pv + tr] + dt_bias[tr]);
        memset(y, 0, sizeof(real) * d);
        memset(Racc, 0, sizeof(real) * T * pv);
        int *node = w->walk;
        for (int tr = 0; tr < T; tr++) node[tr] = 0;
        for (int kk = 0; kk <= D; kk++)  // the trees in lockstep, level by level: their fetches overlap
            for (int tr = 0; tr < T; tr++) {  // the path: decay, read, branch, output, write
                const size_t row = (size_t)tr * N + node[tr], v = ((size_t)t * T + tr) * (D + 1) + kk;
                real *ha = h + row * n * pv, *R = tp->R + v * pv;
                const real dec = exp(-exp(A_log[row]) * dtt[tr]), wdt = dtt[tr];
#if FAST
                if (!bf)
#endif
                    memcpy(tp->hprev + v * n * pv, ha, sizeof(real) * n * pv);
#if FAST
                if (vec) {
                    const __m512 vd = _mm512_set1_ps(dec), vV = _mm512_loadu_ps(V);
                    __m512 vR = _mm512_setzero_ps();
                    uint16_t *hb = hp16 + v * n * pv;
                    for (int j = 0; j < n; j++) {
                        const __m512 h0 = _mm512_loadu_ps(ha + j * 16);
                        if (bf) {  // the previous state, rounded to bfloat16 (nearest, ties to even)
                            const __m512i u = _mm512_castps_si512(h0);
                            const __m512i r = _mm512_add_epi32(u, _mm512_add_epi32(_mm512_set1_epi32(0x7fff),
                                                               _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1))));
                            _mm256_storeu_si256((__m256i *)(hb + j * 16), _mm512_cvtepi32_epi16(_mm512_srli_epi32(r, 16)));
                        }
                        const __m512 hv = _mm512_mul_ps(h0, vd);
                        vR = _mm512_fmadd_ps(_mm512_set1_ps(C[j]), hv, vR);
                        _mm512_storeu_ps(ha + j * 16, _mm512_fmadd_ps(_mm512_set1_ps(wdt * B[j]), vV, hv));
                    }
                    _mm512_storeu_ps(R, vR);
                } else
#endif
                {
                    for (int q = 0; q < pv; q++) R[q] = 0;
                    for (int j = 0; j < n; j++)
                        for (int q = 0; q < pv; q++) {
                            ha[j * pv + q] *= dec;
                            R[q] += C[j] * ha[j * pv + q];
                        }
                    for (int j = 0; j < n; j++)
                        for (int q = 0; q < pv; q++) ha[j * pv + q] += wdt * B[j] * V[q];
                }
                const real s = qdotx(k, pin, Q->w_in, r_node + row, xt, xq, xs, d, g) + bias[row] + dotr(R, Z, pv);
                qaxpy(k, pout, Q->w_out, r_node + row, y, gelu(s), d, g);
                for (int q = 0; q < pv; q++) Racc[tr * pv + q] += R[q];
                tp->node[v] = (int)row, tp->s[v] = s, tp->decay[v] = dec;
                node[tr] = 2 * node[tr] + 1 + (s > 0);
                if (kk < D) {  // the chosen child's state and row, while the other trees take their turn
                    const size_t nr = (size_t)tr * N + node[tr];
                    const char *hs = (const char *)(h + nr * n * pv);
                    for (int o = 0; o < n * pv * (int)sizeof(real); o += 64) __builtin_prefetch(hs + o);
#if FAST
                    if (k && k->ok) __builtin_prefetch(pin->bits + (r_node + nr) * (d / 32));
#endif
                }
            }
        for (int tr = 0; tr < T; tr++)
            for (int q = 0; q < pv; q++) qaxpy(k, pU, Q->U, r_U + (size_t)tr * pv + q, y, Racc[tr * pv + q], d, g);
        // the other subtrees (read-only: their states as they stand at t, the token's own arrival decay): the chain from
        // branch kk enters the other child at level kk + 1 and follows its own decisions down; a tree's D chains and the
        // trees' are independent, so they advance level by level together and their fetches overlap. Each chain's
        // nodes are stored at base(kk) + (level - kk - 1), base(kk) = sum over m < kk of min(D - m, r): a chain runs r
        // levels (the branch gradient's depth).
        int *cn = w->walk;
        for (int tr = 0; tr < T; tr++)
            for (int kk = 0; kk < D; kk++) {
                const size_t v = ((size_t)t * T + tr) * (D + 1) + kk;
                cn[tr * (D + 1) + kk] = 2 * (tp->node[v] - tr * N) + 2 - (tp->s[v] > 0);
            }
        const int rb = BDEPTH(c);
        for (int j = 1; j <= D; j++)
            for (int tr = 0; tr < T; tr++) {
                const size_t ab = ((size_t)t * T + tr) * AL;
                for (int kk = 0, base = 0; kk < j; base += (D - kk < rb ? D - kk : rb), kk++) {
                    if (j - kk > rb) continue;  // this chain has run its r levels
                    const size_t ai = ab + base + (j - kk - 1), row = (size_t)tr * N + cn[tr * (D + 1) + kk];
                    const real dec = exp(-exp(A_log[row]) * dtt[tr]), *hr = h + row * n * pv;
                    real *R = tp->aR + ai * pv;
#if FAST
                    if (vec) {
                        __m512 vR = _mm512_setzero_ps();
                        for (int jj = 0; jj < n; jj++) vR = _mm512_fmadd_ps(_mm512_set1_ps(C[jj]), _mm512_loadu_ps(hr + jj * 16), vR);
                        _mm512_storeu_ps(R, _mm512_mul_ps(vR, _mm512_set1_ps(dec)));
                    } else
#endif
                    {
                        for (int q = 0; q < pv; q++) R[q] = 0;
                        for (int jj = 0; jj < n; jj++)
                            for (int q = 0; q < pv; q++) R[q] += C[jj] * dec * hr[jj * pv + q];
                    }
                    const real s = qdotx(k, pin, Q->w_in, r_node + row, xt, xq, xs, d, g) + bias[row] + dotr(R, Z, pv);
                    tp->anode[ai] = (int)row, tp->as[ai] = s;
                    const int nx = 2 * cn[tr * (D + 1) + kk] + 1 + (s > 0);
                    cn[tr * (D + 1) + kk] = nx;
                    if (j < D && j - kk < rb) {  // the next node's state and row, while the other chains run
                        const size_t nr = (size_t)tr * N + nx;
                        const char *hs = (const char *)(h + nr * n * pv);
                        for (int o = 0; o < n * pv * (int)sizeof(real); o += 64) __builtin_prefetch(hs + o);
#if FAST
                        if (k && k->ok) __builtin_prefetch(pin->bits + (r_node + nr) * (d / 32));
#endif
                    }
                }
            }
        for (int i = 0; i < d; i++) xres[(size_t)t * d + i] += y[i];
    }
}

// One layer backward: g (L, d) is dL/d(layer output) on entry and dL/d(layer input) on return; G accumulates.
// lam: (T * N, n * p) adjoint states, zeroed. hit (T * N) node rows touched, set to 1.
static void layer_bwd(const Cfg *c, const Params *P, const Params *Q, Params *G, int l, int L, real *g, real *lam,
                      unsigned char *hit, const Tape *tp, Work *w) {
    const int d = c->d, T = c->T, D = c->D, n = c->n, pv = c->p, K = c->K, Pd = PDIM(c), N = NODES(c), gs = c->group;
    const int AL = D * (D + 1) / 2;
    const real *norm = P->norm + (size_t)l * d, *conv = P->conv + (size_t)l * K * d;
    const real *A_log = P->A_log + (size_t)l * T * N, *dt_bias = P->dt_bias + (size_t)l * T;
    const size_t r_node = (size_t)l * T * N, r_proj = (size_t)l * Pd, r_U = (size_t)l * T * pv;
    real *gnorm = G->norm + (size_t)l * d, *gconv = G->conv + (size_t)l * K * d;
    real *gw_in = G->w_in + r_node * d, *gw_out = G->w_out + r_node * d, *gbias = G->bias + r_node, *gA = G->A_log + r_node;
    real *gproj = G->proj + r_proj * d, *gdtb = G->dt_bias + (size_t)l * T, *gU = G->U + r_U * d;
    const Codes *k = w->k;
    const Plane *pin = k ? &k->w_in : NULL, *pout = k ? &k->w_out : NULL, *pproj = k ? &k->proj : NULL, *pU = k ? &k->U : NULL;
#if FAST
    const int vec = k && k->ok && pv == 16;
    const uint16_t *hp16 = w->hp16 + (size_t)l * w->Lmax * T * (D + 1) * n * pv;
    const int bf = vec && w->tape_bf16;
#endif
    const size_t V = (size_t)L * T * (D + 1);
    real *ds = w->ds, *dR = w->dR, *dp = w->dp, *dx = w->dx, *gUr = w->gUr, *on = w->on, *Racc = w->Racc;
    memset(ds, 0, sizeof(real) * V), memset(dR, 0, sizeof(real) * V * pv);
    memset(dp, 0, sizeof(real) * (size_t)L * Pd), memset(dx, 0, sizeof(real) * (size_t)L * d);
    // A: per token, what the outputs need (no recurrence): node outputs, readout, branch gradient, s = s0 + <Z, R>
    for (int t = 0; t < L; t++) {
        const real *gy = g + (size_t)t * d, *xt = tp->x + (size_t)t * d, *pt = tp->p + (size_t)t * Pd;
        const real *Z = pt + 2 * n + pv;
        real *dZ = dp + (size_t)t * Pd + 2 * n + pv, *dxt = dx + (size_t)t * d;
        for (int tr = 0; tr < T; tr++) {
            const size_t vb = ((size_t)t * T + tr) * (D + 1), ab = ((size_t)t * T + tr) * AL;
            memset(Racc, 0, sizeof(real) * pv);
            for (int kk = 0; kk <= D; kk++)
                for (int q = 0; q < pv; q++) Racc[q] += tp->R[(vb + kk) * pv + q];
            for (int q = 0; q < pv; q++) {
                gUr[q] = qdotf(k, pU, Q->U, r_U + (size_t)tr * pv + q, gy, d, gs);
                axpy(gU + ((size_t)tr * pv + q) * d, Racc[q], gy, d);
            }
            for (int kk = 0; kk <= D; kk++) {  // each path node's output, <g, out>, and its own gradients
                const size_t v = vb + kk, row = tp->node[v];
                const real s = tp->s[v], go = qdotf(k, pout, Q->w_out, r_node + row, gy, d, gs);
                on[kk] = gelu(s) * go + dotr(gUr, tp->R + v * pv, pv);
                ds[v] += dgelu(s) * go;
                for (int q = 0; q < pv; q++) dR[v * pv + q] += gUr[q];
                hit[row] = 1;
            }
            int ai = 0;
            const int rb = BDEPTH(c);
            for (int kk = 0; kk < D; kk++) {  // the branch: sign * (on - alt) * sigmoid'(s / temp) / temp, r levels each
                const int jend = kk + rb < D ? kk + rb : D;
                real onk = 0, alt = 0;
                for (int j = kk + 1; j <= jend; j++) onk += on[j];
                for (int j = kk + 1; j <= jend; j++, ai++) {
                    const size_t a = ab + ai, row = tp->anode[a];
                    alt += gelu(tp->as[a]) * qdotf(k, pout, Q->w_out, r_node + row, gy, d, gs) + dotr(gUr, tp->aR + a * pv, pv);
                }
                const real s = tp->s[vb + kk], sg = sigm(fabs(s) / c->temp);
                ds[vb + kk] += (s > 0 ? 1 : -1) * (onk - alt) * sg * (1 - sg) / c->temp;
            }
            for (int kk = 0; kk <= D; kk++) {  // s = <x, w_in> + bias + <Z, R>
                const size_t v = vb + kk, row = tp->node[v];
                const real dsv = ds[v];
                for (int q = 0; q < pv; q++) {
                    dZ[q] += dsv * tp->R[v * pv + q];
                    dR[v * pv + q] += dsv * Z[q];
                }
                qaxpy(k, pin, Q->w_in, r_node + row, dxt, dsv, d, gs);
                gbias[row] += dsv;
            }
        }
    }
    // A': the node rows' weight gradients in row order: the visits counting-sorted by row, then each row's sum
    // dw_out += gelu(s) g_t, dw_in += ds x_t over its visits in one run (the row stays in cache; the tokens' g and x
    // are the layer's (L, d) arrays). The same sums as per visit, in another order.
    {
        const int TN = T * N, VD = T * (D + 1);
        int *cnt = w->rcnt, *ord = w->rord;
        memset(cnt, 0, sizeof(int) * (TN + 1));
        for (size_t v = 0; v < V; v++) cnt[tp->node[v] + 1]++;
        for (int r = 0; r < TN; r++) cnt[r + 1] += cnt[r];
        for (size_t v = 0; v < V; v++) ord[cnt[tp->node[v]]++] = (int)v;  // cnt[r] is now the end of row r
        for (int r = 0, start = 0; r < TN; start = cnt[r], r++) {
            if (start == cnt[r]) continue;
            real *go = gw_out + (size_t)r * d, *gi = gw_in + (size_t)r * d;
            for (int e = start; e < cnt[r]; e++) {
                const int v = ord[e], t = v / VD;
                axpy(go, gelu(tp->s[v]), g + (size_t)t * d, d);
                axpy(gi, ds[v], tp->x + (size_t)t * d, d);
            }
        }
    }
    // B: the node states backward in time; per visit: h~ = decay h_prev, R = C^T h~, h = h~ + dt B V^T
    for (int t = L - 1; t >= 0; t--) {
        const real *pt = tp->p + (size_t)t * Pd, *B = pt, *C = pt + n, *Vv = pt + 2 * n;
        real *dpt = dp + (size_t)t * Pd, *dB = dpt, *dC = dpt + n, *dV = dpt + 2 * n, *ddt = dpt + 2 * n + 2 * pv;
        for (int tr = 0; tr < T; tr++) {
            const real dtv = tp->dt[(size_t)t * T + tr];
            for (int kk = 0; kk <= D; kk++) {
                const size_t v = ((size_t)t * T + tr) * (D + 1) + kk, row = tp->node[v];
                real *la = lam + row * n * pv;
                const real *hp = tp->hprev + v * n * pv, dec = tp->decay[v], *dRv = dR + v * pv;
                real ddec = 0;
#if FAST
                if (vec) {
                    const __m512 vV = _mm512_loadu_ps(Vv), vdR = _mm512_loadu_ps(dRv), vd = _mm512_set1_ps(dec);
                    __m512 vBl = _mm512_setzero_ps(), vdec = _mm512_setzero_ps();
                    const uint16_t *hb = hp16 + v * n * pv;
                    for (int j = 0; j < n; j++) {
                        const __m512 lv = _mm512_loadu_ps(la + j * 16);
                        const __m512 hv = bf ? _mm512_castsi512_ps(_mm512_slli_epi32(
                                                   _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(hb + j * 16))), 16))
                                             : _mm512_loadu_ps(hp + j * 16);
                        dB[j] += dtv * _mm512_reduce_add_ps(_mm512_mul_ps(lv, vV));
                        vBl = _mm512_fmadd_ps(_mm512_set1_ps(B[j]), lv, vBl);
                        dC[j] += dec * _mm512_reduce_add_ps(_mm512_mul_ps(hv, vdR));
                        const __m512 dht = _mm512_fmadd_ps(_mm512_set1_ps(C[j]), vdR, lv);
                        vdec = _mm512_fmadd_ps(dht, hv, vdec);
                        _mm512_storeu_ps(la + j * 16, _mm512_mul_ps(vd, dht));
                    }
                    _mm512_storeu_ps(dV, _mm512_fmadd_ps(_mm512_set1_ps(dtv), vBl, _mm512_loadu_ps(dV)));
                    ddt[tr] += _mm512_reduce_add_ps(_mm512_mul_ps(vBl, vV));
                    ddec = _mm512_reduce_add_ps(vdec);
                } else
#endif
                {
                    for (int j = 0; j < n; j++)
                        for (int q = 0; q < pv; q++) {
                            const real lv = la[j * pv + q];
                            dB[j] += dtv * lv * Vv[q];
                            dV[q] += dtv * lv * B[j];
                            ddt[tr] += lv * B[j] * Vv[q];
                            dC[j] += dec * hp[j * pv + q] * dRv[q];
                            const real dht = lv + C[j] * dRv[q];  // dL/dh~
                            ddec += dht * hp[j * pv + q];
                            la[j * pv + q] = dec * dht;  // dL/dh at the node's previous visit
                        }
                }
                const real ea = exp(A_log[row]);
                gA[row] += ddec * dec * -ea * dtv;
                ddt[tr] += ddec * dec * -ea;
            }
        }
    }
    // C: the signals' projection and dt; then the 8-bit activations (straight-through), the conv and the norm
    real *du = w->du;
    memset(du, 0, sizeof(real) * (size_t)L * d);
    for (int t = 0; t < L; t++) {
        const real *pt = tp->p + (size_t)t * Pd, *xt = tp->x + (size_t)t * d;
        real *dpt = dp + (size_t)t * Pd, *dxt = dx + (size_t)t * d;
        for (int tr = 0; tr < T; tr++) {  // dt = softplus(raw + dt_bias)
            const real z = pt[2 * n + 2 * pv + tr] + dt_bias[tr];
            const real dz = dpt[2 * n + 2 * pv + tr] * (z > 20 ? 1 : sigm(z));
            dpt[2 * n + 2 * pv + tr] = dz;
            gdtb[tr] += dz;
        }
        for (int r = 0; r < Pd; r++) {
            axpy(gproj + (size_t)r * d, dpt[r], xt, d);
            qaxpy(k, pproj, Q->proj, r_proj + r, dxt, dpt[r], d, gs);
        }
        for (int j = 0; j < K; j++)
            if (t - j >= 0)
                for (int i = 0; i < d; i++) {
                    gconv[j * d + i] += dxt[i] * tp->u[(size_t)(t - j) * d + i];
                    du[(size_t)(t - j) * d + i] += conv[j * d + i] * dxt[i];
                }
    }
    for (int t = 0; t < L; t++) rmsnorm_bwd(tp->xin + (size_t)t * d, norm, tp->rn[t], du + (size_t)t * d, g + (size_t)t * d, gnorm, d);
}

Work *tss_work_new(const Cfg *c, int Lmax) {
    const int d = c->d, N = NODES(c), T = c->T, nl = c->n_layer;
    const size_t tr = tape_reals(c, Lmax), ti = (size_t)Lmax * T * (c->D + 1 + c->D * (c->D + 1) / 2);
    const size_t V = (size_t)Lmax * T * (c->D + 1), S = (size_t)T * N * c->n * c->p;
    Work *w = calloc(1, sizeof(Work));
    w->Lmax = Lmax;
    w->tapes = calloc(tr * nl, sizeof(real));
    w->itapes = calloc(ti * nl, sizeof(int));
    w->tp = calloc(nl, sizeof(Tape));
    w->x = calloc((size_t)Lmax * d, sizeof(real)), w->g = calloc((size_t)Lmax * d, sizeof(real));
    w->h = calloc(S, sizeof(real)), w->lam = calloc(S, sizeof(real));
    w->xn = calloc(d, sizeof(real)), w->dxn = calloc(d, sizeof(real)), w->lg = calloc(c->vocab, sizeof(real));
    w->ds = calloc(V, sizeof(real)), w->dR = calloc(V * c->p, sizeof(real));
    w->dp = calloc((size_t)Lmax * PDIM(c), sizeof(real));
    w->dx = calloc((size_t)Lmax * d, sizeof(real)), w->du = calloc((size_t)Lmax * d, sizeof(real));
    w->gUr = calloc(c->p, sizeof(real)), w->on = calloc(c->D + 1, sizeof(real)), w->Racc = calloc((size_t)T * c->p, sizeof(real));
    w->walk = calloc((size_t)T * (c->D + 1), sizeof(int));
    w->mx = calloc(d, sizeof(real)), w->y = calloc(d, sizeof(real));
    w->xq8 = calloc((size_t)nl * Lmax * d, 1);
    w->rcnt = calloc((size_t)T * N + 1, sizeof(int));
    w->rord = calloc((size_t)Lmax * T * (c->D + 1), sizeof(int));
    w->hp16 = calloc((size_t)nl * Lmax * T * (c->D + 1) * c->n * c->p, sizeof(uint16_t));
    for (int l = 0; l < nl; l++) {
        tape_carve(c, Lmax, w->tapes + tr * l, w->itapes + ti * l, w->tp + l);
        w->tp[l].xq = w->xq8 + (size_t)l * Lmax * d;
    }
    return w;
}

void tss_work_free(Work *w) {
    free(w->tapes), free(w->itapes), free(w->tp), free(w->x), free(w->g), free(w->h), free(w->lam), free(w->xn);
    free(w->dxn), free(w->lg), free(w->ds), free(w->dR), free(w->dp), free(w->dx), free(w->du), free(w->gUr);
    free(w->on), free(w->Racc), free(w->mx), free(w->y), free(w->xq8), free(w->hp16), free(w->rcnt), free(w->rord), free(w->walk), free(w);
}

void tss_work_bf16(Work *w, int on) { w->tape_bf16 = on; }

// One sequence: tokens (L), labels (L, -100 = unscored). Adds the sequence's summed cross-entropy gradients into G
// (Params-shaped, zeroed by the caller) and its node-row hits into hit (n_layer * T * N bytes). Returns the summed loss;
// *n_scored = the scored tokens. Q: the quantized weights (tss_quantize).
real tss_train_seq(const Cfg *c, const Params *P, const Params *Q, Params *G, const int *tokens, const int *labels, int L,
                   unsigned char *hit, int *n_scored, Work *w, const Codes *k) {
    const int d = c->d, N = NODES(c), T = c->T, nl = c->n_layer;
    if (L > w->Lmax) return NAN;
    w->k = k;
    Tape *tp = w->tp;  // carved for Lmax tokens: the strides are Lmax's, which L <= Lmax respects
    real *x = w->x, *h = w->h;
    for (int t = 0; t < L; t++) memcpy(x + (size_t)t * d, P->embed + (size_t)tokens[t] * d, sizeof(real) * d);
    for (int l = 0; l < nl; l++) {
        memset(h, 0, sizeof(real) * (size_t)T * N * c->n * c->p);
        layer_fwd(c, P, Q, l, L, x, h, tp + l, w);
    }
    // the output layer (tied to the embedding), cross-entropy at the scored tokens
    real loss = 0, *g = w->g, *xn = w->xn, *dxn = w->dxn, *lg = w->lg;
    memset(g, 0, sizeof(real) * (size_t)L * d);
    int ns = 0;
    for (int t = 0; t < L; t++) {
        if (labels[t] < 0) continue;
        real r;
        rmsnorm_fwd(x + (size_t)t * d, P->norm_f, xn, d, &r);
        real m = -INFINITY, z = 0;
        for (int v = 0; v < c->vocab; v++) lg[v] = dotr(P->embed + (size_t)v * d, xn, d), m = lg[v] > m ? lg[v] : m;
        for (int v = 0; v < c->vocab; v++) z += exp(lg[v] - m);
        loss += log(z) + m - lg[labels[t]];
        ns++;
        memset(dxn, 0, sizeof(real) * d);
        for (int v = 0; v < c->vocab; v++) {
            const real pr = exp(lg[v] - m) / z - (v == labels[t]);
            for (int i = 0; i < d; i++) {
                G->embed[(size_t)v * d + i] += pr * xn[i];
                dxn[i] += pr * P->embed[(size_t)v * d + i];
            }
        }
        rmsnorm_bwd(x + (size_t)t * d, P->norm_f, r, dxn, g + (size_t)t * d, G->norm_f, d);
    }
    real *lam = w->lam;
    for (int l = nl - 1; l >= 0; l--) {
        memset(lam, 0, sizeof(real) * (size_t)T * N * c->n * c->p);
        layer_bwd(c, P, Q, G, l, L, g, lam, hit + (size_t)l * T * N, tp + l, w);
    }
    for (int t = 0; t < L; t++)  // the embedding's input side
        for (int i = 0; i < d; i++) G->embed[(size_t)tokens[t] * d + i] += g[(size_t)t * d + i];
    *n_scored = ns;
    return loss;
}

// ----------------------------------------------------------------------------------------------- the training step
// A batch of sequences across threads, each into its own gradient buffer (Gt[i], hits[i], ws[i]); the touched node rows
// and the dense tensors summed into G; clipping by the global norm (over what was touched: the rest is zero); AdamW,
// lazy on node rows (a row moves only in a step that touched it, as sparse embeddings do), dense elsewhere; the ternary
// codes recomputed for the touched rows and the dense matrices. Returns the mean loss over the batch's scored tokens.

typedef struct {
    real lr, beta1, beta2, eps, weight_decay, clip;
    long step;  // the step being taken, from 1
} Hyper;

#define N_FIELDS 11
static void fields(const Params *p, real **f) {
    f[0] = p->embed, f[1] = p->norm_f, f[2] = p->norm, f[3] = p->conv, f[4] = p->w_in, f[5] = p->bias;
    f[6] = p->w_out, f[7] = p->A_log, f[8] = p->proj, f[9] = p->dt_bias, f[10] = p->U;
}

static void sizes(const Cfg *c, size_t *z) {
    const size_t nl = c->n_layer, d = c->d, rows = nl * c->T * NODES(c);
    z[0] = (size_t)c->vocab * d, z[1] = d, z[2] = nl * d, z[3] = nl * c->K * d, z[4] = rows * d, z[5] = rows;
    z[6] = rows * d, z[7] = rows, z[8] = nl * PDIM(c) * d, z[9] = nl * c->T, z[10] = nl * c->T * c->p * d;
}

static int node_field(int f) { return f == 4 || f == 5 || f == 6 || f == 7; }

static void adam(real *p, real *g, real *m, real *v, size_t n, const Hyper *h, real scale) {
    const real b1 = h->beta1, b2 = h->beta2;
    const real c1 = 1 - pow(b1, (real)h->step), c2 = 1 - pow(b2, (real)h->step);
    for (size_t i = 0; i < n; i++) {
        const real gi = g[i] * scale;
        m[i] = b1 * m[i] + (1 - b1) * gi;
        v[i] = b2 * v[i] + (1 - b2) * gi * gi;
        p[i] -= h->lr * (h->weight_decay * p[i] + (m[i] / c1) / (sqrt(v[i] / c2) + h->eps));
        g[i] = 0;
    }
}

real tss_train_step(const Cfg *c, Params *P, Params *Q, Params *G, unsigned char *hit, Params *M, Params *Vm,
                    Params *Gt, unsigned char *hits, Work **ws, int threads, const int *tokens, const int *labels, int B,
                    int L, const Hyper *h, Codes *k) {
    const size_t rows = (size_t)c->n_layer * c->T * NODES(c);
    const int d = c->d;
    real loss = 0;
    int scored = 0;
    if (threads > 64) return NAN;  // the threads' buffers are tabled for at most 64
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) reduction(+ : loss, scored)
    for (int b = 0; b < B; b++) {
#ifdef _OPENMP
        const int th = omp_get_thread_num();
#else
        const int th = 0;
#endif
        int ns = 0;
        loss += tss_train_seq(c, P, Q, Gt + th, tokens + (size_t)b * L, labels + (size_t)b * L, L, hits + rows * th, &ns, ws[th], k);
        scored += ns;
    }
    size_t z[N_FIELDS];
    sizes(c, z);
    real *gf[N_FIELDS], *pf[N_FIELDS], *mf[N_FIELDS], *vf[N_FIELDS];
    fields(G, gf), fields(P, pf), fields(M, mf), fields(Vm, vf);
    // sum the threads' buffers into G (touched rows only for the node tensors), zeroing them; every loop below is
    // parallel over rows or elements (at a large batch most rows are touched: the update is as dense as the model)
    const long R = (long)rows;
#pragma omp parallel for num_threads(threads) schedule(static)
    for (long r = 0; r < R; r++) {
        unsigned char any = 0;
        for (int th = 0; th < threads; th++) any |= hits[rows * th + r], hits[rows * th + r] = 0;
        hit[r] = any;
    }
    real *tf[64][N_FIELDS];
    for (int th = 0; th < threads && th < 64; th++) fields(Gt + th, tf[th]);
    for (int f = 0; f < N_FIELDS; f++) {
        if (node_field(f)) {
            const size_t w = z[f] / rows;  // d for w_in / w_out, 1 for bias / A_log
#pragma omp parallel for num_threads(threads) schedule(static)
            for (long r = 0; r < R; r++)
                if (hit[r])
                    for (int th = 0; th < threads; th++)
                        for (size_t i = 0; i < w; i++) gf[f][r * w + i] += tf[th][f][r * w + i], tf[th][f][r * w + i] = 0;
        } else {
            const long Z = (long)z[f];
#pragma omp parallel for num_threads(threads) schedule(static)
            for (long i = 0; i < Z; i++)
                for (int th = 0; th < threads; th++) gf[f][i] += tf[th][f][i], tf[th][f][i] = 0;
        }
    }
    // the mean over scored tokens, then the global norm's clip
    const real inv = scored ? (real)1 / scored : 0;
    double nrm = 0;
    for (int f = 0; f < N_FIELDS; f++) {
        if (node_field(f)) {
            const size_t w = z[f] / rows;
#pragma omp parallel for num_threads(threads) schedule(static) reduction(+ : nrm)
            for (long r = 0; r < R; r++)
                if (hit[r])
                    for (size_t i = 0; i < w; i++) nrm += (double)gf[f][r * w + i] * gf[f][r * w + i];
        } else {
            const long Z = (long)z[f];
#pragma omp parallel for num_threads(threads) schedule(static) reduction(+ : nrm)
            for (long i = 0; i < Z; i++) nrm += (double)gf[f][i] * gf[f][i];
        }
    }
    nrm = sqrt(nrm) * inv;
    real scale = inv;
    if (h->clip > 0 && nrm > h->clip) scale *= h->clip / (nrm + (real)1e-6);
    for (int f = 0; f < N_FIELDS; f++) {
        if (node_field(f)) {
            const size_t w = z[f] / rows;
#pragma omp parallel for num_threads(threads) schedule(static)
            for (long r = 0; r < R; r++)
                if (hit[r]) adam(pf[f] + r * w, gf[f] + r * w, mf[f] + r * w, vf[f] + r * w, w, h, scale);
        } else {
            const long Z = (long)z[f], chunk = 4096;
#pragma omp parallel for num_threads(threads) schedule(static)
            for (long i = 0; i < Z; i += chunk)
                adam(pf[f] + i, gf[f] + i, mf[f] + i, vf[f] + i, (size_t)(i + chunk < Z ? chunk : Z - i), h, scale);
        }
    }
    // the codes: the touched node rows, the dense matrices
    const int kok = k && k->ok;
#pragma omp parallel for num_threads(threads) schedule(static)
    for (long r = 0; r < R; r++)
        if (hit[r]) {
            tss_ternary(P->w_in + r * d, Q->w_in + r * d, 1, d, c->group);
            tss_ternary(P->w_out + r * d, Q->w_out + r * d, 1, d, c->group);
            if (kok) {
                plane_rows(Q->w_in, &k->w_in, r, r + 1, d, c->group);
                plane_rows(Q->w_out, &k->w_out, r, r + 1, d, c->group);
            }
        }
    const long pr = (long)c->n_layer * PDIM(c), ur = (long)c->n_layer * c->T * c->p;
#pragma omp parallel for num_threads(threads) schedule(static)
    for (long r = 0; r < pr + ur; r++) {
        real *Pm = r < pr ? P->proj : P->U, *Qm = r < pr ? Q->proj : Q->U;
        const long rr = r < pr ? r : r - pr;
        tss_ternary(Pm + rr * d, Qm + rr * d, 1, d, c->group);
        if (kok) plane_rows(Qm, r < pr ? &k->proj : &k->U, rr, rr + 1, d, c->group);
    }
    return scored ? loss / scored : 0;
}
"""


def _c_tensors(model):
    """The parameters in the C trainer's layout: name -> tensors, layer after layer."""
    L = model.layers
    return {"embed": [model.embed.weight], "norm_f": [model.norm_f.weight], "norm": [n.weight for n in model.norms],
            "conv": [l.conv for l in L], "w_in": [l.w_in for l in L], "bias": [l.bias for l in L], "w_out": [l.w_out for l in L],
            "A_log": [l.A_log for l in L], "proj": [l.proj.weight for l in L], "dt_bias": [l.dt_bias for l in L], "U": [l.U for l in L]}


class CTrainer:
    """The sparse C training step (C_TRAIN_SOURCE) on a model's own parameters: the model's tensors become views of
    the C arrays, so the PyTorch model evaluates whatever the trainer has trained, with nothing copied. Readout per tree,
    memory, ternary weights and 8-bit activations (the defaults). double=True builds in float64 (selftest --c)."""

    NAMES = ["embed", "norm_f", "norm", "conv", "w_in", "bias", "w_out", "A_log", "proj", "dt_bias", "U"]

    def __init__(self, model, threads=1, Lmax=4096, lr=3e-3, weight_decay=0.01, clip=1.0, double=False, fast=True,
                 tape_bf16=True):
        import ctypes

        import numpy as np

        c = model.cfg
        assert c.readout == "tree" and c.memory and c.ternary and c.act_bits == 8, "the C trainer: readout tree, ternary, 8-bit"
        self.np, self.ct = np, ctypes
        flags = ("-DREAL=double", "-DTSS_DOUBLE") if double else ()
        lib = self.lib = ctypes.CDLL(build_c(C_TRAIN_SOURCE, "train", flags))
        R = ctypes.c_double if double else ctypes.c_float
        self.dtype = np.float64 if double else np.float32
        PR = self.PR = ctypes.POINTER(R)

        class Cfg(ctypes.Structure):
            _fields_ = [(k, ctypes.c_int) for k in ("vocab", "d", "n_layer", "T", "D", "n", "p", "K", "group", "bdepth")] + [
                ("temp", R)]

        class Params(ctypes.Structure):
            _fields_ = [(k, PR) for k in self.NAMES]

        class Hyper(ctypes.Structure):
            _fields_ = [(k, R) for k in ("lr", "beta1", "beta2", "eps", "weight_decay", "clip")] + [("step", ctypes.c_long)]

        PP, P8, PI = ctypes.POINTER(Params), ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_int)
        lib.tss_work_new.restype = ctypes.c_void_p
        lib.tss_work_new.argtypes = [ctypes.POINTER(Cfg), ctypes.c_int]
        lib.tss_quantize.argtypes = [ctypes.POINTER(Cfg), PP, PP]
        lib.tss_train_seq.restype = R
        lib.tss_train_seq.argtypes = [ctypes.POINTER(Cfg), PP, PP, PP, PI, PI, ctypes.c_int, P8, PI, ctypes.c_void_p,
                                      ctypes.c_void_p]
        lib.tss_codes_new.restype = ctypes.c_void_p
        lib.tss_codes_new.argtypes = [ctypes.POINTER(Cfg)]
        lib.tss_codes_all.argtypes = [ctypes.POINTER(Cfg), PP, ctypes.c_void_p]
        lib.tss_train_step.restype = R
        lib.tss_train_step.argtypes = [ctypes.POINTER(Cfg), PP, PP, PP, P8, PP, PP, PP, P8, ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.c_int, PI, PI, ctypes.c_int, ctypes.c_int, ctypes.POINTER(Hyper), ctypes.c_void_p]
        self.cfg = Cfg(c.vocab_size, c.d_model, c.n_layer, c.trees, c.depth, c.state, c.value, c.d_conv,
                       group_size(c.d_model, c.ternary_group), c.branch_depth, c.temp)
        tensors = _c_tensors(model)
        self.arrays = {k: np.ascontiguousarray(torch.cat([t.detach().flatten() for t in v]).numpy().astype(self.dtype))
                       for k, v in tensors.items()}
        zeros = lambda: {k: np.zeros_like(v) for k, v in self.arrays.items()}  # noqa: E731
        mk = lambda d: Params(*[d[k].ctypes.data_as(PR) for k in self.NAMES])  # noqa: E731
        self.grads, self.q, self.m, self.v = zeros(), zeros(), zeros(), zeros()
        self.P, self.G, self.Q, self.M, self.V = mk(self.arrays), mk(self.grads), mk(self.q), mk(self.m), mk(self.v)
        lib.tss_quantize(ctypes.byref(self.cfg), ctypes.byref(self.P), ctypes.byref(self.Q))
        self.codes = lib.tss_codes_new(ctypes.byref(self.cfg)) if fast else None  # the rows' 2-bit planes (fast path)
        lib.tss_codes_all(ctypes.byref(self.cfg), ctypes.byref(self.Q), self.codes)
        rows = c.n_layer * c.trees * (2 ** (c.depth + 1) - 1)
        self.hit = np.zeros(rows, dtype=np.uint8)
        self.threads = threads
        self.gt = [zeros() for _ in range(threads)]
        self.Gt = (Params * threads)(*[mk(g) for g in self.gt])
        self.hits = np.zeros(threads * rows, dtype=np.uint8)
        self.ws = (ctypes.c_void_p * threads)(*[lib.tss_work_new(ctypes.byref(self.cfg), Lmax) for _ in range(threads)])
        lib.tss_work_bf16.argtypes = [ctypes.c_void_p, ctypes.c_int]
        for w in self.ws:  # the fast path's tape of previous states in bfloat16 (no effect elsewhere)
            lib.tss_work_bf16(w, int(tape_bf16))
        self.hp = Hyper(lr, 0.9, 0.98, 1e-8, weight_decay, clip, 0)
        for k, ts in tensors.items():  # the model's parameters become views of the C arrays
            off = 0
            for t in ts:
                t.data = torch.from_numpy(self.arrays[k][off : off + t.numel()]).view(t.shape)
                off += t.numel()

    def _ptr(self, a, kind):
        return a.ctypes.data_as(self.ct.POINTER(kind))

    def grad(self, tokens, labels):
        """One sequence's summed loss; its gradients added into self.grads (the C layout, not the model's .grad)."""
        tok = self.np.ascontiguousarray(tokens, dtype=self.np.int32)
        lab = self.np.ascontiguousarray(labels, dtype=self.np.int32)
        ns = self.ct.c_int()
        return self.lib.tss_train_seq(self.ct.byref(self.cfg), self.ct.byref(self.P), self.ct.byref(self.Q), self.ct.byref(self.G),
                                      self._ptr(tok, self.ct.c_int), self._ptr(lab, self.ct.c_int), len(tok),
                                      self._ptr(self.hit, self.ct.c_uint8), self.ct.byref(ns), self.ws[0], self.codes)

    def step(self, tokens, labels, lr):
        """One AdamW step on a batch, tokens and labels (B, L); returns the mean loss over the scored tokens."""
        B, L = tokens.shape
        tok = self.np.ascontiguousarray(tokens, dtype=self.np.int32)
        lab = self.np.ascontiguousarray(labels, dtype=self.np.int32)
        self.hp.lr, self.hp.step = lr, self.hp.step + 1
        b = self.ct.byref
        return self.lib.tss_train_step(b(self.cfg), b(self.P), b(self.Q), b(self.G), self._ptr(self.hit, self.ct.c_uint8),
                                       b(self.M), b(self.V), self.Gt, self._ptr(self.hits, self.ct.c_uint8), self.ws, self.threads,
                                       self._ptr(tok, self.ct.c_int), self._ptr(lab, self.ct.c_int), B, L, b(self.hp), self.codes)


def selftest_c():
    """The C training step against autograd, in float64: the loss and every parameter's gradient."""
    import numpy as np

    for bd in (0, 1, 2):  # the branch gradient all the way down, and capped
        torch.manual_seed(0)
        cfg = Config(vocab_size=50, d_model=32, n_layer=2, trees=3, depth=3, state=4, value=6, ternary_group=16, branch_depth=bd)
        model = TSS(cfg).double()
        for layer in model.layers:
            layer.w_in.data *= 4
        ids = torch.randint(0, 50, (2, 20))
        labels = ids.clone()
        labels[:, :5] = -100
        loss = F.cross_entropy(model(ids).flatten(0, 1), labels.flatten(), reduction="sum")
        loss.backward()
        want = {k: torch.cat([t.grad.flatten() for t in ts]).numpy() for k, ts in _c_tensors(model).items()}
        tr = CTrainer(model, Lmax=64, double=True)
        closs = sum(tr.grad(ids[b].numpy(), labels[b].numpy()) for b in range(2))
        assert abs(closs - loss.item()) < 1e-9 * abs(loss.item()), f"C loss {closs} != {loss.item()}"
        worst = max((np.abs(want[k] - tr.grads[k]).max() / (np.abs(want[k]).max() + 1e-30), k) for k in want)
        print(f"branch_depth {bd}: C training step vs autograd, float64: loss equal, worst gradient {worst[0]:.1e} ({worst[1]})")
        assert worst[0] < 1e-12, "the C training step's gradients disagree with autograd"
    print("selftest --c passed")


# ---------------------------------------------------------------------------------------------------------------- lm


def lm(a):
    """A byte-level language model on a text file (the last tenth held out): trained by the C sparse step (or PyTorch),
    evaluated on fixed held-out windows (cross-entropy in nats a byte, as nanoGPT's character-level Shakespeare), then
    sampled by the C inference step."""
    import numpy as np

    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    raw = np.frombuffer(open(a.data, "rb").read(), dtype=np.uint8)
    split = int(len(raw) * 0.9)
    train, held = torch.from_numpy(raw[:split].astype(np.int64)), torch.from_numpy(raw[split:].astype(np.int64))
    cfg = Config(vocab_size=256, d_model=a.d_model, n_layer=a.layers, trees=a.trees, depth=a.depth, state=a.state,
                 value=a.value, branch_depth=a.branch_depth)
    model = TSS(cfg)
    trainer = CTrainer(model, threads=a.threads, lr=a.lr, Lmax=a.seq_len) if a.engine == "c" else None
    opt = None if trainer else torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.01)
    print(f"tss lm ({a.engine}): {a.data} ({len(train):,} bytes to train on, {len(held):,} held out), "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, d {a.d_model}, {a.layers} layers x "
          f"{a.trees} trees of depth {a.depth}, branch gradient depth {a.branch_depth or 'full'}, batch {a.batch} x {a.seq_len}",
          flush=True)

    def windows(data, n, gen):
        starts = torch.randint(0, len(data) - a.seq_len - 1, (n,), generator=gen)
        chunk = data[starts[:, None] + torch.arange(a.seq_len + 1)]
        return chunk[:, :-1], chunk[:, 1:]

    val = windows(held, a.eval_windows, torch.Generator().manual_seed(1234))

    @torch.no_grad()
    def evaluate():
        x, y = val
        return sum(F.cross_entropy(model(x[i : i + 8]).flatten(0, 1), y[i : i + 8].flatten(), reduction="sum").item()
                   for i in range(0, len(x), 8)) / y.numel()

    gen, t0, seen = torch.Generator().manual_seed(a.seed), time.time(), 0
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        x, y = windows(train, a.batch, gen)
        if trainer:
            loss = trainer.step(x.numpy(), y.numpy(), lr)
        else:
            for g in opt.param_groups:
                g["lr"] = lr
            out = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
            opt.zero_grad(set_to_none=True)
            out.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss = out.item()
        seen += x.numel()
        if step % a.eval_every == 0 or step == a.steps:
            el = time.time() - t0
            v = evaluate()
            print(f"  step {step:5d}  train {loss:.3f}  held-out {v:.3f} nats/byte ({v / math.log(2):.2f} bits)  "
                  f"{seen / 1e6:.1f}M tokens  {el / 60:.1f} min  {seen / el:,.0f} tokens/s", flush=True)
    if a.sample:  # the trained model, sampled by the C inference step
        st = c_step(model)
        state, logits = st.new_state(), np.zeros(256, dtype=np.float32)
        rng = np.random.default_rng(a.seed)
        out = list(b"\n")
        st.run(np.array(out, dtype=np.int64), state, logits)
        for _ in range(a.sample):
            p = np.exp((logits - logits.max()) / a.temperature)
            nxt = int(rng.choice(256, p=p / p.sum()))
            out.append(nxt)
            st(nxt, state, logits)
        print("--- sample (C inference step, temperature %.1f) ---" % a.temperature)
        print(bytes(out).decode("utf-8", "replace"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("selftest")
    st.add_argument("--kernels", action="store_true", help="the Triton kernels against the PyTorch paths")
    st.add_argument("--c", action="store_true", help="the sparse C training step against autograd (float64)")
    m = sub.add_parser("mqar")
    m.add_argument("--no-memory", dest="memory", action="store_false")
    m.add_argument("--kv", type=int, default=8)
    m.add_argument("--keys", type=int, default=32)
    m.add_argument("--d-model", type=int, default=64)
    m.add_argument("--layers", type=int, default=2)
    m.add_argument("--trees", type=int, default=4)
    m.add_argument("--depth", type=int, default=5)
    m.add_argument("--state", type=int, default=16)
    m.add_argument("--value", type=int, default=16)
    m.add_argument("--readout", default="tree", choices=["level", "tree", "layer"])
    m.add_argument("--float", dest="quant", action="store_false", help="float weights and activations (the ablation)")
    m.add_argument("--engine", default="torch", choices=["torch", "c"], help="c: the sparse C training step (CPU)")
    m.add_argument("--steps", type=int, default=2000)
    m.add_argument("--batch", type=int, default=64)
    m.add_argument("--lr", type=float, default=3e-3)
    m.add_argument("--eval-every", type=int, default=200)
    m.add_argument("--threads", type=int, default=4)
    m.add_argument("--seed", type=int, default=0)
    lmp = sub.add_parser("lm", help="a byte-level language model on a text file")
    lmp.add_argument("--data", required=True)
    lmp.add_argument("--engine", default="c", choices=["torch", "c"])
    lmp.add_argument("--d-model", type=int, default=128)
    lmp.add_argument("--layers", type=int, default=4)
    lmp.add_argument("--trees", type=int, default=4)
    lmp.add_argument("--depth", type=int, default=7)
    lmp.add_argument("--branch-depth", type=int, default=0, help="cap the branch gradient's depth (0: full)")
    lmp.add_argument("--state", type=int, default=16)
    lmp.add_argument("--value", type=int, default=16)
    lmp.add_argument("--steps", type=int, default=3000)
    lmp.add_argument("--batch", type=int, default=16)
    lmp.add_argument("--seq-len", type=int, default=256)
    lmp.add_argument("--lr", type=float, default=3e-3)
    lmp.add_argument("--warmup", type=int, default=100)
    lmp.add_argument("--eval-every", type=int, default=250)
    lmp.add_argument("--eval-windows", type=int, default=64)
    lmp.add_argument("--threads", type=int, default=4)
    lmp.add_argument("--seed", type=int, default=0)
    lmp.add_argument("--sample", type=int, default=600, help="bytes to sample at the end (0: none)")
    lmp.add_argument("--temperature", type=float, default=0.8)
    bn = sub.add_parser("bench")
    bn.add_argument("--vocab", type=int, default=256)
    bn.add_argument("--d-model", type=int, default=256)
    bn.add_argument("--layers", type=int, default=6)
    bn.add_argument("--trees", type=int, default=4)
    bn.add_argument("--depth", type=int, default=9)
    bn.add_argument("--state", type=int, default=16)
    bn.add_argument("--value", type=int, default=16)
    bn.add_argument("--readout", default="tree", choices=["level", "tree", "layer"])
    bn.add_argument("--tokens", type=int, default=20000)
    a = p.parse_args()
    {"selftest": lambda a: selftest_kernels() if a.kernels else selftest_c() if a.c else selftest(), "mqar": mqar,
     "lm": lm, "bench": bench}[a.cmd](a)


if __name__ == "__main__":
    main()
