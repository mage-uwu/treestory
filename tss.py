"""Tree state space: a language model whose only mixer is a forest of binary trees, and whose trees are the state.

There is no SSM bank, no attention and no MLP. Every node of every tree is a state cell: a small key -> value matrix
h_a (state x value). A token walks one root-to-leaf path per tree; at each node it visits it

    decays the node's state        h_a <- exp(-lambda_a * dt_t) * h_a          (only on a visit: no clock, no bookkeeping)
    reads it by content            R = C_t^T h_a                                (values of earlier tokens routed here,
                                                                                 weighted by <C_t, B_s>)
    computes the node's activation s = <x_t, w_a> + b_a + <Z_t, R>             (memory moves the branch)
    branches                       right if s > 0
    outputs                        gelu(s) * o_a + U_k R                         (U_k: one readout per tree and level)
    writes                         h_a <- h_a + dt_t * B_t V_t^T

and touches nothing else. This is a selective SSM taken to the extreme: dt is zero for every state cell but the
O(depth) ones the tree selects, so the routing is the selection. The route is a coarse address (a bucket) and <C, B>
matches within it. The root is visited by every token (a global, decaying context, i.e. one Mamba-2 head); a level-k
node only by tokens that made the same k decisions. The state grows exponentially with depth (nodes x state x value
per tree), the work per token linearly (depth + 1 nodes per tree, each one state x value read and write).

The efficiency rests on two choices carried over from GTS, and the trees depend on both:
* ternary weights. w_in, w_out, proj and U are absmean codes in {-1, 0, 1} times one scale per group of 128,
  recomputed every step from latent float weights (straight-through), and the trees' input is per-token 8-bit
  integers. A visited row is 2 bits a weight (a 256-wide row in one 64-byte cache line) and its dot product an integer
  sum of +-x: few rows per token, each tiny, no multiplications.
* the branch gradient. A branch learns from the difference between its two subtrees: what the subtree the token took
  output below it, minus what the other one would have, following the token's own decisions and reading the states
  it would have found there (GTS's straight-through gradient, with state). That is how a query learns to go where its
  key was written; a gradient along the taken path alone only says whether that path helped.

Two forms of one function:
* ``step``: the recurrence, a token at a time. What runs at inference: O(1) memory in the sequence length, O(depth)
  work per tree, branches and gathers.
* ``forward``: the training form. Levels run in turn and each level is parallel over time: routing at level k
  depends only on what was written to level-k nodes, i.e. on decisions at levels < k, so there is no circularity.
  Within a level the tokens are sorted by node, so that each node's visitors are contiguous and in time order, and
  the whole level is one segmented linear scan (``segscan``: Mamba-2's chunked scan, the segments' resets as masks),
  linear in length; the branch gradient's other side enters the same scan as read-only entries (no write, no decay
  passed on). This is the shape a GPU wants: after the sort, the threads of a chunk work on one node's tokens, one
  weight row and one decay rate (no divergence), over contiguous memory (coalesced); a level is a sort, gathers and
  batched small matrix products, as a mixture-of-experts dispatch. Training cost per token is flat in length
  (2 layers, d 64, 4 trees of depth 5, 4 CPU threads: ~240 us/token from 64 to 4,096 tokens, against a quadratic
  form's 1.5 s a step at 1,024).
``selftest`` checks, in float64, that the training form equals the recurrence (routes included) and that its
gradients equal those of ``reference``, the path-weight definition over every node.

Inference in C (tss_step.c): the same recurrence on the ternary codes and int8 activations, one token at a time,
touching only the visited rows and states.

    python tss.py selftest
    python tss.py mqar [--float] [--no-memory] [--depth 0] [--readout level|tree|layer]   # associative recall, CPU
    python tss.py bench                   # the C step against the PyTorch recurrence, then tokens/s on one core

Results (one seed each; 2 layers, d 64, 4 trees, node state 16 x 16; MQAR with 8 pairs of 32 keys, 2,000 steps of 64
sequences on 2 CPU threads; recall accuracy at steps 600 / 800 / 2,000):
    stateless trees                                         0.03 (chance)
    depth 0 (one state per tree: a Mamba-2 head), float     - / - / 0.53
    depth 5, float, a gradient along the taken path only    0.20 / 0.83 / 0.991
    depth 5, float, the branch gradient                     0.93 / 0.98 / 0.995
    depth 5, ternary + 8-bit + the branch gradient          0.50 / 0.80 / 0.983
Nodes holding a decayed sum of values (no <C, B> matching within a node) reached only 0.20: the route alone could not
address the right value. The readout of the reads per level, per tree or per layer made no difference to recall.
Speed (bench: d 256, 6 layers, 4 trees of depth 9, 12.9M parameters, vocab 256; a 2.1 GHz Xeon, AVX-512, batch 1):
    plain C 120 us/token; AVX2 (int8 codes, sign-and-add) 78; 2-bit (two bitmasks a row, masked add/sub) 56.
    Several sequences across cores, 2-bit: 64K tokens/s for 4 sequences on 4 cores; 64 sequences on 4 cores 48K tokens/s
    with bfloat16 node states, 11K with float32 (25 MB of state a sequence no longer fits in cache).
The node state, not the weights, is what a stateful tree has to move: 1 KB a visit in float32 against 128 bytes of
weights. Every kernel agrees with the others to float32 rounding; against PyTorch, now and then a branch whose
activation is within rounding of zero turns the other way (summation order), as in GTS.
"""

import argparse
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        """The training form, u: (b, l, d). Levels in turn; within a level, every node's state over time is one
        segmented scan (``segscan``) over the level's tokens sorted by node: linear in length. Going backward each
        branch learns from GTS's straight-through gradient: sign * (on - alt) * sigmoid'(s / temp) / temp, where on is
        what the subtree the token took output below the branch and alt what the other subtree would have, following
        the token's own decisions and reading the states it would have found there (``reference`` is the definition).
        Those reads are the same scan's read-only entries."""
        cfg = self.cfg
        b, L, _ = u.shape
        T, D = cfg.trees, cfg.depth
        x = self.mix(u)
        w_in, w_out, proj, U = self.weights()
        tree = torch.arange(T, device=u.device)
        if cfg.memory:
            B, C, V, Z, dt = self.signals(x, proj)

        def reads(node, queries=()):
            """What each token at node (b, l, trees) of a level reads there (its visitors before it, then its own
            write), and for each node tensor in queries, what that token would read there (no write)."""
            if not cfg.memory:
                return None, [None] * len(queries)
            M, n_nodes = b * L * T, self.w_in.shape[1]
            bi = torch.arange(b, device=u.device).view(b, 1, 1).expand(b, L, T).reshape(-1)
            li = torch.arange(L, device=u.device).view(1, L, 1).expand(b, L, T).reshape(-1)
            ti = tree.view(1, 1, T).expand(b, L, T).reshape(-1)
            nodes = torch.cat([node.reshape(-1)] + [q.reshape(-1) for q in queries])
            k = len(queries) + 1
            bi, li, ti = bi.repeat(k), li.repeat(k), ti.repeat(k)
            member = torch.arange(k * M, device=u.device) < M
            seg = (bi * T + ti) * n_nodes + nodes  # a segment: one node of one tree of one sequence
            order = torch.argsort(seg * L + li)  # contiguous segments, in time within each
            dt_e = dt[bi, li, ti]
            log_decay = -torch.exp(self.A_log[ti, nodes]) * dt_e
            zero = torch.zeros((), dtype=dt.dtype, device=u.device)
            a_e = torch.where(member, log_decay, zero)  # a query decays nothing that comes after it
            X_e = torch.where(member[:, None], dt_e[:, None] * V[bi, li], zero)
            Y = segscan(C[bi, li][order], B[bi, li][order], X_e[order], a_e[order], seg[order])
            Y = torch.zeros_like(Y).index_copy(0, order, Y)
            Y = torch.where(member[:, None], Y, Y * torch.exp(log_decay)[:, None])  # a query's own arrival decay
            R = Y[:M].view(b, L, T, -1)
            return R, [Y[M * (i + 1) : M * (i + 2)].view(b, L, T, -1) for i in range(len(queries))]

        def visit(node, k, R):
            """Each token at node (b, l, trees) of level k, having read R there: its activation s and output."""
            s = torch.einsum("bld,bltd->blt", x, w_in[tree, node]) + self.bias[tree, node]
            if R is None:
                return s, F.gelu(s)[..., None] * w_out[tree, node]
            s = s + (R * Z[:, :, None]).sum(-1)
            return s, F.gelu(s)[..., None] * w_out[tree, node] + torch.einsum("blhp,hpd->blhd", R, self.U_at(U, k))

        node = torch.zeros(b, L, T, dtype=torch.long, device=u.device)
        path = []  # per level: node, s, output
        for k in range(D + 1):
            s, out = visit(node, k, reads(node)[0])
            path.append((node, s, out))
            node = 2 * node + 1 + (s > 0).long()
        y = sum(out for _, _, out in path).sum(2)
        if not torch.is_grad_enabled() or D == 0:
            return y
        with torch.no_grad():  # the branch gradient's two sides: on (taken, below the branch) and alt (the other child down)
            on = torch.stack([out for _, _, out in path]).flip(0).cumsum(0).flip(0)  # on[k] = sum of levels >= k
            chains = []  # per branch level k: the other subtree's node on the current level, and its output so far
            for j in range(1, D + 1):
                a, s_prev = path[j - 1][0], path[j - 1][1]
                chains.append([2 * a + 2 - (s_prev > 0).long(), 0])  # the other child of level j - 1's branch
                _, alt_reads = reads(path[j][0], [c[0] for c in chains])  # the level's visitors, the chains as queries
                for c, R in zip(chains, alt_reads):
                    s_alt, out_alt = visit(c[0], j, R)
                    c[1] = c[1] + out_alt
                    c[0] = 2 * c[0] + 1 + (s_alt > 0).long()
        for k in range(D):
            s_k = path[k][1]
            g = torch.sigmoid(torch.where(s_k > 0, s_k, -s_k) / cfg.temp)
            y = y + torch.einsum("blt,bltd->bld", g - g.detach(), on[k + 1] - chains[k][1])  # zero going forward
        return y

    def reference(self, u):
        """The definition, for selftest: every node of every tree for every token, with its read (what the token would
        find there: the states its visitors wrote before it), times its path weight, the product of the branch values
        above it (the hard step going forward, sigmoid(s / temp) going backward), as GTS's path-weight definition."""
        cfg = self.cfg
        b, L, _ = u.shape
        T, D = cfg.trees, cfg.depth
        x = self.mix(u)
        w_in, w_out, proj, U = self.weights()
        if cfg.memory:
            B, C, V, Z, dt = self.signals(x, proj)
            CB = torch.einsum("btn,bsn->bts", C, B)
        node = torch.zeros(b, L, T, dtype=torch.long)
        pi = torch.ones(b, L, T, 1, dtype=u.dtype)
        y = torch.zeros_like(u)
        for k in range(D + 1):
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
            pi = torch.stack([pi * (1 - g), pi * g], -1).flatten(3)  # children 2i+1, 2i+2
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


def segscan(C, B, X, a, seg, chunk=64):
    """Y_i = sum over j < i with seg_j = seg_i of exp(a_{j+1} + ... + a_i) <C_i, B_j> X_j, over one stream sorted so
    that each segment is contiguous: C, B (M, state); X (M, value); a (M,) log-decays <= 0; seg (M,) ids. Linear in M:
    chunks of ``chunk`` entries, each read within its chunk as a masked quadratic form plus the state carried in from
    the chunks before (Mamba-2's chunked scan, with the segments' resets as masks rather than as log-decays of -inf,
    which would cost float32 its precision). On a GPU each chunk is one program: the same node's tokens in a row
    (one weight row, one decay rate, no divergence) and contiguous loads."""
    M, Q = X.shape[0], chunk
    pad = -M % Q
    if pad:
        z = lambda t: torch.cat([t, t.new_zeros((pad,) + t.shape[1:])])  # noqa: E731
        C, B, X, a = z(C), z(B), z(X), z(a)
        seg = torch.cat([seg, -1 - torch.arange(pad, device=seg.device)])  # padding: segments of its own
    nc = (M + pad) // Q
    C, B, X, a, seg = C.view(nc, Q, -1), B.view(nc, Q, -1), X.view(nc, Q, -1), a.view(nc, Q), seg.view(nc, Q)
    cs = a.cumsum(1)  # inclusive, within the chunk
    strict = torch.ones(Q, Q, dtype=torch.bool, device=a.device).tril(-1)
    same = (seg[:, :, None] == seg[:, None, :]) & strict
    W = torch.exp((cs[:, :, None] - cs[:, None, :]).masked_fill(~same, -torch.inf)) * (C @ B.transpose(1, 2))
    Y = W @ X
    # the state each chunk hands on: that of the segment open at its end
    last = seg[:, -1]
    S = torch.einsum("cj,cjn,cjp->cnp", torch.exp(cs[:, -1:] - cs) * (seg == last[:, None]), B, X)
    # H[c] = sum over c' < c with last[c'] = last[c - 1] of exp(the decays of chunks c' + 1 .. c - 1) S[c']: the state
    # entering chunk c. Segment ids ascend along the stream, so this is the same scan one level up, over the chunks
    # (C = B = 1, the states as values): G[c] = scan + S[c] inclusive, H[c] = G[c - 1]. Few chunks: one masked matrix.
    A = cs[:, -1]
    if nc > Q:
        Sf = S.reshape(nc, -1)
        one = Sf.new_ones(nc, 1)
        G = segscan(one, one, Sf, A, last, chunk) + Sf
    else:
        same_c = (last[:, None] == last[None, :]) & torch.ones(nc, nc, dtype=torch.bool, device=a.device).tril()
        Gc = A.cumsum(0)
        G = (torch.exp((Gc[:, None] - Gc[None, :]).masked_fill(~same_c, -torch.inf)) @ S.reshape(nc, -1))
    H = torch.cat([G.new_zeros(1, G.shape[1]), G[:-1]]).view_as(S)
    reader = seg == torch.cat([last.new_full((1,), -2), last[:-1]])[:, None]  # in the segment that was carried in
    Y = Y + (reader * torch.exp(cs))[..., None] * (C @ H)
    return Y.reshape(nc * Q, -1)[:M]


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


def selftest():
    """In float64: the training form against the token-at-a-time recurrence (logits, routes included), and its
    gradients, the branch gradient's counterfactual side included, against the path-weight definition."""
    torch.manual_seed(0)
    for memory, readout, quant in ((True, "level", True), (True, "tree", True), (True, "layer", False), (False, "tree", True)):
        cfg = Config(vocab_size=50, d_model=32, n_layer=2, trees=3, depth=3, state=4, value=6, memory=memory,
                     readout=readout, ternary=quant, act_bits=8 if quant else 0, ternary_group=16)
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


# -------------------------------------------------------------------------------------------------------------- mqar


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
    print(f"tss mqar: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}K parameters, memory={a.memory}, "
          f"depth {a.depth}, readout {a.readout}, {'ternary' if a.quant else 'float'}, {a.kv} pairs of {a.keys} keys, length {3 * a.kv}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.01)
    gen, t0 = torch.Generator().manual_seed(a.seed), time.time()
    val = mqar_batch(512, a.kv, a.keys, torch.Generator().manual_seed(1234))
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, step / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        ids, labels = mqar_batch(a.batch, a.kv, a.keys, gen)
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


# --------------------------------------------------------------------------------------------------- C inference


def c_step(model, state_bf16=False):
    """tss_step.c built and loaded, with the model packed for it: ternary weights as int8 codes and one scale per
    group, as the forward pass uses them. Returns step(token, state, logits), with .run and .new_state."""
    import ctypes
    import os
    import subprocess

    import numpy as np

    cfg = model.cfg
    assert cfg.ternary and cfg.act_bits == 8 and cfg.memory, "the C step runs the ternary, 8-bit, stateful model"
    here = os.path.dirname(os.path.abspath(__file__))
    so, src = os.path.join(here, "tss_step.so"), os.path.join(here, "tss_step.c")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        subprocess.check_call(["cc", "-O3", "-march=native", "-ffast-math", "-fopenmp", "-shared", "-fPIC", "-o", so, src, "-lm"])
    lib = ctypes.CDLL(so)
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

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
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
    m.add_argument("--steps", type=int, default=2000)
    m.add_argument("--batch", type=int, default=64)
    m.add_argument("--lr", type=float, default=3e-3)
    m.add_argument("--eval-every", type=int, default=200)
    m.add_argument("--threads", type=int, default=4)
    m.add_argument("--seed", type=int, default=0)
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
    {"selftest": lambda a: selftest(), "mqar": mqar, "bench": bench}[a.cmd](a)


if __name__ == "__main__":
    main()
