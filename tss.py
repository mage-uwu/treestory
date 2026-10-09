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
  Within a level, the read is a masked, decayed sum over earlier tokens at the same node (quadratic in length here);
  the branch gradient's other side reads the same way at the nodes a token did not visit.
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
Speed (bench: d 256, 6 layers, 4 trees of depth 9, 12.9M parameters, vocab 256; one core of a 2.1 GHz Xeon, batch 1):
    float 127 us/token, ternary 115 (readout per tree). Time grows with the nodes visited (depth + 1 per tree), not
    with the nodes stored; here everything fits in the 260 MB L3, so the naive loops are the cost, not memory. The C
    codes are one int8 a weight for now (2-bit packing, SIMD and interleaving the trees' dependent fetches are next).
"""

import argparse
import math
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
        self.record, self.stats = False, None
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
        """The training form, u: (b, l, d). Levels in turn, each parallel over time. Going backward each branch learns
        from GTS's straight-through gradient: sign * (on - alt) * sigmoid'(s / temp) / temp, where on is what the
        subtree the token took output below the branch and alt what the other subtree would have, following the
        token's own decisions and reading the states it would have found there (``reference`` is the definition)."""
        cfg = self.cfg
        b, L, _ = u.shape
        T, D = cfg.trees, cfg.depth
        x = self.mix(u)
        w_in, w_out, proj, U = self.weights()
        tree = torch.arange(T, device=u.device)
        mem = None
        if self.record:
            self.stats = [x.detach()]  # the trees' input, then per level: nodes, gelu(s), reads
        if cfg.memory:
            B, C, V, Z, dt = self.signals(x, proj)
            strict = torch.ones(L, L, dtype=torch.bool, device=u.device).tril(-1)[None, :, :, None]  # s < t
            mem = (torch.einsum("btn,bsn->bts", C, B), V, Z, dt, strict)

        def visit(node, k, visitors=None):
            """Each token at node (b, l, trees) of level k: its activation s and its node output (b, l, trees, d).
            visitors: (nodes, inclusive clocks) of the tokens that did visit level k, if not these."""
            s = torch.einsum("bld,bltd->blt", x, w_in[tree, node]) + self.bias[tree, node]
            out = F.gelu(s)[..., None] * w_out[tree, node]
            clock = R = None
            if mem is not None:
                CB, V, Z, dt, strict = mem
                vnode, vclock = visitors if visitors else (node, None)
                same = (node[:, :, None, :] == vnode[:, None, :, :]) & strict  # (b, t, s, trees): s was there before t
                clock = torch.einsum("btsh,bsh->bth", same.to(dt.dtype), dt) + dt  # dt summed over arrivals there, t's own included
                vclock = clock if vclock is None else vclock
                gap = (clock[:, :, None] - vclock[:, None, :]) * torch.exp(self.A_log[tree, node])[:, :, None]
                M = torch.exp(-gap.masked_fill(~same, torch.inf)) * dt[:, None] * CB[..., None]
                R = torch.einsum("btsh,bsp->bthp", M, V)  # what each token reads there, (b, l, trees, value)
                s = s + (R * Z[:, :, None]).sum(-1)
                out = F.gelu(s)[..., None] * w_out[tree, node] + torch.einsum("blhp,hpd->blhd", R, self.U_at(U, k))
            return s, out, clock, R

        node = torch.zeros(b, L, T, dtype=torch.long, device=u.device)
        path = []  # per level: node, s, output, inclusive clock
        for k in range(D + 1):
            s, out, clock, R = visit(node, k)
            path.append((node, s, out, clock))
            if self.record:  # for an optimizer that needs per-node statistics (deepboost.py)
                if s.requires_grad:
                    s.retain_grad()  # the split's pseudo-residual, dL/ds, per token
                self.stats.append((node, F.gelu(s).detach(), None if R is None else R.detach(), s))
            node = 2 * node + 1 + (s > 0).long()
        y = sum(out for _, _, out, _ in path).sum(2)
        if not torch.is_grad_enabled() or D == 0:
            return self._out(y)
        with torch.no_grad():  # the branch gradient's two sides: on (taken, below the branch) and alt (the other child down)
            outs = torch.stack([out for _, _, out, _ in path])
            on = outs.flip(0).cumsum(0).flip(0)  # on[k] = sum of levels >= k
            diff = []
            for k in range(D):
                a, s_k = path[k][0], path[k][1]
                alt_node, alt = 2 * a + 2 - (s_k > 0).long(), 0  # the other child
                for j in range(k + 1, D + 1):
                    s_alt, out_alt = visit(alt_node, j, (path[j][0], path[j][3]))[:2]
                    alt = alt + out_alt
                    alt_node = 2 * alt_node + 1 + (s_alt > 0).long()
                diff.append(on[k + 1] - alt)
        for k in range(D):
            s_k = path[k][1]
            g = torch.sigmoid(torch.where(s_k > 0, s_k, -s_k) / cfg.temp)
            y = y + torch.einsum("blt,bltd->bld", g - g.detach(), diff[k])  # zero going forward
        return self._out(y)

    def _out(self, y):
        if self.record and y.requires_grad:
            y.retain_grad()  # the value's pseudo-residual, dL/dy, per token
            self.out = y
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
    print("selftest passed: training form == recurrence (routes included) and its gradients == the path-weight definition "
          "(ternary and float; stateful and stateless; every readout)")


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


def c_step(model):
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
        subprocess.check_call(["cc", "-O3", "-march=native", "-ffast-math", "-shared", "-fPIC", "-o", so, src, "-lm"])
    lib = ctypes.CDLL(so)
    f, i8 = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int8)
    g = group_size(cfg.d_model, cfg.ternary_group)

    class Ternary(ctypes.Structure):
        _fields_ = [("codes", i8), ("scales", f)]

    class Model(ctypes.Structure):
        _fields_ = [(k, ctypes.c_int) for k in ("vocab", "d", "n_layer", "trees", "depth", "state", "value", "d_conv",
                                                "memory", "readout", "group")] + [
            ("embed", f), ("norm_f", f), ("norm", f), ("conv", f), ("w_in", Ternary), ("bias", f), ("w_out", Ternary),
            ("proj", Ternary), ("dt_bias", f), ("neg_A", f), ("U", Ternary)]

    keep = []

    def floats(fn):
        a = np.ascontiguousarray(torch.cat([fn(l).detach().float().flatten() for l in model.layers]).numpy())
        keep.append(a)
        return a.ctypes.data_as(f)

    def tern(fn):  # the codes and scales ``ternary`` computes, over every layer
        codes, scales = [], []
        for l in model.layers:
            w = fn(l).detach().double().reshape(-1, cfg.d_model)
            wg = w.reshape(w.shape[0], -1, g)
            sc = wg.abs().mean(-1, keepdim=True).clamp(min=1e-8)
            codes.append((wg / sc).clamp(-1, 1).round().to(torch.int8).flatten())
            scales.append(sc.float().flatten())
        c, sc = np.ascontiguousarray(torch.cat(codes).numpy()), np.ascontiguousarray(torch.cat(scales).numpy())
        keep.extend([c, sc])
        return Ternary(c.ctypes.data_as(i8), sc.ctypes.data_as(f))

    embed = np.ascontiguousarray(model.embed.weight.detach().float().numpy())
    norm_f = np.ascontiguousarray(model.norm_f.weight.detach().float().numpy())
    keep.extend([embed, norm_f])
    m = Model(cfg.vocab_size, cfg.d_model, cfg.n_layer, cfg.trees, cfg.depth, cfg.state, cfg.value, cfg.d_conv, 1,
              ["level", "tree", "layer"].index(cfg.readout), g, embed.ctypes.data_as(f), norm_f.ctypes.data_as(f),
              floats(lambda l: model.norms[list(model.layers).index(l)].weight), floats(lambda l: l.conv),
              tern(lambda l: l.w_in), floats(lambda l: l.bias), tern(lambda l: l.w_out), tern(lambda l: l.proj.weight),
              floats(lambda l: l.dt_bias), floats(lambda l: -torch.exp(l.A_log)), tern(lambda l: l.U))
    N = 2 ** (cfg.depth + 1) - 1
    scratch = np.zeros(5 * cfg.d_model + 2 * cfg.state + (3 + cfg.trees) * cfg.value + cfg.trees, dtype=np.float32)
    lib.tss_step.argtypes = [ctypes.POINTER(Model), f, f, ctypes.c_int, f, f]
    lib.tss_run.argtypes = [ctypes.POINTER(Model), f, f, ctypes.POINTER(ctypes.c_int), ctypes.c_int, f, f]

    def new_state():
        return (np.zeros(cfg.n_layer * cfg.trees * N * cfg.state * cfg.value, dtype=np.float32),
                np.zeros(cfg.n_layer * max(cfg.d_conv - 1, 1) * cfg.d_model, dtype=np.float32))

    def step(token, state, logits):
        lib.tss_step(ctypes.byref(m), state[0].ctypes.data_as(f), state[1].ctypes.data_as(f), int(token),
                     logits.ctypes.data_as(f), scratch.ctypes.data_as(f))

    def run(tokens, state, logits):
        tok = np.ascontiguousarray(tokens, dtype=np.int32)
        lib.tss_run(ctypes.byref(m), state[0].ctypes.data_as(f), state[1].ctypes.data_as(f),
                    tok.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(tok), logits.ctypes.data_as(f), scratch.ctypes.data_as(f))

    step.keep, step.new_state, step.run = (keep, m), new_state, run
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
    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, a.tokens)
    t0 = time.perf_counter()
    step.run(tokens, cs, logits)  # the loop in C: no Python per token
    c_rate = a.tokens / (time.perf_counter() - t0)
    t0 = time.perf_counter()
    for t in range(200):
        model.step(torch.tensor([t % cfg.vocab_size]), states)
    py_rate = 200 / (time.perf_counter() - t0)
    print(f"  one core, batch 1: C {c_rate:,.0f} tokens/s ({1e6 / c_rate:.1f} us/token); PyTorch step {py_rate:,.0f} tokens/s")


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
