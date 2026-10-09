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

Two forms of one function:
* ``step``: the recurrence, a token at a time. What runs at inference: O(1) memory in the sequence length, O(depth)
  work per tree, branches and gathers.
* ``forward``: the training form. Levels run in turn and each level is parallel over time: routing at level k
  depends only on what was written to level-k nodes, i.e. on decisions at levels < k, so there is no circularity.
  Within a level, the read is a masked, decayed sum over earlier tokens at the same node (quadratic in length here).
Routing trains straight-through: the hard step going forward, sigmoid(s / temp) of the chosen side going backward,
along the path. ``selftest`` checks that the two forms agree exactly, routes included, in float64.

Inference in C (tss_step.c): the same recurrence, one token at a time, touching only the visited rows and states.

    python tss.py selftest                # training form == recurrence, float64, routes included
    python tss.py mqar [--no-memory] [--depth 0] [--readout level|tree|layer]   # associative recall, on a CPU
    python tss.py bench                   # the C step against the PyTorch recurrence, then tokens/s on one core

Results so far (one seed each, 2 layers, d 64, 4 trees, node state 16 x 16; MQAR with 8 pairs of 32 keys, 2,000
steps of 64 sequences, 2 CPU threads, ~7 minutes):
    stateless trees                         0.03 recall (chance)
    depth 0 (one state per tree: a Mamba-2 head)   0.53
    depth 5, readout per level / tree / layer      0.992 / 0.991 / 0.991
A first version whose nodes held a decayed sum of values (no <C, B> matching inside a node) reached only 0.20: it
learned "the answer is one of the values seen", but the route alone could not address the right one.
Speed (bench: d 256, 6 layers, 4 trees of depth 9, 13.8M parameters, vocab 256; one core, batch 1, float32):
    stateless 56 us/token; stateful, readout per level 250, per tree 127, per layer 121.
What remains is memory-bound: each token gathers ~240 random weight rows (1 KB each in float32) and ~240 node states.
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

    def U_at(self, k):
        """The readout of level k's reads, (trees, value, d); shared by levels and trees as cfg.readout says."""
        U = self.U[:, k if self.cfg.readout == "level" else 0]
        return U.expand(self.cfg.trees, -1, -1)

    def mix(self, u):
        """Causal depthwise conv: x[t] = sum_j conv[j] * u[t - j]."""
        L = u.shape[1]
        up = F.pad(u, (0, 0, self.cfg.d_conv - 1, 0))
        return sum(self.conv[j] * up[:, self.cfg.d_conv - 1 - j : self.cfg.d_conv - 1 - j + L] for j in range(self.cfg.d_conv))

    def signals(self, x):
        """B, C (b, l, state); V, Z (b, l, value); dt (b, l, trees)."""
        n, p = self.cfg.state, self.cfg.value
        B, C, V, Z, dt = self.proj(x).split([n, n, p, p, self.cfg.trees], -1)
        return B, C, V, Z, F.softplus(dt + self.dt_bias)

    def forward(self, u):
        """The training form. u: (b, l, d)."""
        cfg = self.cfg
        b, L, _ = u.shape
        T = cfg.trees
        x = self.mix(u)
        tree = torch.arange(T, device=u.device)
        node = torch.zeros(b, L, T, dtype=torch.long, device=u.device)
        P = torch.ones(b, L, T, dtype=u.dtype, device=u.device)  # straight-through path weight: 1 going forward
        y = torch.zeros_like(u)
        if cfg.memory:
            B, C, V, Z, dt = self.signals(x)
            CB = torch.einsum("btn,bsn->bts", C, B)
            causal = torch.ones(L, L, dtype=torch.bool, device=u.device).tril()[None, :, :, None]  # s <= t
            strict = causal & ~torch.eye(L, dtype=torch.bool, device=u.device)[None, :, :, None]
        for k in range(cfg.depth + 1):
            s = torch.einsum("bld,bltd->blt", x, self.w_in[tree, node]) + self.bias[tree, node]
            if cfg.memory:
                same = node[:, :, None, :] == node[:, None, :, :]  # (b, t, s, trees): visited the same node
                clock = torch.einsum("btsh,bsh->bth", (same & causal).to(dt.dtype), dt)  # dt summed over visits <= t
                lam = torch.exp(self.A_log[tree, node])  # the node's own rate, (b, t, trees)
                gap = (clock[:, :, None] - clock[:, None, :]) * lam[:, :, None]  # decay exponent from s to t
                M = torch.exp(-gap.masked_fill(~(same & strict), torch.inf)) * dt[:, None] * CB[..., None]
                R = torch.einsum("btsh,bsp->bthp", M, V)  # what each token reads, (b, l, trees, value)
                s = s + (R * Z[:, :, None]).sum(-1)
                y = y + torch.einsum("blh,blhp,hpd->bld", P, R, self.U_at(k))
            y = y + torch.einsum("blt,bltd->bld", P * F.gelu(s), self.w_out[tree, node])
            right = s > 0
            p = torch.sigmoid(torch.where(right, s, -s) / cfg.temp)
            P = P * (1 + p - p.detach())
            node = 2 * node + 1 + right.long()
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
        x = (self.conv * hist).sum(1)
        st["conv"] = hist[:, :-1]
        bi, ti = torch.arange(b)[:, None], torch.arange(T)[None, :]
        node = torch.zeros(b, T, dtype=torch.long)
        y = torch.zeros_like(u)
        if cfg.memory:
            B, C, V, Z, dt = self.signals(x)
            BV = B[:, None, :, None] * V[:, None, None, :]  # (b, 1, state, value)
            h = st["h"]
        for k in range(cfg.depth + 1):
            s = (x[:, None] * self.w_in[ti, node]).sum(-1) + self.bias[ti, node]
            if cfg.memory:
                ha = h[bi, ti, node] * torch.exp(-torch.exp(self.A_log[ti, node]) * dt)[..., None, None]  # decay on arrival
                R = torch.einsum("bn,bhnp->bhp", C, ha)  # read
                s = s + (R * Z[:, None]).sum(-1)
                y = y + torch.einsum("bhp,hpd->bd", R, self.U_at(k))
                h[bi, ti, node] = ha + dt[..., None, None] * BV  # then write
            y = y + torch.einsum("bh,bhd->bd", F.gelu(s), self.w_out[ti, node])
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

    def forward(self, ids):
        x = self.embed(ids)
        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))
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
    """The training form against the recurrence, token at a time, in float64: logits and every route."""
    torch.manual_seed(0)
    for memory, readout in ((True, "level"), (True, "tree"), (True, "layer"), (False, "level")):
        cfg = Config(vocab_size=50, d_model=32, n_layer=2, trees=3, depth=4, state=8, memory=memory, readout=readout)
        model = TSS(cfg).double()
        for layer in model.layers:  # spread the logits so both branches get traffic
            layer.w_in.data *= 4
        ids = torch.randint(0, 50, (3, 40))
        par = model(ids)
        states = model.init_state(3, dtype=torch.float64)
        seq = torch.stack([model.step(ids[:, t], states) for t in range(40)], 1)
        err = (par - seq).abs().max().item()
        assert err < 1e-10, f"memory={memory}: training form != recurrence ({err:.1e})"
        par.sum().backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
        assert not missing, f"no gradient: {missing}"
    print("selftest passed: training form == token-at-a-time recurrence (stateful and stateless); every parameter learns")


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
                 state=a.state, value=a.value, memory=a.memory, readout=a.readout)
    model = TSS(cfg)
    print(f"tss mqar: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}K parameters, memory={a.memory}, "
          f"depth {a.depth}, readout {a.readout}, {a.kv} pairs of {a.keys} keys, length {3 * a.kv}", flush=True)
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
    """tss_step.c built and loaded, with the model's weights packed for it. Returns step(token, h, hist, logits)."""
    import ctypes
    import os
    import subprocess

    import numpy as np

    here = os.path.dirname(os.path.abspath(__file__))
    so = os.path.join(here, "tss_step.so")
    src = os.path.join(here, "tss_step.c")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        subprocess.check_call(["cc", "-O3", "-march=native", "-ffast-math", "-shared", "-fPIC", "-o", so, src, "-lm"])
    lib = ctypes.CDLL(so)
    cfg, f = model.cfg, ctypes.POINTER(ctypes.c_float)

    class Model(ctypes.Structure):
        _fields_ = [(k, ctypes.c_int) for k in ("vocab", "d", "n_layer", "trees", "depth", "state", "value", "d_conv", "memory",
                                                    "readout")] + [
            (k, f) for k in ("embed", "norm_f", "norm", "conv", "w_in", "bias", "w_out", "proj", "dt_bias", "neg_A", "U")]

    def cat(fn):
        return np.ascontiguousarray(torch.cat([fn(i, l).detach().float().flatten() for i, l in enumerate(model.layers)]).numpy())

    keep = {"embed": model.embed.weight.detach().float().numpy().copy(), "norm_f": model.norm_f.weight.detach().float().numpy().copy(),
            "norm": cat(lambda i, l: model.norms[i].weight), "conv": cat(lambda i, l: l.conv), "w_in": cat(lambda i, l: l.w_in),
            "bias": cat(lambda i, l: l.bias), "w_out": cat(lambda i, l: l.w_out)}
    if cfg.memory:
        keep.update(proj=cat(lambda i, l: l.proj.weight), dt_bias=cat(lambda i, l: l.dt_bias),
                    neg_A=cat(lambda i, l: -torch.exp(l.A_log)), U=cat(lambda i, l: l.U))
    m = Model(cfg.vocab_size, cfg.d_model, cfg.n_layer, cfg.trees, cfg.depth, cfg.state, cfg.value, cfg.d_conv,
              int(cfg.memory), ["level", "tree", "layer"].index(cfg.readout),
              *[keep[k].ctypes.data_as(f) if k in keep else None for k in ("embed", "norm_f", "norm", "conv", "w_in", "bias",
                                                                             "w_out", "proj", "dt_bias", "neg_A", "U")])
    N = 2 ** (cfg.depth + 1) - 1
    scratch = np.zeros(4 * cfg.d_model + 2 * cfg.state + (3 + cfg.trees) * cfg.value + cfg.trees, dtype=np.float32)
    lib.tss_step.argtypes = [ctypes.POINTER(Model), f, f, ctypes.c_int, f, f]

    def new_state():
        return (np.zeros(cfg.n_layer * cfg.trees * N * cfg.state * cfg.value, dtype=np.float32),
                np.zeros(cfg.n_layer * max(cfg.d_conv - 1, 1) * cfg.d_model, dtype=np.float32))

    def step(token, state, logits):
        lib.tss_step(ctypes.byref(m), state[0].ctypes.data_as(f), state[1].ctypes.data_as(f), int(token),
                     logits.ctypes.data_as(f), scratch.ctypes.data_as(f))

    lib.tss_run.argtypes = [ctypes.POINTER(Model), f, f, ctypes.POINTER(ctypes.c_int), ctypes.c_int, f, f]

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
                 value=a.value, memory=a.memory, readout=a.readout)
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
          f"memory={cfg.memory}")
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
    m.add_argument("--steps", type=int, default=2000)
    m.add_argument("--batch", type=int, default=64)
    m.add_argument("--lr", type=float, default=3e-3)
    m.add_argument("--eval-every", type=int, default=200)
    m.add_argument("--threads", type=int, default=4)
    m.add_argument("--seed", type=int, default=0)
    bn = sub.add_parser("bench")
    bn.add_argument("--no-memory", dest="memory", action="store_false")
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
