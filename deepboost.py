"""DeepBoost: gradient boosting's update rules, applied to a tree model's nodes inside backpropagation.

The correspondence. Gradient boosting fits each new weak learner to the pseudo-residuals, the negative gradient of the
loss at each example, by least squares; XGBoost sets a leaf's value to -G / (H + lambda), the sums of the examples'
gradients and curvatures in that leaf. In a tree model every node is a weak learner over the tokens that reach it,
with a split function s = <x, w_in> + b and a value w_out (and each tree's readout U of its memory reads), and
backpropagation hands every node its pseudo-residuals: -dL/dy at the value, -dL/ds at the split (the branch gradient,
taken subtree minus the other one, included). So, per batch, after one forward and backward pass:

    values   w_out[a] -= eta * G_a / (H_a + lambda),   G_a = sum coef g (the gradient), H_a = sum coef^2 over the
             tokens at a                                                  (XGBoost's leaf, unit curvature)
    splits   [w_in; b][a] -= eta_split * (sum x x^T + lambda I)^-1 sum x g_s   (the least-squares fit of the node's
             pseudo-residuals onto its inputs: a boosting round for the split function, the tokens at a only)
    readouts U_t -= eta * (sum R R^T + lambda I)^-1 sum R g              (the same fit, the tree's reads as inputs)
    the rest (embeddings, norms, conv, proj, decays): Adam

The sums are running averages over batches (as XGBoost's histograms, accumulated per node, sparsely: a token adds
only to the nodes it visited); lambda is relative to their mean scale; eta is boosting's shrinkage. A node visited
rarely gets a full step, where SGD would scale it down by its traffic.

    python deepboost.py mqar --opt adam      --lr 3e-3
    python deepboost.py mqar --opt deepboost --lr 3e-3 --eta 0.1
    python deepboost.py lm   --opt deepboost ...       # bytes of the Python standard library's source

Shoot-out (one seed each; 2 layers, d 64, 4 trees of depth 5, ternary; 1,000 steps, warmup then cosine; 2 CPU threads)
MQAR, recall at steps 400 / 600 / 1,000 (batch 64; 8 pairs of 32 keys):
    Adam 1e-3 / 3e-3 / 1e-2                          0.17 / 0.18 / 0.19,  0.20 / 0.20 / 0.21,  0.91 / 0.97 / 0.986
    DeepBoost, eta 0.1, eta_split 0.3, Adam 1e-2     0.87 / 0.96 / 0.983  (the best of 7 settings at Adam 1e-2)
    DeepBoost, any setting, Adam 3e-3                0.19 - 0.20 at 1,000 (5 settings)
    DeepBoost --fisher (4 settings of eta, lambda)   0.03 - 0.33 at 1,000
LM, held-out bits per byte at steps 200 / 600 / 1,000 (batch 32 x 64 bytes):
    Adam 1e-2                                        3.27 / 2.90 / 2.826     (Adam 3e-3: 2.988)
    DeepBoost, eta 0.1, eta_split 0.3, Adam 1e-2     3.20 / 2.92 / 2.854     (Adam 3e-3: 2.970)
Seconds per step: MQAR Adam 0.33, DeepBoost 0.40; LM Adam 0.55, DeepBoost 0.64 (--fisher about twice that).
So far it is a tie at best: at Adam's best rate Adam ends slightly ahead on both tasks; DeepBoost leads early on the
LM and ends ahead at the lower rate. Whether recall takes off is set by the rate of the dense parameters, not by the
node updates. The Fisher curvature (XGBoost's h) makes it worse: early in training it is tiny and the Newton steps
too large, and damping enough to stop that leaves too little step.
"""

import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tss  # noqa: E402


class DeepBoost:
    def __init__(self, model, lr=3e-3, eta=0.1, eta_split=1.0, lam=0.1, beta=0.95, weight_decay=0.01, fisher=False):
        self.layers = list(model.layers)
        for layer in self.layers:
            layer.record = True
            assert layer.cfg.readout in ("tree", "layer"), "readout per level: not done"
        self.boosted = {id(p) for layer in self.layers for p in (layer.w_in, layer.bias, layer.w_out)}
        self.boosted |= {id(layer.U) for layer in self.layers if layer.cfg.memory}
        rest = [p for p in model.parameters() if id(p) not in self.boosted]
        self.adam = torch.optim.AdamW(rest, lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)
        # shrinkage for values and readouts, and for splits: with unit curvature the two residuals differ in scale
        self.param_groups = self.adam.param_groups + [{"lr": eta}, {"lr": eta_split}]
        self.lam, self.beta, self.t, self.fisher = lam, beta, 0, fisher
        self.curv = None
        self.ema = [{} for _ in self.layers]

    def curvature(self, logits, labels):
        """XGBoost's h per token, at every node's split and every layer's output: the Fisher, from one backward pass of
        labels sampled from the model's own predictions (its expected squared gradients are the Gauss-Newton
        curvature). Call between the forward pass and the real backward pass."""
        with torch.no_grad():
            sel = labels != -100
            self.n_scored = int(sel.sum())  # the loss is a mean over these: its curvature too
            sampled = torch.multinomial(F.softmax(logits[sel].float(), -1), 1).squeeze(-1)
        loss = F.cross_entropy(logits[sel], sampled, reduction="sum")  # per token: h = g^2 of its own loss
        loss.backward(retain_graph=True)
        self.curv = [(layer.out.grad.pow(2), [s.grad.pow(2) for *_, s in layer.stats[1:]]) for layer in self.layers]
        for layer in self.layers:
            layer.out.grad = None
            for *_, s in layer.stats[1:]:
                s.grad = None
        self.zero_grad()

    def zero_grad(self, set_to_none=True):
        self.adam.zero_grad(set_to_none=set_to_none)
        for layer in self.layers:
            for p in (layer.w_in, layer.bias, layer.w_out) + ((layer.U,) if layer.cfg.memory else ()):
                p.grad = None

    def _avg(self, store, key, value):
        """Running average with Adam's bias correction."""
        prev = store.get(key)
        store[key] = value if prev is None else self.beta * prev + (1 - self.beta) * value
        return store[key] / (1 - self.beta**self.t) if prev is not None else value

    @staticmethod
    def _damped(A, lam):
        """A (.., k, k) + lam * (the mean diagonal over the blocks that have data) I."""
        diag = A.diagonal(dim1=-2, dim2=-1)
        scale = diag[diag.sum(-1) > 0].mean() if (diag.sum(-1) > 0).any() else diag.new_tensor(1.0)
        return A + (lam * scale + 1e-12) * torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)

    @torch.no_grad()
    def step(self):
        self.t += 1
        eta, eta_split = self.param_groups[-2]["lr"], self.param_groups[-1]["lr"]
        for layer, store in zip(self.layers, self.ema):
            cfg = layer.cfg
            T, d = cfg.trees, cfg.d_model
            N = layer.w_in.shape[1]
            x, *levels = layer.stats
            n_tok = self.n_scored if self.fisher else x.shape[0] * x.shape[1]
            xt = torch.cat([x, torch.ones_like(x[..., :1])], -1)  # the split's inputs, with its bias
            H = torch.zeros(T * N, dtype=x.dtype)
            A = torch.zeros(T * N, d + 1, d + 1, dtype=x.dtype)
            outer = (xt[..., :, None] * xt[..., None, :]).flatten(0, 1)  # (tokens, d + 1, d + 1)
            tree = torch.arange(T) * N
            Rsum = 0
            hy, hs = self.curv[self.layers.index(layer)] if self.fisher else (None, [None] * len(levels))
            if self.fisher:
                H = torch.zeros(T * N, d, dtype=x.dtype)  # per coordinate: diag of the Gauss-Newton
                hy = hy.flatten(0, 1)  # (tokens, d)
            for (node, coef, R, _), h_s in zip(levels, hs):
                idx = (node + tree).flatten(0, 1)  # (tokens, trees): flat node index
                c2 = coef.flatten(0, 1) ** 2
                for t in range(T):  # each tree adds its tokens' statistics at their nodes
                    if self.fisher:
                        H.index_add_(0, idx[:, t], c2[:, t, None] * hy)
                        A.index_add_(0, idx[:, t], h_s.flatten(0, 1)[:, t, None, None] * outer)
                    else:
                        H.index_add_(0, idx[:, t], c2[:, t])
                        A.index_add_(0, idx[:, t], outer)
                if R is not None:
                    Rsum = Rsum + R
            H = self._avg(store, "H", H / n_tok)
            A = self._avg(store, "A", A / n_tok)
            # values: XGBoost's leaf, -G / (H + lambda)
            w_out = layer.w_out.view(T * N, d)
            scale = H[H > 0].mean() if (H > 0).any() else H.new_tensor(1.0)
            Hd = H if self.fisher else H[:, None]
            w_out -= eta * layer.w_out.grad.view(T * N, d) / (Hd + self.lam * scale + 1e-12)
            # splits: the least-squares fit of each node's pseudo-residuals onto its inputs
            G = torch.cat([layer.w_in.grad.view(T * N, d), layer.bias.grad.view(T * N, 1)], -1)
            step = torch.linalg.solve(self._damped(A, self.lam), G.unsqueeze(-1)).squeeze(-1)
            layer.w_in.view(T * N, d).sub_(eta_split * step[:, :d])
            layer.bias.view(T * N).sub_(eta_split * step[:, d])
            # readouts: the same fit, the reads as inputs
            if cfg.memory:
                if cfg.readout == "layer":
                    Rsum = Rsum.sum(2, keepdim=True)
                hR = hy.view(*Rsum.shape[:2], 1, d).mean(-1, keepdim=True) if self.fisher else 1.0  # h per token
                AU = self._avg(store, "AU", torch.einsum("blhp,blhq->hpq", Rsum * hR, Rsum) / n_tok)
                U = layer.U.view(-1, cfg.value, d)
                U -= eta * torch.linalg.solve(self._damped(AU, self.lam), layer.U.grad.view_as(U))
        self.adam.step()


# ----------------------------------------------------------------------------------------------------------- tasks


def lm_data():
    """Bytes of the Python standard library's own source: 4 MB to train on, the last 256 KB held out."""
    files = sorted(glob.glob(os.path.join(os.path.dirname(os.__file__), "**", "*.py"), recursive=True))
    data = np.frombuffer(b"".join(open(f, "rb").read() for f in files), dtype=np.uint8)
    assert len(data) > 5_000_000, "not enough source"
    return torch.from_numpy(data[:4_000_000].astype(np.int64)), torch.from_numpy(data[-262_144:].astype(np.int64))


def lm_batch(data, b, L, gen):
    starts = torch.randint(0, len(data) - L - 1, (b,), generator=gen)
    idx = starts[:, None] + torch.arange(L + 1)
    chunk = data[idx]
    return chunk[:, :-1], chunk[:, 1:]


def run(a):
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    if a.task == "mqar":
        cfg = tss.Config(vocab_size=2 * a.keys, d_model=a.d_model, n_layer=a.layers, trees=a.trees, depth=a.depth)
        val = tss.mqar_batch(512, a.kv, a.keys, torch.Generator().manual_seed(1234))
        batch = lambda gen: tss.mqar_batch(a.batch, a.kv, a.keys, gen)  # noqa: E731
    else:
        cfg = tss.Config(vocab_size=256, d_model=a.d_model, n_layer=a.layers, trees=a.trees, depth=a.depth)
        train, held = lm_data()
        vg = torch.Generator().manual_seed(1234)
        val = [lm_batch(held, 32, a.seq_len, vg) for _ in range(8)]
        batch = lambda gen: lm_batch(train, a.batch, a.seq_len, gen)  # noqa: E731
    model = tss.TSS(cfg)
    if a.opt == "adam":
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.01)
    else:
        opt = DeepBoost(model, lr=a.lr, eta=a.eta, eta_split=a.eta_split, lam=a.lam, fisher=a.fisher)
    base = [g["lr"] for g in opt.param_groups]
    gen, t0, curve = torch.Generator().manual_seed(a.seed), time.time(), []

    @torch.no_grad()
    def evaluate():
        model.eval()
        for layer in model.layers:
            layer.record = False
        if a.task == "mqar":
            ids, labels = val
            pred, sel = model(ids).argmax(-1), labels != -100
            m = (pred[sel] == labels[sel]).float().mean().item()
        else:
            m = sum(F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item() for x, y in val) / len(val) / math.log(2)
        for layer in model.layers:
            layer.record = a.opt == "deepboost"
        model.train()
        return m

    name = "recall" if a.task == "mqar" else "val bits/byte"
    print(f"{a.task} {a.opt}: lr {a.lr}" + (f", eta {a.eta}, eta_split {a.eta_split}, lambda {a.lam}, fisher {a.fisher}" if a.opt == "deepboost" else "")
          + f", {sum(p.numel() for p in model.parameters()) / 1e3:.0f}K parameters", flush=True)
    for step in range(1, a.steps + 1):
        mult = min(1.0, step / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        for g, lr in zip(opt.param_groups, base):
            g["lr"] = lr * mult
        x, y = batch(gen)
        logits = model(x).flatten(0, 1)
        loss = F.cross_entropy(logits, y.flatten())
        opt.zero_grad(set_to_none=True)
        if a.opt == "deepboost" and a.fisher:
            opt.curvature(logits, y.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            m = evaluate()
            curve.append((step, m))
            print(f"  step {step:5d}  loss {loss.item():.4f}  {name} {m:.4f}  {time.time() - t0:.0f}s", flush=True)
    print(f"RESULT {a.task} {a.opt} lr={a.lr} eta={a.eta} eta_split={a.eta_split} lam={a.lam} fisher={a.fisher} final={curve[-1][1]:.4f} "
          f"s_per_step={(time.time() - t0) / a.steps:.3f} curve={curve}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task", choices=["mqar", "lm"])
    p.add_argument("--opt", choices=["adam", "deepboost"], required=True)
    p.add_argument("--lr", type=float, default=3e-3, help="Adam's rate (DeepBoost: for the parameters it does not boost)")
    p.add_argument("--eta", type=float, default=0.1, help="DeepBoost's shrinkage")
    p.add_argument("--eta-split", type=float, default=1.0, help="DeepBoost's shrinkage for the splits")
    p.add_argument("--fisher", action="store_true", help="DeepBoost with XGBoost's h: the sampled Fisher per token")
    p.add_argument("--lam", type=float, default=0.1, help="DeepBoost's damping, relative to the statistics' scale")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--kv", type=int, default=8)
    p.add_argument("--keys", type=int, default=32)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--trees", type=int, default=4)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args())


if __name__ == "__main__":
    main()
