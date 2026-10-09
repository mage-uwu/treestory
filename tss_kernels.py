"""The tree state space's training kernels: the segmented scan and the grouped per-node products, in PyTorch and in
Triton (on a GPU, or in Triton's interpreter with TRITON_INTERPRET=1). tss.py's training form calls ``segscan``,
``group_dot`` and ``group_outer``; each runs its Triton kernels when ``use_triton(tensor)`` says so, else PyTorch, and
both compute the same values and gradients (``selftest`` here checks them against each other and the definitions).

The scan. Over one stream sorted so that each segment (one node of one tree of one sequence) is contiguous and in time
order, with h the node's state, a sum of K V^T decayed by the entries' log-decays a:

    Y_i  = sum over j < i in i's segment of exp(a_{j+1} + .. + a_i) (Q_i . K_j) V_j     = Q_i^T h       (value-sized)
    Y2_i = sum over the same j of           exp(a_{j+1} + .. + a_i) (V2_i . V_j) K_j    = h V2_i        (key-sized)

Forward is Y with Q = C, K = B, V = X. The backward pass is the same two reads: dC = Y2 of the forward scan read with
V2 = dY; dX and dB = Y and Y2 of the reverse scan (keys C, values dY) read with Q = B and V2 = X; da is a segmented
suffix sum of <dY, Y> - <X, dX>. Chunks of CHUNK entries: within a chunk a masked quadratic form, across chunks the
state each chunk hands on, which is the same scan one level up (the chunks as entries, K = Q = 1, the states as values).
A chunk's threads share one node's tokens (one decay rate; at a segment boundary a mask, not a branch) and read
contiguous memory.

The grouped products. Within a level the entries are also sorted by node (tree, node, sequence, time), so each node's
tokens are contiguous: blocks of BLOCK rows aligned to the nodes (as a mixture of experts' dispatch) load their node's
weight row once. s = <row, W[node]>, out = coef W[node]; the weights' gradients accumulate per block, atomically.
"""

import os

import torch

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
