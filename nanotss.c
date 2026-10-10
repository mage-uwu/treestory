// nanotss.c: the tree state space in one C file. A byte-level language model whose only mixer is a forest of binary
// trees whose nodes are its state, trained and run on a CPU: no Python, no framework.
//
//   cc -O3 -march=native -ffast-math -fopenmp -o nanotss nanotss.c -lm
//   ./nanotss --check                                  # the generator computes what training trains; training learns
//   ./nanotss --data input.txt --save model.bin        # train (held-out loss as it goes), then sample
//   ./nanotss --load model.bin --sample 600 --prompt "ROMEO:"
//
// The model: byte embedding (tied to the output layer) -> n_layer x [x + TreeSSM(RMSNorm(x))] -> RMSNorm -> logits.
// A TreeSSM layer: a causal depthwise conv (width 3) to 8-bit activations; a ternary projection to B, C, V, Z, dt; T
// binary trees of depth D whose every node holds a small state (state x value). A token walks one path per tree; at
// each node it decays the node's state (on a visit only), reads it by content (C^T h), lets the read move the branch
// (s = <x, w_in> + bias + <Z, R>), outputs gelu(s) w_out, and writes dt B V^T. Each tree's reads, summed along its path,
// go out through a ternary readout U. Weights w_in, w_out, proj, U are ternary (absmean codes and one scale per group
// of 128, recomputed from float latent weights, straight-through); on AVX-512 they also live as 2-bit planes, so a
// visited row is a cache line and its products are masked integer and float adds.
//
// Training is sparse: the forward walk records a tape; the backward touches what the forward touched (the visited
// rows, the visited nodes' states as adjoints run backward in time) and, for the branch gradient (GTS's straight-
// through gradient: the subtree taken minus the other subtree, followed down by the token's own decisions), walks
// through the other subtrees, sampled (--walks m per tree and token, reweighted: unbiased; 0 = every one, exact).
// AdamW, lazy on node rows, parallel over threads. This file's model and training core is nanotss.py's C trainer
// (C_TRAIN_SOURCE), whose gradients equal PyTorch autograd's (float64); nanotss.py also holds the training form in
// PyTorch, its definition, the GPU kernels and the checks. Tiny Shakespeare with the defaults (1.16M parameters,
// 5,000 steps, 4 threads): held-out ~1.60 nats a byte in ~10 minutes.
//
// ---------------------------------------------------------------------------------------------------------------------
// Tree state space: the training step on a CPU, sparse. One sequence's loss and gradients, the same function and the
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
    int walks;  // 0: every branch's walk through the other subtree (exact); m > 0: m in expectation, sampled and
                // reweighted (an unbiased estimate of the branch gradient)
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
    real *bw;                           // (L, T, D + 1): 1 / q for a walked branch, 0 for one not walked
} Tape;

static size_t tape_reals(const Cfg *c, int L) {
    const size_t V = (size_t)L * c->T * (c->D + 1), A = (size_t)L * c->T * c->D * (c->D + 1) / 2;
    return (size_t)L * (3 * c->d + 2 + PDIM(c) + c->T) + V * (1 + c->p + c->n * c->p + 1 + 1) + A * (1 + c->p);
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
    tp->bw = buf, buf += V;
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
    uint64_t rng;    // the walks' sampler (cfg walks > 0)
    int no_alt;      // forward only (evaluation): skip the branch gradient's walks
} Work;

static inline real urand(uint64_t *s) {  // xorshift64*, uniform in [0, 1)
    *s ^= *s >> 12, *s ^= *s << 25, *s ^= *s >> 27;
    return (real)((*s * 2685821657736338717ULL) >> 11) * (real)(1.0 / 9007199254740992.0);
}

void tss_work_seed(Work *w, uint64_t seed) { w->rng = seed * 0x9E3779B97F4A7C15ULL + 1; }

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
        // nodes are stored at base(kk) + (level - kk - 1), base(kk) = sum over m < kk of (D - m).
        int *cn = w->walk;
        for (int tr = 0; tr < T && !w->no_alt; tr++) {
            const size_t vb = ((size_t)t * T + tr) * (D + 1);
            real tot = 0;
            for (int kk = 0; kk < D; kk++) {
                const size_t v = vb + kk;
                cn[tr * (D + 1) + kk] = 2 * (tp->node[v] - tr * N) + 2 - (tp->s[v] > 0);
                const real sg = sigm(fabs(tp->s[v]) / c->temp);
                tp->bw[v] = sg * (1 - sg);  // sigma': the branch term's weight
                tot += tp->bw[v];
            }
            for (int kk = 0; kk < D; kk++) {  // walk branch kk with probability q = min(1, m sigma' / sum sigma')
                const size_t v = vb + kk;
                if (c->walks <= 0) {
                    tp->bw[v] = 1;
                    continue;
                }
                const real q = tot > 0 ? fmin(1, c->walks * tp->bw[v] / tot) : 1;
                tp->bw[v] = urand(&w->rng) < q ? 1 / q : 0;
            }
        }
        for (int j = 1; j <= D && !w->no_alt; j++)
            for (int tr = 0; tr < T; tr++) {
                const size_t ab = ((size_t)t * T + tr) * AL;
                for (int kk = 0, base = 0; kk < j; base += D - kk, kk++) {
                    if (tp->bw[((size_t)t * T + tr) * (D + 1) + kk] == 0) continue;  // a branch not walked this time
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
                    if (j < D) {  // the next node's state and row, while the other chains run
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
        const real *gy = g + (size_t)t * d, *pt = tp->p + (size_t)t * Pd;
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
            for (int kk = 0; kk < D; kk++) {  // the branch: sign * (on - alt) * sigmoid'(s / temp) / temp
                const real bwk = tp->bw[vb + kk];
                if (bwk == 0) {  // not walked: no term (its expectation is carried by the walked ones' weights)
                    ai += D - kk;
                    continue;
                }
                real onk = 0, alt = 0;
                for (int j = kk + 1; j <= D; j++) onk += on[j];
                for (int j = kk + 1; j <= D; j++, ai++) {
                    const size_t a = ab + ai, row = tp->anode[a];
                    alt += gelu(tp->as[a]) * qdotf(k, pout, Q->w_out, r_node + row, gy, d, gs) + dotr(gUr, tp->aR + a * pv, pv);
                }
                const real s = tp->s[vb + kk], sg = sigm(fabs(s) / c->temp);
                ds[vb + kk] += bwk * (s > 0 ? 1 : -1) * (onk - alt) * sg * (1 - sg) / c->temp;
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
        tss_work_seed(ws[th], (uint64_t)h->step * 1000003ULL + (uint64_t)b);
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

// =====================================================================================================================
// The standalone program: a model's parameters and their initialization, a byte-level text corpus, the training loop
// (tss_train_step above: sparse forward and backward, AdamW), held-out evaluation (forward only), generation (the
// recurrence a token at a time), checkpoints, and a self-check that the generator computes what training trained.
// =====================================================================================================================
#include <stdio.h>
#include <time.h>

static uint64_t g_rng = 88172645463325252ULL;
static double rand01(void) {
    g_rng ^= g_rng >> 12, g_rng ^= g_rng << 25, g_rng ^= g_rng >> 27;
    return (double)((g_rng * 2685821657736338717ULL) >> 11) * (1.0 / 9007199254740992.0);
}
static double randn(void) {  // Box-Muller
    double u = rand01(), v = rand01();
    if (u < 1e-300) u = 1e-300;
    return sqrt(-2 * log(u)) * cos(6.283185307179586 * v);
}
static double wall(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void params_alloc(const Cfg *c, Params *P) {
    size_t z[N_FIELDS];
    sizes(c, z);
    real *f[N_FIELDS];
    for (int i = 0; i < N_FIELDS; i++) f[i] = calloc(z[i] ? z[i] : 1, sizeof(real));
    P->embed = f[0], P->norm_f = f[1], P->norm = f[2], P->conv = f[3], P->w_in = f[4], P->bias = f[5];
    P->w_out = f[6], P->A_log = f[7], P->proj = f[8], P->dt_bias = f[9], P->U = f[10];
}

// nanotss.py's initialization (the same distributions)
static void params_init(const Cfg *c, Params *P) {
    const int d = c->d, T = c->T, D = c->D, N = NODES(c), Pd = PDIM(c), K = c->K, pv = c->p;
    const real k_in = 1 / sqrt((real)d), k_out = 1 / sqrt((real)(T * (D + 1))), k_u = k_out / sqrt((real)pv);
#define UNI(k) ((real)((2 * rand01() - 1) * (k)))
    for (size_t i = 0; i < (size_t)c->vocab * d; i++) P->embed[i] = (real)(0.02 * randn());
    for (int i = 0; i < d; i++) P->norm_f[i] = 1;
    for (int l = 0; l < c->n_layer; l++) {
        for (int i = 0; i < d; i++) P->norm[(size_t)l * d + i] = 1;
        for (int i = 0; i < K * d; i++) P->conv[(size_t)l * K * d + i] = i < d;  // tap 0 = 1: the identity
        const size_t r0 = (size_t)l * T * N;
        for (size_t r = r0; r < r0 + (size_t)T * N; r++) {
            for (int i = 0; i < d; i++) P->w_in[r * d + i] = UNI(k_in);
            P->bias[r] = UNI(k_in);
            for (int i = 0; i < d; i++) P->w_out[r * d + i] = UNI(k_out);
            P->A_log[r] = (real)log(1 + 15 * rand01());  // decay rates uniform in [1, 16]
        }
        for (size_t i = 0; i < (size_t)Pd * d; i++) P->proj[(size_t)l * Pd * d + i] = UNI(k_in);
        for (int t = 0; t < T; t++) {  // dt log-uniform in [0.001, 0.1], through softplus's inverse
            double dt = exp(rand01() * (log(0.1) - log(0.001)) + log(0.001));
            if (dt < 1e-4) dt = 1e-4;
            P->dt_bias[(size_t)l * T + t] = (real)(dt + log(-expm1(-dt)));
        }
        for (size_t i = 0; i < (size_t)T * pv * d; i++) P->U[(size_t)l * T * pv * d + i] = UNI(k_u);
    }
#undef UNI
}

typedef struct {
    Cfg c;
    Params P, Q, G, M, V, *Gt;
    unsigned char *hit, *hits;
    Work **ws;
    Codes *k;
    int threads, Lmax;
} Model;

static Model *model_new(const Cfg *c, int threads, int Lmax) {
    Model *m = calloc(1, sizeof(Model));
    m->c = *c, m->threads = threads, m->Lmax = Lmax;
    params_alloc(c, &m->P), params_alloc(c, &m->Q), params_alloc(c, &m->G), params_alloc(c, &m->M), params_alloc(c, &m->V);
    m->Gt = calloc(threads, sizeof(Params));
    for (int t = 0; t < threads; t++) params_alloc(c, m->Gt + t);
    const size_t rows = (size_t)c->n_layer * c->T * NODES(c);
    m->hit = calloc(rows, 1), m->hits = calloc(rows * threads, 1);
    m->ws = calloc(threads, sizeof(Work *));
    for (int t = 0; t < threads; t++) {
        m->ws[t] = tss_work_new(c, Lmax);
        tss_work_bf16(m->ws[t], 1);
    }
    m->k = tss_codes_new(c);
    return m;
}

static void model_requantize(Model *m) {  // the quantized weights and their codes, from the latent ones
    tss_quantize(&m->c, &m->P, &m->Q);
    tss_codes_all(&m->c, &m->Q, m->k);
}

// Forward only over one sequence: the summed cross-entropy at the scored tokens (labels >= 0); logits (L, vocab), if
// given, receives every position's logits.
static double eval_seq(Model *m, const int *tok, const int *lab, int L, Work *w, int *ns, real *logits) {
    const Cfg *c = &m->c;
    const int d = c->d, N = NODES(c), T = c->T;
    w->k = m->k, w->no_alt = 1;
    real *x = w->x, *h = w->h, *xn = w->xn, *lg = w->lg;
    for (int t = 0; t < L; t++) memcpy(x + (size_t)t * d, m->P.embed + (size_t)tok[t] * d, sizeof(real) * d);
    for (int l = 0; l < c->n_layer; l++) {
        memset(h, 0, sizeof(real) * (size_t)T * N * c->n * c->p);
        layer_fwd(c, &m->P, &m->Q, l, L, x, h, w->tp + l, w);
    }
    double loss = 0;
    *ns = 0;
    for (int t = 0; t < L; t++) {
        if (!logits && lab[t] < 0) continue;
        real r;
        rmsnorm_fwd(x + (size_t)t * d, m->P.norm_f, xn, d, &r);
        real mx = -INFINITY, z = 0;
        for (int v = 0; v < c->vocab; v++) lg[v] = dotr(m->P.embed + (size_t)v * d, xn, d), mx = lg[v] > mx ? lg[v] : mx;
        if (logits) memcpy(logits + (size_t)t * c->vocab, lg, sizeof(real) * c->vocab);
        if (lab[t] < 0) continue;
        for (int v = 0; v < c->vocab; v++) z += exp(lg[v] - mx);
        loss += log(z) + mx - lg[lab[t]];
        (*ns)++;
    }
    w->no_alt = 0;
    return loss;
}

// The recurrence, a token at a time (what inference runs): the node states and the conv's history carried along.
typedef struct {
    real *h, *hist;               // (n_layer, T * N, n * p), (n_layer, K - 1, d): the latest first
    real *x, *u, *mx, *xt, *y, *p, *R, *Racc;
    int8_t *xq;
} Gen;

static Gen *gen_new(const Cfg *c) {
    Gen *g = calloc(1, sizeof(Gen));
    const int d = c->d;
    g->h = calloc((size_t)c->n_layer * c->T * NODES(c) * c->n * c->p, sizeof(real));
    g->hist = calloc((size_t)c->n_layer * (c->K > 1 ? c->K - 1 : 1) * d, sizeof(real));
    g->x = calloc(d, sizeof(real)), g->u = calloc(d, sizeof(real)), g->mx = calloc(d, sizeof(real));
    g->xt = calloc(d, sizeof(real)), g->y = calloc(d, sizeof(real)), g->p = calloc(PDIM(c), sizeof(real));
    g->R = calloc(c->p, sizeof(real)), g->Racc = calloc(c->p, sizeof(real)), g->xq = calloc(d, 1);
    return g;
}

static void gen_step(Model *m, Gen *g, int token, real *logits) {
    const Cfg *c = &m->c;
    const Params *P = &m->P, *Q = &m->Q;
    const Codes *k = m->k;
    const int d = c->d, T = c->T, D = c->D, n = c->n, pv = c->p, K = c->K, Pd = PDIM(c), N = NODES(c), gs = c->group;
    memcpy(g->x, P->embed + (size_t)token * d, sizeof(real) * d);
    for (int l = 0; l < c->n_layer; l++) {
        const real *conv = P->conv + (size_t)l * K * d, *bias = P->bias + (size_t)l * T * N, *A_log = P->A_log + (size_t)l * T * N;
        const size_t r_node = (size_t)l * T * N, r_proj = (size_t)l * Pd, r_U = (size_t)l * T * pv;
        real *hl = g->h + (size_t)l * T * N * n * pv, *hb = g->hist + (size_t)l * (K - 1) * d, rr;
        rmsnorm_fwd(g->x, P->norm + (size_t)l * d, g->u, d, &rr);
        for (int i = 0; i < d; i++) g->mx[i] = conv[i] * g->u[i];
        for (int j = 1; j < K; j++)
            for (int i = 0; i < d; i++) g->mx[i] += conv[j * d + i] * hb[(size_t)(j - 1) * d + i];
        if (K > 1) {
            memmove(hb + d, hb, sizeof(real) * (size_t)(K - 2) * d);
            memcpy(hb, g->u, sizeof(real) * d);
        }
        real xs;
        quant8(g->mx, g->xt, g->xq, &xs, d);
        for (int r = 0; r < Pd; r++) g->p[r] = qdotx(k, k ? &k->proj : NULL, Q->proj, r_proj + r, g->xt, g->xq, xs, d, gs);
        const real *B = g->p, *C = g->p + n, *V = g->p + 2 * n, *Z = g->p + 2 * n + pv;
        memset(g->y, 0, sizeof(real) * d);
        for (int tr = 0; tr < T; tr++) {
            const real dt = softplus(g->p[2 * n + 2 * pv + tr] + P->dt_bias[(size_t)l * T + tr]);
            memset(g->Racc, 0, sizeof(real) * pv);
            int a = 0;
            for (int kk = 0; kk <= D; kk++) {
                const size_t row = (size_t)tr * N + a;
                real *ha = hl + row * n * pv;
                const real dec = exp(-exp(A_log[row]) * dt);
                for (int q = 0; q < pv; q++) g->R[q] = 0;
                for (int j = 0; j < n; j++)
                    for (int q = 0; q < pv; q++) {
                        ha[j * pv + q] *= dec;
                        g->R[q] += C[j] * ha[j * pv + q];
                    }
                const real s = qdotx(k, k ? &k->w_in : NULL, Q->w_in, r_node + row, g->xt, g->xq, xs, d, gs) + bias[row] +
                               dotr(g->R, Z, pv);
                qaxpy(k, k ? &k->w_out : NULL, Q->w_out, r_node + row, g->y, gelu(s), d, gs);
                for (int q = 0; q < pv; q++) g->Racc[q] += g->R[q];
                for (int j = 0; j < n; j++)
                    for (int q = 0; q < pv; q++) ha[j * pv + q] += dt * B[j] * V[q];
                a = 2 * a + 1 + (s > 0);
            }
            for (int q = 0; q < pv; q++) qaxpy(k, k ? &k->U : NULL, Q->U, r_U + (size_t)tr * pv + q, g->y, g->Racc[q], d, gs);
        }
        for (int i = 0; i < d; i++) g->x[i] += g->y[i];
    }
    real rr;
    rmsnorm_fwd(g->x, P->norm_f, g->u, d, &rr);
    for (int v = 0; v < c->vocab; v++) logits[v] = dotr(P->embed + (size_t)v * d, g->u, d);
}

static int sample_logits(const real *lg, int V, double temp) {
    double mx = -1e300, z = 0;
    for (int v = 0; v < V; v++) mx = lg[v] > mx ? lg[v] : mx;
    for (int v = 0; v < V; v++) z += exp((lg[v] - mx) / temp);
    double r = rand01() * z;
    for (int v = 0; v < V; v++)
        if ((r -= exp((lg[v] - mx) / temp)) <= 0) return v;
    return V - 1;
}

// checkpoints: "TSSC", the Cfg, every field of the latent parameters (float32 or double as built)
static int save(const Model *m, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    size_t z[N_FIELDS];
    sizes(&m->c, z);
    real *pf[N_FIELDS];
    fields(&m->P, pf);
    const int rs = sizeof(real);
    fwrite("TSSC", 1, 4, f), fwrite(&rs, sizeof(int), 1, f), fwrite(&m->c, sizeof(Cfg), 1, f);
    for (int i = 0; i < N_FIELDS; i++) fwrite(pf[i], sizeof(real), z[i], f);
    return fclose(f);
}

static Model *load(const char *path, int threads, int Lmax) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[4];
    int rs;
    Cfg c;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TSSC", 4) || fread(&rs, sizeof(int), 1, f) != 1 || rs != (int)sizeof(real) ||
        fread(&c, sizeof(Cfg), 1, f) != 1) {
        fclose(f);
        return NULL;
    }
    Model *m = model_new(&c, threads, Lmax);
    size_t z[N_FIELDS];
    sizes(&c, z);
    real *pf[N_FIELDS];
    fields(&m->P, pf);
    for (int i = 0; i < N_FIELDS; i++)
        if (fread(pf[i], sizeof(real), z[i], f) != z[i]) {
            fclose(f);
            return NULL;
        }
    fclose(f);
    model_requantize(m);
    return m;
}

static void generate(Model *m, const char *prompt, int n, double temp) {
    Gen *g = gen_new(&m->c);
    real *lg = malloc(sizeof(real) * m->c.vocab);
    const char *pr = prompt && *prompt ? prompt : "\n";
    for (const char *q = pr; *q; q++) {
        gen_step(m, g, (unsigned char)*q, lg);
        fputc(*q, stdout);
    }
    for (int i = 0; i < n; i++) {
        const int v = sample_logits(lg, m->c.vocab, temp);
        fputc(v, stdout);
        gen_step(m, g, v, lg);
    }
    fputc('\n', stdout);
    free(lg);
}

// --check: the generator against the training forward pass (the same logits at every position), and a few training
// steps (the loss must fall)
static int check(int threads) {
    Cfg c = {.vocab = 256, .d = 64, .n_layer = 2, .T = 4, .D = 5, .n = 16, .p = 16, .K = 3, .group = 64, .walks = 2, .temp = 1};
    const int L = 96;
    Model *m = model_new(&c, threads, L);
    params_init(&c, &m->P);
    model_requantize(m);
    int tok[96], lab[96], ns;
    for (int t = 0; t < L; t++) tok[t] = (int)(rand01() * 256), lab[t] = -1;
    real *lf = malloc(sizeof(real) * L * c.vocab), *lg = malloc(sizeof(real) * c.vocab);
    eval_seq(m, tok, lab, L, m->ws[0], &ns, lf);
    Gen *g = gen_new(&c);
    double worst = 0, scale = 0;
    for (int t = 0; t < L; t++) {
        gen_step(m, g, tok[t], lg);
        for (int v = 0; v < c.vocab; v++) {
            const double e = fabs(lg[v] - lf[(size_t)t * c.vocab + v]);
            worst = e > worst ? e : worst, scale = fabs(lf[(size_t)t * c.vocab + v]) > scale ? fabs(lf[(size_t)t * c.vocab + v]) : scale;
        }
    }
    printf("generator vs training forward: worst logit difference %.1e (largest logit %.2f)\n", worst, scale);
    // a fixed batch learned: a sequence repeating a short phrase
    const int B = 8;
    int *bt = malloc(sizeof(int) * B * L), *bl = malloc(sizeof(int) * B * L);
    const char *phrase = "the tree state space learns. ";
    for (int b = 0; b < B; b++)
        for (int t = 0; t < L; t++) bt[b * L + t] = (unsigned char)phrase[(b + t) % 29], bl[b * L + t] = (unsigned char)phrase[(b + t + 1) % 29];
    Hyper hp = {.lr = (real)3e-3, .beta1 = (real)0.9, .beta2 = (real)0.98, .eps = (real)1e-8, .weight_decay = (real)0.01, .clip = 1};
    double first = 0, last = 0;
    for (int s = 1; s <= 60; s++) {
        hp.step = s;
        last = tss_train_step(&c, &m->P, &m->Q, &m->G, m->hit, &m->M, &m->V, m->Gt, m->hits, m->ws, threads, bt, bl, B, L, &hp, m->k);
        if (s == 1) first = last;
    }
    printf("training on a repeated phrase: loss %.3f -> %.3f in 60 steps\n", first, last);
    const int ok = worst < 1e-3 * (scale > 1 ? scale : 1) && last < 0.5 * first;
    printf(ok ? "check passed\n" : "CHECK FAILED\n");
    return ok ? 0 : 1;
}

static void usage(void) {
    fprintf(stderr,
            "nanotss: the tree state space, trained and run on a CPU (one file, C).\n"
            "  train:    nanotss --data input.txt [--steps 5000] [--batch 16] [--seq 256] [--lr 3e-3] [--warmup 100]\n"
            "                    [--d 128] [--layers 4] [--trees 4] [--depth 7] [--state 16] [--value 16] [--walks 2]\n"
            "                    [--threads 4] [--eval-every 250] [--eval-windows 64] [--seed 0] [--save model.bin]\n"
            "                    [--sample 600] [--temperature 0.8] [--prompt TEXT]\n"
            "  generate: nanotss --load model.bin --sample 600 [--prompt TEXT] [--temperature 0.8]\n"
            "  check:    nanotss --check\n");
}

int main(int argc, char **argv) {
    const char *data = NULL, *save_path = NULL, *load_path = NULL, *prompt = "\n";
    int steps = 5000, B = 16, L = 256, warmup = 100, threads = 4, eval_every = 250, eval_windows = 64, n_sample = 600;
    int d = 128, layers = 4, trees = 4, depth = 7, state = 16, value = 16, walks = 2, do_check = 0;
    double lr = 3e-3, temperature = 0.8;
    unsigned long long seed = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
#define ARG(name, target, conv) else if (!strcmp(a, name) && v) target = conv(v), i++;
        if (!strcmp(a, "--check")) do_check = 1;
        ARG("--data", data, (const char *))
        ARG("--save", save_path, (const char *))
        ARG("--load", load_path, (const char *))
        ARG("--prompt", prompt, (const char *))
        ARG("--steps", steps, atoi) ARG("--batch", B, atoi) ARG("--seq", L, atoi) ARG("--warmup", warmup, atoi)
        ARG("--threads", threads, atoi) ARG("--eval-every", eval_every, atoi) ARG("--eval-windows", eval_windows, atoi)
        ARG("--sample", n_sample, atoi) ARG("--d", d, atoi) ARG("--layers", layers, atoi) ARG("--trees", trees, atoi)
        ARG("--depth", depth, atoi) ARG("--state", state, atoi) ARG("--value", value, atoi) ARG("--walks", walks, atoi)
        ARG("--lr", lr, atof) ARG("--temperature", temperature, atof) ARG("--seed", seed, atoll)
        else {
            usage();
            return 2;
        }
#undef ARG
    }
    g_rng ^= (seed + 1) * 0x9E3779B97F4A7C15ULL;
    if (threads > 64) threads = 64;
    if (do_check) return check(threads);
    if (load_path && !data) {  // generate from a checkpoint
        Model *m = load(load_path, 1, 16);
        if (!m) {
            fprintf(stderr, "cannot load %s\n", load_path);
            return 1;
        }
        generate(m, prompt, n_sample, temperature);
        return 0;
    }
    if (!data) {
        usage();
        return 2;
    }
    FILE *f = fopen(data, "rb");
    if (!f) {
        fprintf(stderr, "cannot read %s\n", data);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    const long nbytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *raw = malloc(nbytes);
    if (fread(raw, 1, nbytes, f) != (size_t)nbytes) return 1;
    fclose(f);
    const long split = (long)(nbytes * 0.9);
    if (split < L + 2 || nbytes - split < L + 2) {
        fprintf(stderr, "%s is too short for --seq %d\n", data, L);
        return 1;
    }
    int g = 128;
    while (d % g) g--;  // the largest group <= 128 dividing d
    Cfg c = {.vocab = 256, .d = d, .n_layer = layers, .T = trees, .D = depth, .n = state, .p = value, .K = 3, .group = g,
             .walks = walks, .temp = 1};
    Model *m = load_path ? load(load_path, threads, L) : model_new(&c, threads, L);
    if (!m) {
        fprintf(stderr, "cannot load %s\n", load_path);
        return 1;
    }
    if (!load_path) {
        params_init(&c, &m->P);
        model_requantize(m);
    }
    c = m->c;
    size_t z[N_FIELDS], np = 0;
    sizes(&c, z);
    for (int i = 0; i < N_FIELDS; i++) np += z[i];
    printf("nanotss: %s (%ld bytes to train on, %ld held out), %.2fM parameters, d %d, %d layers x %d trees of depth %d, "
           "state %d x %d, batch %d x %d, walks %d, %d threads%s\n",
           data, split, nbytes - split, np / 1e6, c.d, c.n_layer, c.T, c.D, c.n, c.p, B, L, c.walks, threads,
           m->k && m->k->ok ? " (2-bit rows, AVX-512)" : "");
    fflush(stdout);
    // fixed held-out windows
    int *et = malloc(sizeof(int) * eval_windows * L), *el = malloc(sizeof(int) * eval_windows * L);
    uint64_t keep = g_rng;
    g_rng = 1234;
    for (int w = 0; w < eval_windows; w++) {
        const long s0 = split + (long)(rand01() * (nbytes - split - L - 1));
        for (int t = 0; t < L; t++) et[w * L + t] = raw[s0 + t], el[w * L + t] = raw[s0 + t + 1];
    }
    g_rng = keep;
    int *bt = malloc(sizeof(int) * B * L), *bl = malloc(sizeof(int) * B * L);
    Hyper hp = {.lr = (real)lr, .beta1 = (real)0.9, .beta2 = (real)0.98, .eps = (real)1e-8, .weight_decay = (real)0.01, .clip = 1};
    const double t0 = wall();
    double seen = 0;
    for (int step = 1; step <= steps; step++) {
        double sc = step < warmup ? (double)step / warmup : 1;
        sc *= 0.1 + 0.9 * 0.5 * (1 + cos(3.141592653589793 * step / steps));  // cosine to 10% of the peak
        hp.lr = (real)(lr * sc), hp.step = step;
        for (int b = 0; b < B; b++) {
            const long s0 = (long)(rand01() * (split - L - 1));
            for (int t = 0; t < L; t++) bt[b * L + t] = raw[s0 + t], bl[b * L + t] = raw[s0 + t + 1];
        }
        const double loss = tss_train_step(&c, &m->P, &m->Q, &m->G, m->hit, &m->M, &m->V, m->Gt, m->hits, m->ws, threads, bt,
                                           bl, B, L, &hp, m->k);
        seen += (double)B * L;
        if (step % eval_every == 0 || step == steps) {
            const double el_t = wall() - t0;
            double tot = 0;
            long cnt = 0;
#pragma omp parallel for num_threads(threads) reduction(+ : tot, cnt) schedule(dynamic, 1)
            for (int w = 0; w < eval_windows; w++) {
#ifdef _OPENMP
                const int th = omp_get_thread_num();
#else
                const int th = 0;
#endif
                int ns;
                tot += eval_seq(m, et + (size_t)w * L, el + (size_t)w * L, L, m->ws[th], &ns, NULL);
                cnt += ns;
            }
            const double v = tot / cnt;
            printf("  step %5d  train %.3f  held-out %.3f nats/byte (%.2f bits)  %.1fM tokens  %.1f min  %.0f tokens/s\n", step,
                   loss, v, v / log(2), seen / 1e6, el_t / 60, seen / el_t);
            fflush(stdout);
        }
    }
    if (save_path) printf(save(m, save_path) ? "could not save %s\n" : "saved %s\n", save_path);
    if (n_sample > 0) {
        printf("--- sample (temperature %.1f) ---\n", temperature);
        generate(m, prompt, n_sample, temperature);
    }
    return 0;
}
