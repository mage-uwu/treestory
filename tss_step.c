// Tree state space: one token through the recurrence, in plain C (what tss.py's ``step`` computes), on the trained
// representation: ternary weights as int8 codes {-1, 0, 1} with one float scale per group, the trees' input as
// per-token int8 activations, so every dot product with a weight row is an integer sum of +-x, one scale per group.
// Per layer and tree it touches depth + 1 rows of w_in and w_out and depth + 1 node states, nothing else.
// Build: cc -O3 -march=native -ffast-math -fopenmp -shared -fPIC -o tss_step.so tss_step.c -lm   (tss.py bench does it)
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
    Packed w_in2, w_out2, proj2, U2;  // the same four matrices in 2 bits (tss_step_packed)
    int state_bf16;       // tss_step_packed: the node states in bfloat16 (half the bytes; rounded at every write)
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

// x -> int8 codes and the factor back to x (tss.py's quantize_activations, 8 bits)
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

// tss.py's quantize_activations on 16 lanes: absmax, x * 127 / max rounded half to even (the default rounding mode)
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
