// Tree state space: one token through the recurrence, in plain C (what tss.py's ``step`` computes), on the trained
// representation: ternary weights as int8 codes {-1, 0, 1} with one float scale per group, the trees' input as
// per-token int8 activations, so every dot product with a weight row is an integer sum of +-x, one scale per group.
// Per layer and tree it touches depth + 1 rows of w_in and w_out and depth + 1 node states, nothing else.
// Build: cc -O3 -march=native -ffast-math -shared -fPIC -o tss_step.so tss_step.c -lm   (tss.py bench does it)
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct {  // a ternary matrix: rows of d codes, rows x (d / group) scales
    const int8_t *codes;
    const float *scales;
} Ternary;

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
void tss_step(const Model *m, float *h, float *hist, int token, float *logits, float *scratch) {
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

// n tokens in a row (teacher-forced), the logits of the last; what a benchmark should time.
void tss_run(const Model *m, float *h, float *hist, const int *tokens, int n, float *logits, float *scratch) {
    for (int i = 0; i < n; i++) tss_step(m, h, hist, tokens[i], logits, scratch);
}
