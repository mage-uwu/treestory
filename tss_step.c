// Tree state space: one token through the recurrence, in plain C (what tss.py's ``step`` computes).
// Per layer and tree it touches depth + 1 rows of w_in and w_out and depth + 1 node states, nothing else.
// Build: cc -O3 -march=native -ffast-math -shared -fPIC -o tss_step.so tss_step.c -lm   (tss.py bench builds and loads it)
#include <math.h>
#include <string.h>

typedef struct {
    int vocab, d, n_layer, trees, depth, state, value, d_conv, memory, readout;  // readout: 0 level, 1 tree, 2 layer
    const float *embed;   // (vocab, d), tied with the output layer
    const float *norm_f;  // (d)
    // per layer, concatenated over layers:
    const float *norm;    // (d)
    const float *conv;    // (d_conv, d)
    const float *w_in;    // (trees, nodes, d)
    const float *bias;    // (trees, nodes)
    const float *w_out;   // (trees, nodes, d)
    const float *proj;    // (2 * state + 2 * value + trees, d)  [B, C, V, Z, dt]
    const float *dt_bias; // (trees)
    const float *neg_A;   // (trees, nodes): -exp(A_log)
    const float *U;       // (trees, depth + 1, value, d), (trees, 1, value, d) or (1, 1, value, d) by readout
} Model;

static float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678f)); }
static float softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

static float dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static void rmsnorm(float *out, const float *x, const float *w, int d) {
    float ss = dot(x, x, d) / d;
    float r = 1.0f / sqrtf(ss + 1e-5f);
    for (int i = 0; i < d; i++) out[i] = x[i] * r * w[i];
}

// h: (n_layer, trees, nodes, state, value); hist: (n_layer, d_conv - 1, d);
// scratch: >= 4 d + 2 state + (3 + trees) value + trees floats.
// Writes the next-token logits (vocab).
void tss_step(const Model *m, float *h, float *hist, int token, float *logits, float *scratch) {
    const int d = m->d, T = m->trees, D = m->depth, n = m->state, pv = m->value, K = m->d_conv;
    const int N = (1 << (D + 1)) - 1;
    float *x = scratch, *u = x + d, *mx = u + d, *y = mx + d, *p = y + d;  // p: B, C, V, Z, dt
    float *R = p + 2 * n + 2 * pv + T, *Racc = R + pv;  // Racc: the reads summed, per tree or for the layer
    const int nU = m->readout == 0 ? T * (D + 1) : m->readout == 1 ? T : 1;
    memcpy(x, m->embed + (size_t)token * d, d * sizeof(float));
    for (int l = 0; l < m->n_layer; l++) {
        const float *conv = m->conv + (size_t)l * K * d;
        const float *w_in = m->w_in + (size_t)l * T * N * d, *w_out = m->w_out + (size_t)l * T * N * d;
        const float *bias = m->bias + (size_t)l * T * N, *neg_A = m->neg_A + (size_t)l * T * N;
        const float *U = m->U + (size_t)l * nU * pv * d;
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
        memset(y, 0, d * sizeof(float));
        float *B = p, *C = p + n, *V = p + 2 * n, *Z = p + 2 * n + pv, *dt = p + 2 * n + 2 * pv;
        if (m->memory) {
            const float *proj = m->proj + (size_t)l * (2 * n + 2 * pv + T) * d;
            for (int r = 0; r < 2 * n + 2 * pv + T; r++) p[r] = dot(proj + (size_t)r * d, mx, d);
            for (int t = 0; t < T; t++) dt[t] = softplus(dt[t] + m->dt_bias[l * T + t]);
        }
        if (m->memory && m->readout) memset(Racc, 0, (size_t)T * pv * sizeof(float));
        for (int t = 0; t < T; t++) {
            int a = 0;
            for (int k = 0; k <= D; k++) {
                const size_t row = (size_t)t * N + a;
                float s = dot(mx, w_in + row * d, d) + bias[row];
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
                        const float *Uk = U + ((size_t)t * (D + 1) + k) * pv * d;
                        for (int q = 0; q < pv; q++) {
                            const float r = R[q], *Uq = Uk + (size_t)q * d;
                            for (int i = 0; i < d; i++) y[i] += r * Uq[i];
                        }
                    } else {
                        float *acc = Racc + (m->readout == 1 ? t * pv : 0);
                        for (int q = 0; q < pv; q++) acc[q] += R[q];
                    }
                }
                const float g = gelu(s), *o = w_out + row * d;
                for (int i = 0; i < d; i++) y[i] += g * o[i];
                a = 2 * a + 1 + (s > 0.0f);  // the branch
            }
        }
        if (m->memory && m->readout) {  // one readout per tree, or one for the layer
            for (int t = 0; t < (m->readout == 1 ? T : 1); t++)
                for (int q = 0; q < pv; q++) {
                    const float r = Racc[t * pv + q], *Uq = U + ((size_t)t * pv + q) * d;
                    for (int i = 0; i < d; i++) y[i] += r * Uq[i];
                }
        }
        for (int i = 0; i < d; i++) x[i] += y[i];
    }
    rmsnorm(u, x, m->norm_f, d);
    for (int v = 0; v < m->vocab; v++) logits[v] = dot(m->embed + (size_t)v * d, u, d);
}

// n tokens in a row (teacher-forced), the logits of the last; what a benchmark should time.
void tss_run(const Model *m, float *h, float *hist, const int *tokens, int n, float *logits, float *scratch) {
    for (int i = 0; i < n; i++) tss_step(m, h, hist, tokens[i], logits, scratch);
}
