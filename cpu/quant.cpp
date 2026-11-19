// cpu/quant.cpp: SPEC 6 step 2 for the mixed variants and the per-tensor variant.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <immintrin.h>

#include "kernels.h"
#include "params.h"

constexpr int SCALE_BLOCK = 64;  // columns per OpenMP work item in the weight-scale pass

// SPEC 6 step 2: roundf rounds half away from zero and the division is IEEE float division.
static int8_t quantize_one(float v, float scale) {
    float q = roundf(v / scale);
    if (q > INT8_MAX_F) q = INT8_MAX_F;
    if (q < -INT8_MAX_F) q = -INT8_MAX_F;
    return (int8_t)q;
}

// If Nrm is empty or the max is 0 the scale is 1.0f.
static float scale_from_max(float m) { return (m == 0.0f) ? 1.0f : m / INT8_MAX_F; }

// FP32 -> FP16 with round-to-nearest-even, then back to float (SPEC 17.2).
static float half_round(float v) { return _cvtsh_ss(_cvtss_sh(v, _MM_FROUND_TO_NEAREST_INT)); }

static void row_scales(const Problem& p, const Select& sel, float* sx) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < p.M; i++) {
        const float* row = p.X + (size_t)i * p.K;
        float m = 0.0f;
        for (int n = 0; n < sel.n_nrm; n++) m = std::max(m, std::fabs(row[sel.nrm_idx[n]]));
        sx[i] = scale_from_max(m);
    }
}

static void col_scales(const Problem& p, const Select& sel, float* sw) {
    const int nblocks = (p.N + SCALE_BLOCK - 1) / SCALE_BLOCK;
#pragma omp parallel for schedule(static)
    for (int b = 0; b < nblocks; b++) {
        const int j0 = b * SCALE_BLOCK;
        const int jn = std::min(SCALE_BLOCK, p.N - j0);
        float m[SCALE_BLOCK] = {0.0f};
        for (int n = 0; n < sel.n_nrm; n++) {
            const float* w = p.W + (size_t)sel.nrm_idx[n] * p.N + j0;
            for (int jj = 0; jj < jn; jj++) m[jj] = std::max(m[jj], std::fabs(w[jj]));
        }
        for (int jj = 0; jj < jn; jj++) sw[j0 + jj] = scale_from_max(m[jj]);
    }
}

// Gather the normal channels of each row into a compact row. The padding column of an odd
// channel count is never written: it stays zero from the one-time fill at allocation (SPEC 8).
static void quantize_x(const Problem& p, const Select& sel, const Quant& q) {
    const size_t stride = (size_t)PAIR * q.kp;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < p.M; i++) {
        const float* row = p.X + (size_t)i * p.K;
        int8_t* out = q.xq + (size_t)i * stride;
        for (int n = 0; n < sel.n_nrm; n++) out[n] = quantize_one(row[sel.nrm_idx[n]], q.sx[i]);
    }
}

// Pack W so that channels 2*pr and 2*pr+1 of the normal list sit side by side for every column.
static void pack_w(const Problem& p, const Select& sel, const Quant& q) {
#pragma omp parallel for schedule(static)
    for (int pr = 0; pr < q.kp; pr++) {
        for (int side = 0; side < PAIR; side++) {
            const int n = PAIR * pr + side;
            if (n >= sel.n_nrm) continue;  // odd tail: the zero padding stays
            const float* w = p.W + (size_t)sel.nrm_idx[n] * p.N;
            int8_t* out = q.wp + (size_t)pr * p.N * PAIR + side;
            for (int j = 0; j < p.N; j++) out[PAIR * j] = quantize_one(w[j], q.sw[j]);
        }
    }
}

static void convert_outliers(const Problem& p, const Select& sel, const Quant& q) {
    const int k = sel.k;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < p.M; i++)
        for (int s = 0; s < k; s++)
            q.xh[(size_t)i * k + s] = half_round(p.X[(size_t)i * p.K + sel.s_idx[s]]);
#pragma omp parallel for schedule(static)
    for (int s = 0; s < k; s++)
        for (int j = 0; j < p.N; j++)
            q.wh[(size_t)s * p.N + j] = half_round(p.W[(size_t)sel.s_idx[s] * p.N + j]);
}

void quantize_mixed(const Problem& p, const Select& sel, Quant* q) {
    // SPEC 6 step 2
    row_scales(p, sel, q->sx);
    col_scales(p, sel, q->sw);
    quantize_x(p, sel, *q);
    pack_w(p, sel, *q);
    convert_outliers(p, sel, *q);
}

// Row maxima are computed in parallel, then combined serially; a max is exact in any order.
static float global_max_abs(const float* data, int rows, int cols, float* row_max) {
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        float m = 0.0f;
        for (int c = 0; c < cols; c++) m = std::max(m, std::fabs(data[(size_t)r * cols + c]));
        row_max[r] = m;
    }
    float m = 0.0f;
    for (int r = 0; r < rows; r++) m = std::max(m, row_max[r]);
    return m;
}

void quantize_pertensor(const Problem& p, PerTensor* pt) {
    // SPEC 6, per-tensor variant
    pt->sx = scale_from_max(global_max_abs(p.X, p.M, p.K, pt->xmax));
    pt->sw = scale_from_max(global_max_abs(p.W, p.K, p.N, pt->wmax));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < p.M; i++)
        for (int c = 0; c < p.K; c++)
            pt->xq[(size_t)i * p.K + c] = quantize_one(p.X[(size_t)i * p.K + c], pt->sx);
#pragma omp parallel for schedule(static)
    for (int c = 0; c < p.K; c++)
        for (int j = 0; j < p.N; j++)
            pt->wq[(size_t)c * p.N + j] = quantize_one(p.W[(size_t)c * p.N + j], pt->sw);
}
