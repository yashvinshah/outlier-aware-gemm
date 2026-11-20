// cpu/int8_scalar.cpp: SPEC 6 steps 3, 4 and 5 as plain loops. Built without auto-vectorization
// on purpose (SPEC 12): this file is the scalar rung of the ladder. No intrinsics here.
#include <cstddef>

#include "kernels.h"

void int8_gemm_scalar(const Problem& p, const Quant& q, float* ci8) {
    // SPEC 6 step 3
    const int N = p.N;
    const int ntile = N / COL_BLOCK;
    const size_t stride = (size_t)PAIR * q.kp;
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < p.M; i++) {
        for (int jb = 0; jb < ntile; jb++) {
            const int j0 = jb * COL_BLOCK;
            int32_t acc[COL_BLOCK] = {0};
            const int8_t* x = q.xq + (size_t)i * stride;
            for (int pr = 0; pr < q.kp; pr++) {
                const int32_t x0 = x[PAIR * pr];
                const int32_t x1 = x[PAIR * pr + 1];
                const int8_t* w = q.wp + ((size_t)pr * N + j0) * PAIR;
                for (int jj = 0; jj < COL_BLOCK; jj++)
                    acc[jj] += x0 * (int32_t)w[PAIR * jj] + x1 * (int32_t)w[PAIR * jj + 1];
            }
            for (int jj = 0; jj < COL_BLOCK; jj++)
                ci8[(size_t)i * N + j0 + jj] = ((float)acc[jj] * q.sx[i]) * q.sw[j0 + jj];
        }
    }
}

void outlier_gemm_scalar(const Problem& p, const Select& sel, const Quant& q, float* co) {
    // SPEC 6 step 4: per element, channels in ascending order, separate multiply and add.
    const int N = p.N;
    const int k = sel.k;
    const int ntile = N / COL_BLOCK;
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < p.M; i++) {
        for (int jb = 0; jb < ntile; jb++) {
            const int j0 = jb * COL_BLOCK;
            float acc[COL_BLOCK];
            for (int jj = 0; jj < COL_BLOCK; jj++) acc[jj] = 0.0f;
            for (int s = 0; s < k; s++) {
                const float x = q.xh[(size_t)i * k + s];
                const float* w = q.wh + (size_t)s * N + j0;
                for (int jj = 0; jj < COL_BLOCK; jj++) acc[jj] = acc[jj] + x * w[jj];
            }
            for (int jj = 0; jj < COL_BLOCK; jj++) co[(size_t)i * N + j0 + jj] = acc[jj];
        }
    }
}

void merge_scalar(const Problem& p, const float* ci8, const float* co, float* C) {
    // SPEC 6 step 5
#pragma omp parallel for schedule(static)
    for (int i = 0; i < p.M; i++)
        for (int j = 0; j < p.N; j++) {
            const size_t pos = (size_t)i * p.N + j;
            C[pos] = ci8[pos] + co[pos];
        }
}
