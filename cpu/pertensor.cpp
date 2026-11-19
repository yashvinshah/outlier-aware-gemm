// cpu/pertensor.cpp: INT8 GEMM of the int8_pertensor variant (SPEC 6 step 3 with scalar scales).
#include <cstddef>

#include "kernels.h"

// Plain loops: the compiler may vectorize them, the results are exact integers either way.
void int8_gemm_pertensor(const Problem& p, const PerTensor& pt, float* C) {
    // SPEC 6 step 3
    const int N = p.N;
    const int ntile = N / COL_BLOCK;
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < p.M; i++) {
        for (int jb = 0; jb < ntile; jb++) {
            const int j0 = jb * COL_BLOCK;
            int32_t acc[COL_BLOCK] = {0};
            const int8_t* x = pt.xq + (size_t)i * p.K;
            for (int c = 0; c < p.K; c++) {
                const int32_t xv = x[c];
                const int8_t* w = pt.wq + (size_t)c * N + j0;
                for (int jj = 0; jj < COL_BLOCK; jj++) acc[jj] += xv * (int32_t)w[jj];
            }
            for (int jj = 0; jj < COL_BLOCK; jj++)
                C[(size_t)i * N + j0 + jj] = ((float)acc[jj] * pt.sx) * pt.sw;
        }
    }
}
