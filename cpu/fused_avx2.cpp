// cpu/fused_avx2.cpp: SPEC 6 steps 3, 4 and 5 in one loop nest (SPEC 7, mixed_fused_avx2).
#include <cstddef>

#include "kernels.h"

// Ci8 and Co live only in tile-sized buffers, never in M x N arrays, and C is written once.
void fused_gemm_avx2(const Problem& p, const Select& sel, const Quant& q, float* C) {
    const int N = p.N;
    const int k = sel.k;
    const int ntile = N / COL_BLOCK;
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < p.M; i++) {
        for (int jb = 0; jb < ntile; jb++) {
            const int j0 = jb * COL_BLOCK;
            float ci8[COL_BLOCK];
            float* c = C + (size_t)i * N + j0;
            int8_tile_avx2(q, i, N, j0, ci8);
            if (k > 0) {
                float co[COL_BLOCK];
                outlier_tile_avx2(q, k, i, N, j0, co);
                for (int jj = 0; jj < COL_BLOCK; jj++) c[jj] = ci8[jj] + co[jj];  // SPEC 6 step 5
            } else {
                for (int jj = 0; jj < COL_BLOCK; jj++) c[jj] = ci8[jj];  // k = 0: C = Ci8
            }
        }
    }
}
