// common/fp32_gemm.cpp: FP32 reference GEMM, loop order i, k, j (SPEC 7).
#include "fp32_gemm.h"

#include <cstddef>

// One output row. Both versions call this, so every element is summed over k in the same
// ascending order and starts from 0.0f; this is what makes them bit-identical (SPEC 17.10).
static void gemm_row(const float* X, const float* W, float* C, int i, int K, int N) {
    float* c = C + (size_t)i * N;
    for (int j = 0; j < N; j++) c[j] = 0.0f;
    for (int k = 0; k < K; k++) {
        float a = X[(size_t)i * K + k];
        const float* w = W + (size_t)k * N;
        for (int j = 0; j < N; j++) c[j] = c[j] + a * w[j];
    }
}

void fp32_gemm_serial(const float* X, const float* W, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) gemm_row(X, W, C, i, K, N);
}

void fp32_gemm_omp(const float* X, const float* W, float* C, int M, int K, int N) {
#pragma omp parallel for
    for (int i = 0; i < M; i++) gemm_row(X, W, C, i, K, N);
}
