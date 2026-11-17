// common/fp32_gemm.h: FP32 reference GEMM, C = X * W, all row-major (SPEC 7).
// X is M x K, W is K x N, C is M x N.
#pragma once

void fp32_gemm_serial(const float* X, const float* W, float* C, int M, int K, int N);
void fp32_gemm_omp(const float* X, const float* W, float* C, int M, int K, int N);
