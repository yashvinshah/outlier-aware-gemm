// gpu/fp32_cublas.cu: fp32_cublas variant. cuBLAS SGEMM with CUBLAS_DEFAULT_MATH (no TF32).
#include "stages.h"
#include "util.h"

static void check_cublas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) die("cuBLAS call failed", what);
}

// Called once before timing: handle creation is outside every timed region (SPEC 8).
void make_cublas(cublasHandle_t* handle) {
    check_cublas(cublasCreate(handle), "cublasCreate");
    check_cublas(cublasSetMathMode(*handle, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
}

// cuBLAS is column-major. Row-major C (M x N) is column-major C^T (N x M) = W^T * X^T, so the operands are
// swapped: W (row-major K x N) is column-major W^T with leading dimension N, X likewise with leading dimension K.
void run_fp32_cublas(cublasHandle_t handle, const DevBuffers& d, const Dims& dm) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dm.N, dm.M, dm.K, &alpha, d.W, dm.N, d.X, dm.K,
                             &beta, d.C, dm.N),
                 "cublasSgemm");
}
