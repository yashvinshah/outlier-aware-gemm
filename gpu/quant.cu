// gpu/quant.cu: SPEC 6 step 2. Scales, INT8 quantization with __dp4a packing, FP16 gather of outlier channels.
#include "../common/params.h"
#include "stages.h"
#include "util.h"

constexpr unsigned BYTE_MASK  = 0xffu;
constexpr int      BYTE_BITS  = 8;
constexpr float    UNIT_SCALE = 1.0f;  // SPEC 6 step 2: scale used when Nrm is empty or the max is 0

// SPEC 6 step 2: IEEE division, roundf (half away from zero), clamp to [-127, 127].
__device__ int quantize_value(float v, float scale) {
    float r = roundf(v / scale);
    r = fminf(fmaxf(r, -INT8_MAX_F), INT8_MAX_F);
    return (int)r;
}

__global__ void row_scale_kernel(const float* X, const int* Nrm, int M, int K, int n_norm, float* sx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    float m = 0.0f;
    for (int t = 0; t < n_norm; t++) m = fmaxf(m, fabsf(X[(size_t)i * K + Nrm[t]]));
    sx[i] = (m == 0.0f) ? UNIT_SCALE : m / INT8_MAX_F;
}

__global__ void col_scale_kernel(const float* W, const int* Nrm, int N, int n_norm, float* sw) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float m = 0.0f;
    for (int t = 0; t < n_norm; t++) m = fmaxf(m, fabsf(W[(size_t)Nrm[t] * N + j]));
    sw[j] = (m == 0.0f) ? UNIT_SCALE : m / INT8_MAX_F;
}

// Every word is written in full, padding bytes included, so no zero-fill is needed before the kernel.
__global__ void pack_x_kernel(const float* X, const int* Nrm, const float* sx, int M, int K, int n_norm,
                              int kw, int* Xq) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)M * kw) return;
    int i = (int)(idx / kw);
    int w = (int)(idx % kw);
    unsigned word = 0;
    for (int e = 0; e < PACK_WIDTH; e++) {
        int t = w * PACK_WIDTH + e;
        if (t < n_norm) {
            unsigned q = (unsigned)quantize_value(X[(size_t)i * K + Nrm[t]], sx[i]) & BYTE_MASK;
            word |= q << (BYTE_BITS * e);
        }
    }
    Xq[idx] = (int)word;
}

// W is stored transposed: Wq[j][w] holds channels 4w..4w+3 of column j. Threads vary in j so reads coalesce.
__global__ void pack_w_kernel(const float* W, const int* Nrm, const float* sw, int N, int n_norm, int kw,
                              int* Wq) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)N * kw) return;
    int j = (int)(idx % N);
    int w = (int)(idx / N);
    unsigned word = 0;
    for (int e = 0; e < PACK_WIDTH; e++) {
        int t = w * PACK_WIDTH + e;
        if (t < n_norm) {
            unsigned q = (unsigned)quantize_value(W[(size_t)Nrm[t] * N + j], sw[j]) & BYTE_MASK;
            word |= q << (BYTE_BITS * e);
        }
    }
    Wq[(size_t)j * kw + w] = (int)word;
}

__global__ void gather_xh_kernel(const float* X, const int* S, int M, int K, int k, __half* Xh) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)M * k) return;
    int i = (int)(idx / k);
    int s = (int)(idx % k);
    Xh[idx] = __float2half_rn(X[(size_t)i * K + S[s]]);
}

__global__ void gather_wh_kernel(const float* W, const int* S, int N, int k, __half* Wh) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)k * N) return;
    int s = (int)(idx / N);
    int j = (int)(idx % N);
    Wh[idx] = __float2half_rn(W[(size_t)S[s] * N + j]);
}

void run_pack(const DevBuffers& d, const Dims& dm) {
    pack_x_kernel<<<blocks_for((size_t)dm.M * dm.kw), BLOCK_1D>>>(d.X, d.Nrm, d.sx, dm.M, dm.K, dm.n_norm,
                                                                  dm.kw, d.Xq);
    pack_w_kernel<<<blocks_for((size_t)dm.N * dm.kw), BLOCK_1D>>>(d.W, d.Nrm, d.sw, dm.N, dm.n_norm, dm.kw,
                                                                  d.Wq);
    CUDA_CHECK(cudaGetLastError());
}

void run_quantize(const DevBuffers& d, const Dims& dm) {
    // SPEC 6 step 2
    row_scale_kernel<<<blocks_for(dm.M), BLOCK_1D>>>(d.X, d.Nrm, dm.M, dm.K, dm.n_norm, d.sx);
    col_scale_kernel<<<blocks_for(dm.N), BLOCK_1D>>>(d.W, d.Nrm, dm.N, dm.n_norm, d.sw);
    CUDA_CHECK(cudaGetLastError());
    run_pack(d, dm);
    if (dm.k == 0) return;
    gather_xh_kernel<<<blocks_for((size_t)dm.M * dm.k), BLOCK_1D>>>(d.X, d.S, dm.M, dm.K, dm.k, d.Xh);
    gather_wh_kernel<<<blocks_for((size_t)dm.k * dm.N), BLOCK_1D>>>(d.W, d.S, dm.N, dm.k, d.Wh);
    CUDA_CHECK(cudaGetLastError());
}
