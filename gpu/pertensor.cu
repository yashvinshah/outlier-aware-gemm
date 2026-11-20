// gpu/pertensor.cu: quantize stage of int8_pertensor (SPEC 6, per-tensor variant): one scale for all of X and
// one for all of W. The INT8 GEMM itself is run_int8_gemm with scale arrays that hold the scalar.
#include "../common/params.h"
#include "stages.h"
#include "util.h"

constexpr float UNIT_SCALE = 1.0f;  // scale used when the max is 0

// Block-wide max over BLOCK_1D values; BLOCK_1D is a power of two.
__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    return sh[0];
}

__global__ void partial_max_kernel(const float* a, size_t n, float* partial) {
    __shared__ float sh[BLOCK_1D];
    float m = 0.0f;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) m = fmaxf(m, fabsf(a[i]));
    float bm = block_max(m, sh);
    if (threadIdx.x == 0) partial[blockIdx.x] = bm;
}

// Launched with one block. Max is exact in any order, so the result does not depend on the reduction tree.
__global__ void scale_from_partial_kernel(const float* partial, float* out) {
    __shared__ float sh[BLOCK_1D];
    float m = 0.0f;
    for (int i = threadIdx.x; i < PT_BLOCKS; i += blockDim.x) m = fmaxf(m, partial[i]);
    float bm = block_max(m, sh);
    if (threadIdx.x == 0) *out = (bm == 0.0f) ? UNIT_SCALE : bm / INT8_MAX_F;
}

__global__ void fill_scales_kernel(const float* scalar, int M, int N, float* sx, float* sw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) sx[i] = scalar[0];
    if (i < N) sw[i] = scalar[1];
}

void run_quantize_pertensor(const DevBuffers& d, const Dims& dm) {
    // SPEC 6 per-tensor variant: global maxima, then the scalar scales, then the shared packing kernels
    partial_max_kernel<<<PT_BLOCKS, BLOCK_1D>>>(d.X, (size_t)dm.M * dm.K, d.partial);
    partial_max_kernel<<<PT_BLOCKS, BLOCK_1D>>>(d.W, (size_t)dm.K * dm.N, d.partial + PT_BLOCKS);
    scale_from_partial_kernel<<<1, BLOCK_1D>>>(d.partial, d.scalar);
    scale_from_partial_kernel<<<1, BLOCK_1D>>>(d.partial + PT_BLOCKS, d.scalar + 1);
    int longest = (dm.M > dm.N) ? dm.M : dm.N;
    fill_scales_kernel<<<blocks_for(longest), BLOCK_1D>>>(d.scalar, dm.M, dm.N, d.sx, d.sw);
    CUDA_CHECK(cudaGetLastError());
    run_pack(d, dm);
}
