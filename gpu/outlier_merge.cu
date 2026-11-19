// gpu/outlier_merge.cu: SPEC 6 step 4 (outlier GEMM into Co) and step 5 (merge into C), templated on TILE.
#include "stages.h"
#include "util.h"

// The compact outlier channels are walked in ascending order, TILE at a time, so the float32
// accumulation order is exactly the one of SPEC 6 step 4. The multiply and the add are separate
// statements, and the build uses --fmad=false, so no fused multiply-add can appear.
template <int TILE>
__global__ void outlier_kernel(const __half* Xh, const __half* Wh, int M, int N, int k, float* Co) {
    __shared__ float xs[TILE][TILE];
    __shared__ float ws[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;
    for (int s0 = 0; s0 < k; s0 += TILE) {
        xs[ty][tx] = (row < M && s0 + tx < k) ? __half2float(Xh[(size_t)row * k + s0 + tx]) : 0.0f;
        ws[ty][tx] = (s0 + ty < k && col < N) ? __half2float(Wh[(size_t)(s0 + ty) * N + col]) : 0.0f;
        __syncthreads();
        int lim = (k - s0 < TILE) ? (k - s0) : TILE;
        for (int w = 0; w < lim; w++) {
            float p = xs[ty][w] * ws[w][tx];
            acc = acc + p;
        }
        __syncthreads();
    }
    if (row < M && col < N) Co[(size_t)row * N + col] = acc;
}

template <int TILE>
__global__ void merge_kernel(const float* Ci8, const float* Co, int M, int N, float* C) {
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    if (row < M && col < N) {
        size_t idx = (size_t)row * N + col;
        C[idx] = Ci8[idx] + Co[idx];
    }
}

template <int TILE>
static void launch_outlier(const DevBuffers& d, const Dims& dm) {
    dim3 block(TILE, TILE);
    dim3 grid((dm.N + TILE - 1) / TILE, (dm.M + TILE - 1) / TILE);
    outlier_kernel<TILE><<<grid, block>>>(d.Xh, d.Wh, dm.M, dm.N, dm.k, d.Co);
    CUDA_CHECK(cudaGetLastError());
}

template <int TILE>
static void launch_merge(const DevBuffers& d, const Dims& dm) {
    dim3 block(TILE, TILE);
    dim3 grid((dm.N + TILE - 1) / TILE, (dm.M + TILE - 1) / TILE);
    merge_kernel<TILE><<<grid, block>>>(d.Ci8, d.Co, dm.M, dm.N, d.C);
    CUDA_CHECK(cudaGetLastError());
}

void run_outlier(int tile, const DevBuffers& d, const Dims& dm) {
    // SPEC 6 step 4
    switch (tile) {
        case 8: launch_outlier<8>(d, dm); break;
        case 16: launch_outlier<16>(d, dm); break;
        case 32: launch_outlier<32>(d, dm); break;
        default: die("unsupported tile", "must be 8, 16 or 32");
    }
}

void run_merge(int tile, const DevBuffers& d, const Dims& dm) {
    // SPEC 6 step 5
    switch (tile) {
        case 8: launch_merge<8>(d, dm); break;
        case 16: launch_merge<16>(d, dm); break;
        case 32: launch_merge<32>(d, dm); break;
        default: die("unsupported tile", "must be 8, 16 or 32");
    }
}
