// gpu/fused.cu: SPEC 6 steps 3, 4 and 5 in one kernel. Ci8 and Co are never stored as M x N arrays.
#include "stages.h"
#include "util.h"

// The INT8 loop runs first, then the outlier loop in ascending channel order, then one write of C.
// The loops are the ones of int8_gemm.cu and outlier_merge.cu, so the bits equal the separate version.
template <int TILE>
__global__ void fused_kernel(const int* Xq, const int* Wq, const float* sx, const float* sw, const __half* Xh,
                             const __half* Wh, int M, int N, int kw, int k, float* C) {
    __shared__ int xq_s[TILE][TILE];
    __shared__ int wq_s[TILE][TILE];
    __shared__ float xo_s[TILE][TILE];
    __shared__ float wo_s[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    int wcol = blockIdx.x * TILE + ty;
    int acc = 0;
    for (int t0 = 0; t0 < kw; t0 += TILE) {
        int wi = t0 + tx;
        xq_s[ty][tx] = (row < M && wi < kw) ? Xq[(size_t)row * kw + wi] : 0;
        wq_s[ty][tx] = (wcol < N && wi < kw) ? Wq[(size_t)wcol * kw + wi] : 0;
        __syncthreads();
        for (int w = 0; w < TILE; w++) acc = __dp4a(xq_s[ty][w], wq_s[tx][w], acc);
        __syncthreads();
    }
    float co = 0.0f;
    for (int s0 = 0; s0 < k; s0 += TILE) {
        xo_s[ty][tx] = (row < M && s0 + tx < k) ? __half2float(Xh[(size_t)row * k + s0 + tx]) : 0.0f;
        wo_s[ty][tx] = (s0 + ty < k && col < N) ? __half2float(Wh[(size_t)(s0 + ty) * N + col]) : 0.0f;
        __syncthreads();
        int lim = (k - s0 < TILE) ? (k - s0) : TILE;
        for (int w = 0; w < lim; w++) {
            float p = xo_s[ty][w] * wo_s[w][tx];
            co = co + p;
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        float ci8 = ((float)acc * sx[row]) * sw[col];
        C[(size_t)row * N + col] = (k > 0) ? (ci8 + co) : ci8;  // SPEC 6 step 5: no merge when k = 0
    }
}

template <int TILE>
static void launch_fused(const DevBuffers& d, const Dims& dm) {
    dim3 block(TILE, TILE);
    dim3 grid((dm.N + TILE - 1) / TILE, (dm.M + TILE - 1) / TILE);
    fused_kernel<TILE><<<grid, block>>>(d.Xq, d.Wq, d.sx, d.sw, d.Xh, d.Wh, dm.M, dm.N, dm.kw, dm.k, d.C);
    CUDA_CHECK(cudaGetLastError());
}

void run_fused(int tile, const DevBuffers& d, const Dims& dm) {
    // SPEC 6 steps 3, 4, 5
    switch (tile) {
        case 8: launch_fused<8>(d, dm); break;
        case 16: launch_fused<16>(d, dm); break;
        case 32: launch_fused<32>(d, dm); break;
        default: die("unsupported tile", "must be 8, 16 or 32");
    }
}
