// gpu/int8_gemm.cu: SPEC 6 step 3. __dp4a INT8 GEMM with dequantization, templated on TILE.
#include "stages.h"
#include "util.h"

// One thread per output element. Block (bx, by) covers rows by*TILE.. and columns bx*TILE..; the packed
// channel dimension is walked in chunks of TILE words held in shared memory.
template <int TILE>
__global__ void int8_gemm_kernel(const int* Xq, const int* Wq, const float* sx, const float* sw, int M, int N,
                                 int kw, float* out) {
    __shared__ int xs[TILE][TILE];
    __shared__ int ws[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    int wcol = blockIdx.x * TILE + ty;  // W column whose packed words this thread loads
    int acc = 0;
    for (int t0 = 0; t0 < kw; t0 += TILE) {
        int wi = t0 + tx;
        xs[ty][tx] = (row < M && wi < kw) ? Xq[(size_t)row * kw + wi] : 0;
        ws[ty][tx] = (wcol < N && wi < kw) ? Wq[(size_t)wcol * kw + wi] : 0;
        __syncthreads();
        for (int w = 0; w < TILE; w++) acc = __dp4a(xs[ty][w], ws[tx][w], acc);
        __syncthreads();
    }
    // SPEC 6 step 3: exactly this order of operations.
    if (row < M && col < N) out[(size_t)row * N + col] = ((float)acc * sx[row]) * sw[col];
}

template <int TILE>
static void launch_int8(const DevBuffers& d, const Dims& dm, float* out) {
    dim3 block(TILE, TILE);
    dim3 grid((dm.N + TILE - 1) / TILE, (dm.M + TILE - 1) / TILE);
    int8_gemm_kernel<TILE><<<grid, block>>>(d.Xq, d.Wq, d.sx, d.sw, dm.M, dm.N, dm.kw, out);
    CUDA_CHECK(cudaGetLastError());
}

void run_int8_gemm(int tile, const DevBuffers& d, const Dims& dm, float* out) {
    // SPEC 6 step 3
    switch (tile) {
        case 8: launch_int8<8>(d, dm, out); break;
        case 16: launch_int8<16>(d, dm, out); break;
        case 32: launch_int8<32>(d, dm, out); break;
        default: die("unsupported tile", "must be 8, 16 or 32");
    }
}
