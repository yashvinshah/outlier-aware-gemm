// gpu/stages.h: problem dimensions, device buffers and the host launchers of every stage.
#pragma once
#include <cstddef>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <vector>

constexpr int PACK_WIDTH = 4;     // channels per 32-bit word: __dp4a consumes four int8 values at once
constexpr int BLOCK_1D   = 256;   // threads per block of the one-dimensional kernels
constexpr int PT_BLOCKS  = 1024;  // partial-max blocks for the per-tensor global max

// k = isolated channels, n_norm = K - k, kw = packed words per row of the normal channels.
struct Dims {
    int M, K, N, k, n_norm, kw;
};

// Buffers a variant does not need stay null.
struct DevBuffers {
    float* X;
    float* W;
    float* C;
    float* colmax;
    int* S;
    int* Nrm;
    float* sx;
    float* sw;
    int* Xq;  // [M][kw] packed int8 words
    int* Wq;  // [N][kw] packed int8 words, W transposed
    __half* Xh;  // [M][k]
    __half* Wh;  // [k][N]
    float* Ci8;
    float* Co;
    float* partial;  // 2 * PT_BLOCKS per-block maxima of |X| and |W|
    float* scalar;   // per-tensor scales: [0] for X, [1] for W
};

// Host-side scratch for the select stage, sized before timing.
struct HostScratch {
    std::vector<float> colmax;
    std::vector<int> order;
    std::vector<int> S;
    std::vector<int> Nrm;
    std::vector<char> is_s;
};

inline int blocks_for(size_t n) { return (int)((n + BLOCK_1D - 1) / BLOCK_1D); }

void run_select(const DevBuffers& d, const Dims& dm, HostScratch& h);
void run_quantize(const DevBuffers& d, const Dims& dm);
void run_pack(const DevBuffers& d, const Dims& dm);
void run_quantize_pertensor(const DevBuffers& d, const Dims& dm);
void run_int8_gemm(int tile, const DevBuffers& d, const Dims& dm, float* out);
void run_outlier(int tile, const DevBuffers& d, const Dims& dm);
void run_merge(int tile, const DevBuffers& d, const Dims& dm);
void run_fused(int tile, const DevBuffers& d, const Dims& dm);
void make_cublas(cublasHandle_t* handle);
void run_fp32_cublas(cublasHandle_t handle, const DevBuffers& d, const Dims& dm);
