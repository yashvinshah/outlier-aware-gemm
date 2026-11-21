// cpu/kernels.h: data structures and entry points shared by the CPU benchmark files.
#pragma once
#include <cstddef>
#include <cstdint>

// Output columns handled by one tile: 8 AVX2 registers of 8 floats. Every SPEC 5 shape has N
// divisible by it, and bench_cpu exits with an error for any other N.
constexpr int COL_BLOCK = 64;
constexpr int PAIR = 2;  // channels consumed together by _mm256_madd_epi16
// mixed_sep_avx2_opt1 computes ROW_BLOCK rows at once; bench_cpu exits with an error when M is not a multiple.
constexpr int ROW_BLOCK = 4;

struct Problem {
    int M, K, N;
    const float* X;  // M x K
    const float* W;  // K x N
};

// SPEC 6 step 1 results.
struct Select {
    int k;             // isolated channels
    int n_nrm;         // K - k
    int32_t* s_idx;    // k channel indices, ascending
    int32_t* nrm_idx;  // K - k channel indices, ascending (the identity list when k = 0)
    float* colmax;     // K, scratch
    int32_t* order;    // K, scratch
    uint8_t* is_s;     // K, scratch
};

// SPEC 6 step 2 results for the mixed variants.
struct Quant {
    int kp;      // channel pairs of the normal channels, (n_nrm + 1) / 2
    float* sx;   // M
    float* sw;   // N
    int8_t* xq;  // M x (PAIR * kp): gathered normal channels; an odd tail column stays zero
    int8_t* wp;  // kp x N x PAIR: the two channels of a pair side by side for each column
    float* xh;   // M x k: FP16-rounded outlier channels of X, stored as float (SPEC 17.2)
    float* wh;   // k x N: FP16-rounded outlier rows of W, stored as float
    int32_t* xpk;  // M x kp: each channel pair of xq as one 32-bit word; only mixed_sep_avx2_opt1 fills it
};

struct PerTensor {
    float sx, sw;
    int8_t* xq;   // M x K
    int8_t* wq;   // K x N
    float* xmax;  // M, scratch: row maxima of X
    float* wmax;  // K, scratch: row maxima of W
};

// select.cpp
void select_channels(const Problem& p, Select* sel);

// quant.cpp
void quantize_mixed(const Problem& p, const Select& sel, Quant* q);
void quantize_pertensor(const Problem& p, PerTensor* pt);

// pertensor.cpp
void int8_gemm_pertensor(const Problem& p, const PerTensor& pt, float* C);

// int8_scalar.cpp (built without auto-vectorization)
void int8_gemm_scalar(const Problem& p, const Quant& q, float* ci8);
void outlier_gemm_scalar(const Problem& p, const Select& sel, const Quant& q, float* co);
void merge_scalar(const Problem& p, const float* ci8, const float* co, float* C);

// int8_avx2.cpp. The tile functions write COL_BLOCK floats of row i starting at column j0.
void int8_tile_avx2(const Quant& q, int i, int N, int j0, float* out);
void outlier_tile_avx2(const Quant& q, int k, int i, int N, int j0, float* out);
void int8_gemm_avx2(const Problem& p, const Quant& q, float* ci8);
void outlier_gemm_avx2(const Problem& p, const Select& sel, const Quant& q, float* co);
void merge_plain(const Problem& p, const float* ci8, const float* co, float* C);

// int8_avx2_opt1.cpp
void pack_x_pairs(const Problem& p, Quant* q);
void int8_gemm_avx2_opt1(const Problem& p, const Quant& q, float* ci8);

// fused_avx2.cpp
void fused_gemm_avx2(const Problem& p, const Select& sel, const Quant& q, float* C);
