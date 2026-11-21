// cpu/int8_avx2_opt1.cpp: SPEC 6 step 3 for mixed_sep_avx2_opt1. Same integer arithmetic and the same
// dequantization order as int8_tile_avx2 (int8_avx2.cpp), so the output bits do not change. The change is
// the loop structure: a micro-tile of ROW_BLOCK rows x MICRO_COLS columns reuses every loaded and
// sign-extended W vector for all ROW_BLOCK rows, and the X channel pairs come pre-packed as 32-bit words
// (pack_x_pairs), so each broadcast is one load instead of a pack in registers.
#include <cstddef>
#include <immintrin.h>

#include "kernels.h"

constexpr int VEC = 8;                          // floats per AVX2 register
constexpr int MICRO_COLS = 2 * VEC;             // columns per micro-tile: 2 registers per row
constexpr int MICRO_VECS = MICRO_COLS / VEC;
constexpr int MICRO_PER_TILE = COL_BLOCK / MICRO_COLS;

// Two signed 8-bit values as sign-extended 16-bit halves of one 32-bit word (low half = x0); the same
// packing as pack_pair in int8_avx2.cpp.
static int32_t pack_pair(int8_t x0, int8_t x1) {
    const uint32_t lo = (uint16_t)(int16_t)x0;
    const uint32_t hi = (uint16_t)(int16_t)x1;
    return (int32_t)(lo | (hi << 16));
}

void pack_x_pairs(const Problem& p, Quant* q) {
    // SPEC 6 step 2 data layout only; the quantized values are the ones quantize_mixed produced.
#pragma omp parallel for schedule(static)
    for (int i = 0; i < p.M; i++) {
        const int8_t* x = q->xq + (size_t)i * PAIR * q->kp;
        int32_t* out = q->xpk + (size_t)i * q->kp;
        for (int pr = 0; pr < q->kp; pr++) out[pr] = pack_pair(x[PAIR * pr], x[PAIR * pr + 1]);
    }
}

// out points at row i0, column j0 of the output; rows are N floats apart.
static void micro_tile(const Quant& q, int i0, int N, int j0, float* out) {
    // SPEC 6 step 3. Not _mm256_maddubs_epi16: it is unsigned x signed and saturates.
    __m256i acc[ROW_BLOCK][MICRO_VECS];
    const int32_t* x[ROW_BLOCK];
    for (int r = 0; r < ROW_BLOCK; r++) {
        x[r] = q.xpk + (size_t)(i0 + r) * q.kp;
        for (int t = 0; t < MICRO_VECS; t++) acc[r][t] = _mm256_setzero_si256();
    }
    for (int pr = 0; pr < q.kp; pr++) {
        const int8_t* w = q.wp + ((size_t)pr * N + j0) * PAIR;
        __m256i w16[MICRO_VECS];
        for (int t = 0; t < MICRO_VECS; t++)
            w16[t] = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w + PAIR * VEC * t)));
        for (int r = 0; r < ROW_BLOCK; r++) {
            const __m256i xa = _mm256_set1_epi32(x[r][pr]);
            for (int t = 0; t < MICRO_VECS; t++)
                acc[r][t] = _mm256_add_epi32(acc[r][t], _mm256_madd_epi16(xa, w16[t]));
        }
    }
    // Dequantize in the fixed order ((float)acc * sx) * sw.
    for (int r = 0; r < ROW_BLOCK; r++) {
        const __m256 sxv = _mm256_set1_ps(q.sx[i0 + r]);
        for (int t = 0; t < MICRO_VECS; t++) {
            __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(acc[r][t]), sxv);
            f = _mm256_mul_ps(f, _mm256_loadu_ps(q.sw + j0 + VEC * t));
            _mm256_storeu_ps(out + (size_t)r * N + VEC * t, f);
        }
    }
}

void int8_gemm_avx2_opt1(const Problem& p, const Quant& q, float* ci8) {
    const int nrow = p.M / ROW_BLOCK;
    const int ntile = p.N / COL_BLOCK;
    // One work item is ROW_BLOCK rows x COL_BLOCK columns, so the four micro-tiles of an item read the
    // same cache lines of the packed W.
#pragma omp parallel for collapse(2) schedule(static)
    for (int ib = 0; ib < nrow; ib++)
        for (int jb = 0; jb < ntile; jb++)
            for (int m = 0; m < MICRO_PER_TILE; m++) {
                const int i0 = ib * ROW_BLOCK;
                const int j0 = jb * COL_BLOCK + m * MICRO_COLS;
                micro_tile(q, i0, p.N, j0, ci8 + (size_t)i0 * p.N + j0);
            }
}
