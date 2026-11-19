// cpu/int8_avx2.cpp: SPEC 6 steps 3 and 4 with AVX2 intrinsics, step 5 as a plain loop.
// The tile functions are also used by fused_avx2.cpp, so both variants run identical arithmetic.
#include <cstddef>
#include <immintrin.h>

#include "kernels.h"

constexpr int VEC = 8;                 // floats per AVX2 register
constexpr int NVEC = COL_BLOCK / VEC;  // registers per tile row

// Two signed 8-bit values as sign-extended 16-bit halves of one 32-bit lane (low half = x0).
static int32_t pack_pair(int8_t x0, int8_t x1) {
    const uint32_t lo = (uint16_t)(int16_t)x0;
    const uint32_t hi = (uint16_t)(int16_t)x1;
    return (int32_t)(lo | (hi << 16));
}

void int8_tile_avx2(const Quant& q, int i, int N, int j0, float* out) {
    // SPEC 6 step 3. Not _mm256_maddubs_epi16: it is unsigned x signed and saturates.
    __m256i acc[NVEC];
    for (int t = 0; t < NVEC; t++) acc[t] = _mm256_setzero_si256();
    const int8_t* x = q.xq + (size_t)i * PAIR * q.kp;
    for (int pr = 0; pr < q.kp; pr++) {
        const __m256i xa = _mm256_set1_epi32(pack_pair(x[PAIR * pr], x[PAIR * pr + 1]));
        const int8_t* w = q.wp + ((size_t)pr * N + j0) * PAIR;
        for (int t = 0; t < NVEC; t++) {
            const __m128i raw = _mm_loadu_si128((const __m128i*)(w + PAIR * VEC * t));
            const __m256i w16 = _mm256_cvtepi8_epi16(raw);
            acc[t] = _mm256_add_epi32(acc[t], _mm256_madd_epi16(xa, w16));
        }
    }
    // Dequantize in the fixed order ((float)acc * sx) * sw.
    const __m256 sxv = _mm256_set1_ps(q.sx[i]);
    for (int t = 0; t < NVEC; t++) {
        __m256 f = _mm256_mul_ps(_mm256_cvtepi32_ps(acc[t]), sxv);
        f = _mm256_mul_ps(f, _mm256_loadu_ps(q.sw + j0 + VEC * t));
        _mm256_storeu_ps(out + VEC * t, f);
    }
}

void outlier_tile_avx2(const Quant& q, int k, int i, int N, int j0, float* out) {
    // SPEC 6 step 4: ascending channels per element; vectorized across columns, never channels.
    __m256 co[NVEC];
    for (int t = 0; t < NVEC; t++) co[t] = _mm256_setzero_ps();
    const float* x = q.xh + (size_t)i * k;
    for (int s = 0; s < k; s++) {
        const __m256 xv = _mm256_set1_ps(x[s]);
        const float* w = q.wh + (size_t)s * N + j0;
        for (int t = 0; t < NVEC; t++)
            co[t] = _mm256_add_ps(co[t], _mm256_mul_ps(xv, _mm256_loadu_ps(w + VEC * t)));
    }
    for (int t = 0; t < NVEC; t++) _mm256_storeu_ps(out + VEC * t, co[t]);
}

void int8_gemm_avx2(const Problem& p, const Quant& q, float* ci8) {
    const int ntile = p.N / COL_BLOCK;
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < p.M; i++)
        for (int jb = 0; jb < ntile; jb++) {
            const int j0 = jb * COL_BLOCK;
            int8_tile_avx2(q, i, p.N, j0, ci8 + (size_t)i * p.N + j0);
        }
}

void outlier_gemm_avx2(const Problem& p, const Select& sel, const Quant& q, float* co) {
    const int ntile = p.N / COL_BLOCK;
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < p.M; i++)
        for (int jb = 0; jb < ntile; jb++) {
            const int j0 = jb * COL_BLOCK;
            outlier_tile_avx2(q, sel.k, i, p.N, j0, co + (size_t)i * p.N + j0);
        }
}

void merge_plain(const Problem& p, const float* ci8, const float* co, float* C) {
    // SPEC 6 step 5
#pragma omp parallel for schedule(static)
    for (int i = 0; i < p.M; i++)
        for (int j = 0; j < p.N; j++) {
            const size_t pos = (size_t)i * p.N + j;
            C[pos] = ci8[pos] + co[pos];
        }
}
