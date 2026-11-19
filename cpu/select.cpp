// cpu/select.cpp: SPEC 6 step 1, shared by all mixed variants.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>

#include "kernels.h"

constexpr int SEL_BLOCK = 64;  // channels per OpenMP work item in the column-max pass

// A column max is exact in any order, so each thread owns a block of channels (SPEC 17.4).
static void column_max(const Problem& p, float* colmax) {
    const int nblocks = (p.K + SEL_BLOCK - 1) / SEL_BLOCK;
#pragma omp parallel for schedule(static)
    for (int b = 0; b < nblocks; b++) {
        const int c0 = b * SEL_BLOCK;
        const int c1 = std::min(p.K, c0 + SEL_BLOCK);
        for (int c = c0; c < c1; c++) colmax[c] = 0.0f;
        for (int i = 0; i < p.M; i++) {
            const float* row = p.X + (size_t)i * p.K;
            for (int c = c0; c < c1; c++) colmax[c] = std::max(colmax[c], std::fabs(row[c]));
        }
    }
}

// The comparator is a strict total order (colmax descending, index ascending), which is SPEC's
// tie rule. std::sort therefore gives the same order as a stable sort, without its hidden buffer.
static void rank_channels(Select* sel, int K) {
    const float* cm = sel->colmax;
    for (int c = 0; c < K; c++) sel->order[c] = c;
    std::sort(sel->order, sel->order + K, [cm](int32_t a, int32_t b) {
        return cm[a] > cm[b] || (cm[a] == cm[b] && a < b);
    });
    memcpy(sel->s_idx, sel->order, sizeof(int32_t) * (size_t)sel->k);
    std::sort(sel->s_idx, sel->s_idx + sel->k);
}

static void split_channels(Select* sel, int K) {
    memset(sel->is_s, 0, (size_t)K);
    for (int n = 0; n < sel->k; n++) sel->is_s[sel->s_idx[n]] = 1;
    int m = 0;
    for (int c = 0; c < K; c++)
        if (sel->is_s[c] == 0) sel->nrm_idx[m++] = c;
}

void select_channels(const Problem& p, Select* sel) {
    // SPEC 6 step 1
    column_max(p, sel->colmax);
    rank_channels(sel, p.K);
    split_channels(sel, p.K);
}
