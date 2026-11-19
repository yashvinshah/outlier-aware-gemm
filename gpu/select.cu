// gpu/select.cu: SPEC 6 step 1. Column max on the device, top-k on the host (SPEC 8, select stage).
#include <algorithm>
#include <vector>

#include "stages.h"
#include "util.h"

__global__ void colmax_kernel(const float* X, int M, int K, float* colmax) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K) return;
    float m = 0.0f;
    for (int i = 0; i < M; i++) m = fmaxf(m, fabsf(X[(size_t)i * K + c]));
    colmax[c] = m;
}

// S = first k channels of the order, sorted ascending; Nrm = the rest, ascending (SPEC 6 step 1).
static void split_lists(HostScratch& h, const Dims& dm) {
    std::fill(h.is_s.begin(), h.is_s.end(), 0);
    for (int s = 0; s < dm.k; s++) {
        h.S[s] = h.order[s];
        h.is_s[h.order[s]] = 1;
    }
    std::sort(h.S.begin(), h.S.end());
    int n = 0;
    for (int c = 0; c < dm.K; c++)
        if (!h.is_s[c]) h.Nrm[n++] = c;
}

void run_select(const DevBuffers& d, const Dims& dm, HostScratch& h) {
    // SPEC 6 step 1
    colmax_kernel<<<blocks_for(dm.K), BLOCK_1D>>>(d.X, dm.M, dm.K, d.colmax);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h.colmax.data(), d.colmax, sizeof(float) * dm.K, cudaMemcpyDeviceToHost));
    for (int c = 0; c < dm.K; c++) h.order[c] = c;
    const float* cm = h.colmax.data();
    // Stable sort over an ascending index list: ties keep ascending channel order.
    std::stable_sort(h.order.begin(), h.order.end(), [cm](int a, int b) { return cm[a] > cm[b]; });
    split_lists(h, dm);
    CUDA_CHECK(cudaMemcpy(d.S, h.S.data(), sizeof(int) * dm.k, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.Nrm, h.Nrm.data(), sizeof(int) * dm.n_norm, cudaMemcpyHostToDevice));
}
