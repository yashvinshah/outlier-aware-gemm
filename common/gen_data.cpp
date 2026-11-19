// common/gen_data.cpp: seeded synthetic data generator (SPEC 4.2 to 4.4).
// Usage: gen_data --M <int> --K <int> --N <int> --true-pct <float> --data-dir <path>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

#include "checksum.h"
#include "fp32_gemm.h"
#include "params.h"

constexpr double PI         = 3.14159265358979323846;
constexpr double TWO_POW_53 = 9007199254740992.0;
constexpr double X_SIGMA    = 1.0;  // X base entries are plain standard normals (SPEC 4.2)

struct Args {
    int M, K, N;
    double true_pct;
    std::string true_tag;  // --true-pct string with "." replaced by "p" (SPEC 17.8)
    std::string data_dir;
};

static void die(const char* what, const char* detail) {
    fprintf(stderr, "gen_data: %s: %s\n", what, detail);
    exit(1);
}

static void* checked_malloc(size_t bytes) {
    void* p = malloc(bytes);
    if (p == nullptr) die("out of memory", "malloc failed");
    return p;
}

static int parse_int(const char* flag, const char* s) {
    char* end = nullptr;
    long v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || v <= 0) die("flag needs a positive integer", flag);
    return (int)v;
}

static Args parse_args(int argc, char** argv) {
    Args a = {0, 0, 0, -1.0, "", ""};
    for (int i = 1; i + 1 < argc; i += 2) {
        const char* flag = argv[i];
        const char* val = argv[i + 1];
        if (strcmp(flag, "--M") == 0) a.M = parse_int(flag, val);
        else if (strcmp(flag, "--K") == 0) a.K = parse_int(flag, val);
        else if (strcmp(flag, "--N") == 0) a.N = parse_int(flag, val);
        else if (strcmp(flag, "--true-pct") == 0) {
            char* end = nullptr;
            a.true_pct = strtod(val, &end);
            if (end == val || *end != '\0' || a.true_pct < 0.0) die("bad value", flag);
            a.true_tag = val;
            std::replace(a.true_tag.begin(), a.true_tag.end(), '.', 'p');
        } else if (strcmp(flag, "--data-dir") == 0) a.data_dir = val;
        else die("unknown flag", flag);
    }
    if (a.M == 0 || a.K == 0 || a.N == 0 || a.true_pct < 0.0 || a.data_dir.empty())
        die("usage", "gen_data --M <int> --K <int> --N <int> --true-pct <float> --data-dir <path>");
    return a;
}

static double u01(std::mt19937_64& rng) {
    return ((double)(rng() >> 11) + 0.5) * (1.0 / TWO_POW_53);
}

// Box-Muller pairs; z0 then z1 of each pair fill the array in order (SPEC 4.2).
// sigma = 1.0 for X is exact, so (float)(sigma * z) equals (float)z.
static void fill_normals(float* out, size_t count, uint64_t seed, double sigma) {
    std::mt19937_64 rng(seed);
    for (size_t p = 0; p < count; p += 2) {
        double u1 = u01(rng);
        double u2 = u01(rng);
        double r = std::sqrt(-2.0 * std::log(u1));
        double a = 2.0 * PI * u2;
        out[p] = (float)(sigma * (r * std::cos(a)));
        if (p + 1 < count) out[p + 1] = (float)(sigma * (r * std::sin(a)));
    }
}

// Returns K entries; the first n_true are the sorted true outlier channels (SPEC 4.2).
static int32_t* make_outliers(int K, int n_true) {
    int32_t* idx = (int32_t*)checked_malloc(sizeof(int32_t) * (size_t)K);
    for (int i = 0; i < K; i++) idx[i] = i;
    std::mt19937_64 rng(SEED_OUTLIERS);
    for (int i = K - 1; i >= 1; i--) {
        int j = (int)(rng() % (uint64_t)(i + 1));
        std::swap(idx[i], idx[j]);
    }
    std::sort(idx, idx + n_true);
    return idx;
}

static void scale_outliers(float* X, int M, int K, const int32_t* set, int n_true) {
    for (int i = 0; i < M; i++)
        for (int n = 0; n < n_true; n++) {
            size_t pos = (size_t)i * K + (size_t)set[n];
            X[pos] = X[pos] * OUTLIER_SCALE;
        }
}

static void write_and_checksum(const std::string& dir, const std::string& name,
                               const void* data, size_t bytes) {
    std::string path = dir + "/" + name;
    FILE* f = fopen(path.c_str(), "wb");
    if (f == nullptr) die("cannot open for writing", path.c_str());
    if (fwrite(data, 1, bytes, f) != bytes) die("write failed", path.c_str());
    if (fclose(f) != 0) die("close failed", path.c_str());
    char hex[FNV_HEX_LEN];
    fnv1a_hex(fnv1a(data, bytes), hex);
    printf("CHECKSUM %s %s\n", name.c_str(), hex);
}

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    // SPEC 2: k and n_true use floor(K * pct / 100 + 0.5) in double.
    int n_true = (int)std::floor((double)a.K * a.true_pct / 100.0 + 0.5);
    if (n_true > a.K) die("true-pct gives more outlier channels than K", a.true_tag.c_str());

    size_t mk = (size_t)a.M * a.K, kn = (size_t)a.K * a.N, mn = (size_t)a.M * a.N;
    float* X = (float*)checked_malloc(sizeof(float) * mk);
    float* W = (float*)checked_malloc(sizeof(float) * kn);
    float* ref = (float*)checked_malloc(sizeof(float) * mn);

    fill_normals(X, mk, SEED_X, X_SIGMA);
    fill_normals(W, kn, SEED_W, W_SIGMA);
    int32_t* outliers = make_outliers(a.K, n_true);
    scale_outliers(X, a.M, a.K, outliers, n_true);
    fp32_gemm_omp(X, W, ref, a.M, a.K, a.N);  // SPEC 4.2: reference is computed after scaling

    std::string m = std::to_string(a.M), k = std::to_string(a.K), n = std::to_string(a.N);
    write_and_checksum(a.data_dir, "X_" + m + "x" + k + "_t" + a.true_tag + ".f32", X, sizeof(float) * mk);
    write_and_checksum(a.data_dir, "W_" + k + "x" + n + ".f32", W, sizeof(float) * kn);
    write_and_checksum(a.data_dir, "ref_" + m + "x" + k + "x" + n + "_t" + a.true_tag + ".f32", ref,
                       sizeof(float) * mn);
    write_and_checksum(a.data_dir, "outliers_K" + k + "_t" + a.true_tag + ".i32", outliers,
                       sizeof(int32_t) * (size_t)n_true);

    free(X);
    free(W);
    free(ref);
    free(outliers);
    return 0;
}
