// gpu/bench_gpu.cu: runs exactly one configuration (SPEC 11) and appends one CSV row (SPEC 10).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../common/checksum.h"
#include "../common/params.h"
#include "stages.h"
#include "util.h"

constexpr int    DEFAULT_WARMUPS = 3;   // SPEC 8
constexpr int    DEFAULT_REPS    = 10;  // SPEC 8
constexpr double PERCENT         = 100.0;
constexpr double ROUND_HALF      = 0.5;
constexpr double FLOP_PER_MAC    = 2.0;  // SPEC 9: gflops = 2 * M * K * N / (total_ms_median * 1e6)
constexpr double GFLOPS_DIVISOR  = 1e6;

enum Variant { V_FP32_CUBLAS, V_PERTENSOR, V_MIXED_SEP, V_MIXED_FUSED, V_MIXED_SEP_OPT1 };

struct Args {
    Variant variant;
    std::string variant_name, shape, true_tag, data_dir, out_csv, write_c;
    int M, K, N, tile, warmups, reps;
    double true_pct, iso_pct;
};

struct HostData {
    float* X;
    float* W;
    float* ref;
    int32_t* outliers;  // only read for mixed variants
    int n_true;
};

struct Ctx {
    Variant variant;
    int tile;
    Dims dm;
    DevBuffers d;
    HostScratch h;
    cublasHandle_t cublas;
};

struct RepTimes {
    double ms[NUM_STAGES];
    double total;
};

static bool is_mixed(Variant v) { return v == V_MIXED_SEP || v == V_MIXED_FUSED || v == V_MIXED_SEP_OPT1; }

static Variant parse_variant(const char* name) {
    if (strcmp(name, "fp32_cublas") == 0) return V_FP32_CUBLAS;
    if (strcmp(name, "int8_pertensor") == 0) return V_PERTENSOR;
    if (strcmp(name, "mixed_sep") == 0) return V_MIXED_SEP;
    if (strcmp(name, "mixed_fused") == 0) return V_MIXED_FUSED;
    if (strcmp(name, "mixed_sep_opt1") == 0) return V_MIXED_SEP_OPT1;
    die("unknown variant", name);
}

static int parse_int(const char* flag, const char* s) {
    char* end = nullptr;
    long v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || v < 0) die("flag needs a non-negative integer", flag);
    return (int)v;
}

static double parse_double(const char* flag, const char* s) {
    char* end = nullptr;
    double v = strtod(s, &end);
    if (end == s || *end != '\0' || v < 0.0) die("flag needs a non-negative number", flag);
    return v;
}

static void parse_flag(Args& a, const char* flag, const char* val) {
    if (strcmp(flag, "--variant") == 0) a.variant_name = val;
    else if (strcmp(flag, "--shape") == 0) a.shape = val;
    else if (strcmp(flag, "--M") == 0) a.M = parse_int(flag, val);
    else if (strcmp(flag, "--K") == 0) a.K = parse_int(flag, val);
    else if (strcmp(flag, "--N") == 0) a.N = parse_int(flag, val);
    else if (strcmp(flag, "--true-pct") == 0) {
        a.true_pct = parse_double(flag, val);
        a.true_tag = val;  // SPEC 17.8: the file tag is the string with "." replaced by "p"
        for (char& ch : a.true_tag)
            if (ch == '.') ch = 'p';
    } else if (strcmp(flag, "--iso-pct") == 0) a.iso_pct = parse_double(flag, val);
    else if (strcmp(flag, "--tile") == 0) a.tile = parse_int(flag, val);
    else if (strcmp(flag, "--warmups") == 0) a.warmups = parse_int(flag, val);
    else if (strcmp(flag, "--reps") == 0) a.reps = parse_int(flag, val);
    else if (strcmp(flag, "--data-dir") == 0) a.data_dir = val;
    else if (strcmp(flag, "--out-csv") == 0) a.out_csv = val;
    else if (strcmp(flag, "--write-c") == 0) a.write_c = val;
    else die("unknown flag", flag);
}

static Args parse_args(int argc, char** argv) {
    Args a = {};
    a.tile = DEFAULT_TILE;
    a.warmups = DEFAULT_WARMUPS;
    a.reps = DEFAULT_REPS;
    a.true_pct = -1.0;
    a.iso_pct = -1.0;
    for (int i = 1; i + 1 < argc; i += 2) parse_flag(a, argv[i], argv[i + 1]);
    if (a.variant_name.empty()) die("missing flag", "--variant");
    a.variant = parse_variant(a.variant_name.c_str());
    if (a.shape.empty() || a.M == 0 || a.K == 0 || a.N == 0) die("missing flag", "--shape, --M, --K, --N");
    if (a.true_pct < 0.0 || a.data_dir.empty() || a.out_csv.empty()) die("missing flag", "--true-pct, --data-dir, --out-csv");
    if (is_mixed(a.variant) && a.iso_pct < 0.0) die("missing flag", "--iso-pct (required for mixed_*)");
    if (a.tile != 8 && a.tile != 16 && a.tile != 32) die("--tile must be 8, 16 or 32", "--tile");
    if (a.reps < 1) die("flag must be at least 1", "--reps");
    return a;
}

// SPEC 2: k and n_true use floor(K * pct / 100 + 0.5) in double.
static int channel_count(int K, double pct) { return (int)std::floor((double)K * pct / PERCENT + ROUND_HALF); }

static Dims make_dims(const Args& a) {
    Dims dm;
    dm.M = a.M;
    dm.K = a.K;
    dm.N = a.N;
    dm.k = is_mixed(a.variant) ? channel_count(a.K, a.iso_pct) : 0;
    if (dm.k >= dm.K) die("iso-pct isolates every channel", "--iso-pct");
    dm.n_norm = dm.K - dm.k;
    dm.kw = (dm.n_norm + PACK_WIDTH - 1) / PACK_WIDTH;
    return dm;
}

static HostData load_host(const Args& a, const Dims& dm) {
    std::string m = std::to_string(a.M), k = std::to_string(a.K), n = std::to_string(a.N);
    std::string dir = a.data_dir + "/";
    HostData hd = {};
    hd.X = (float*)read_file_exact((dir + "X_" + m + "x" + k + "_t" + a.true_tag + ".f32").c_str(),
                                   sizeof(float) * (size_t)dm.M * dm.K);
    hd.W = (float*)read_file_exact((dir + "W_" + k + "x" + n + ".f32").c_str(), sizeof(float) * (size_t)dm.K * dm.N);
    hd.ref = (float*)read_file_exact((dir + "ref_" + m + "x" + k + "x" + n + "_t" + a.true_tag + ".f32").c_str(),
                                     sizeof(float) * (size_t)dm.M * dm.N);
    if (!is_mixed(a.variant)) return hd;
    hd.n_true = channel_count(a.K, a.true_pct);
    if (hd.n_true <= 0) die("true-pct gives no outlier channels", "--true-pct");
    hd.outliers = (int32_t*)read_file_exact((dir + "outliers_K" + k + "_t" + a.true_tag + ".i32").c_str(),
                                            sizeof(int32_t) * (size_t)hd.n_true);
    return hd;
}

static void dev_alloc(void** p, size_t bytes) { CUDA_CHECK(cudaMalloc(p, bytes)); }

// All device buffers are allocated before timing (SPEC 8). Buffers a variant does not use stay null.
static void alloc_device(Ctx& c) {
    const Dims& dm = c.dm;
    DevBuffers& d = c.d;
    size_t mk = (size_t)dm.M * dm.K, kn = (size_t)dm.K * dm.N, mn = (size_t)dm.M * dm.N;
    d = DevBuffers();
    dev_alloc((void**)&d.X, sizeof(float) * mk);
    dev_alloc((void**)&d.W, sizeof(float) * kn);
    dev_alloc((void**)&d.C, sizeof(float) * mn);
    if (c.variant == V_FP32_CUBLAS) return;
    dev_alloc((void**)&d.sx, sizeof(float) * dm.M);
    dev_alloc((void**)&d.sw, sizeof(float) * dm.N);
    dev_alloc((void**)&d.Nrm, sizeof(int) * dm.n_norm);
    dev_alloc((void**)&d.Xq, sizeof(int) * (size_t)dm.M * dm.kw);
    dev_alloc((void**)&d.Wq, sizeof(int) * (size_t)dm.N * dm.kw);
    if (c.variant == V_PERTENSOR) {
        dev_alloc((void**)&d.partial, sizeof(float) * 2 * PT_BLOCKS);
        dev_alloc((void**)&d.scalar, sizeof(float) * 2);
        return;
    }
    if (dm.k == 0) return;
    dev_alloc((void**)&d.colmax, sizeof(float) * dm.K);
    dev_alloc((void**)&d.S, sizeof(int) * dm.k);
    dev_alloc((void**)&d.Xh, sizeof(__half) * (size_t)dm.M * dm.k);
    dev_alloc((void**)&d.Wh, sizeof(__half) * (size_t)dm.k * dm.N);
    if (c.variant == V_MIXED_FUSED) return;
    dev_alloc((void**)&d.Ci8, sizeof(float) * mn);
    dev_alloc((void**)&d.Co, sizeof(float) * mn);
}

// One-time setup, outside every timed region: input copies, scratch sizes, the identity channel list
// (when no select stage runs) and the cuBLAS handle.
static void setup_device(Ctx& c, const HostData& hd) {
    const Dims& dm = c.dm;
    CUDA_CHECK(cudaMemcpy(c.d.X, hd.X, sizeof(float) * (size_t)dm.M * dm.K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c.d.W, hd.W, sizeof(float) * (size_t)dm.K * dm.N, cudaMemcpyHostToDevice));
    if (c.variant == V_FP32_CUBLAS) {
        make_cublas(&c.cublas);
        return;
    }
    if (dm.k == 0) {
        std::vector<int> identity(dm.K);
        for (int i = 0; i < dm.K; i++) identity[i] = i;
        CUDA_CHECK(cudaMemcpy(c.d.Nrm, identity.data(), sizeof(int) * dm.K, cudaMemcpyHostToDevice));
        return;
    }
    c.h.colmax.resize(dm.K);
    c.h.order.resize(dm.K);
    c.h.is_s.resize(dm.K);
    c.h.S.resize(dm.k);
    c.h.Nrm.resize(dm.n_norm);
}

static RepTimes rep_cublas(Ctx& c) {
    RepTimes r = {};
    double t = stage_start_ms();
    run_fp32_cublas(c.cublas, c.d, c.dm);
    r.total = stage_elapsed_ms(t);
    return r;
}

static RepTimes rep_pertensor(Ctx& c) {
    RepTimes r = {};
    double t = stage_start_ms();
    run_quantize_pertensor(c.d, c.dm);
    r.ms[ST_QUANT] = stage_elapsed_ms(t);
    t = stage_start_ms();
    run_int8_gemm(c.tile, c.d, c.dm, c.d.C);
    r.ms[ST_INT8] = stage_elapsed_ms(t);
    return r;
}

// Select and quantize are shared by both mixed variants. A skipped stage stays 0.0 (SPEC 8).
static void select_and_quantize(Ctx& c, RepTimes& r) {
    if (c.dm.k > 0) {
        double t = stage_start_ms();
        run_select(c.d, c.dm, c.h);
        r.ms[ST_SELECT] = stage_elapsed_ms(t);
    }
    double t = stage_start_ms();
    run_quantize(c.d, c.dm);
    r.ms[ST_QUANT] = stage_elapsed_ms(t);
}

static RepTimes rep_mixed_sep(Ctx& c) {
    RepTimes r = {};
    select_and_quantize(c, r);
    // With k = 0 the INT8 result is C itself and steps 4 and 5 are skipped (SPEC 6).
    double t = stage_start_ms();
    auto int8_gemm = (c.variant == V_MIXED_SEP_OPT1) ? run_int8_gemm_opt1 : run_int8_gemm;
    int8_gemm(c.tile, c.d, c.dm, c.dm.k > 0 ? c.d.Ci8 : c.d.C);
    r.ms[ST_INT8] = stage_elapsed_ms(t);
    if (c.dm.k == 0) return r;
    t = stage_start_ms();
    run_outlier(c.tile, c.d, c.dm);
    r.ms[ST_OUTLIER] = stage_elapsed_ms(t);
    t = stage_start_ms();
    run_merge(c.tile, c.d, c.dm);
    r.ms[ST_MERGE] = stage_elapsed_ms(t);
    return r;
}

static RepTimes rep_mixed_fused(Ctx& c) {
    RepTimes r = {};
    select_and_quantize(c, r);
    double t = stage_start_ms();
    run_fused(c.tile, c.d, c.dm);
    r.ms[ST_INT8] = stage_elapsed_ms(t);
    return r;
}

// SPEC 8: total = sum of the stage times of the repetition.
static RepTimes run_rep(Ctx& c) {
    RepTimes r;
    if (c.variant == V_FP32_CUBLAS) return rep_cublas(c);
    if (c.variant == V_PERTENSOR) r = rep_pertensor(c);
    else if (c.variant == V_MIXED_SEP || c.variant == V_MIXED_SEP_OPT1) r = rep_mixed_sep(c);
    else r = rep_mixed_fused(c);
    r.total = 0.0;
    for (int s = 0; s < NUM_STAGES; s++) r.total += r.ms[s];
    return r;
}

static bool stage_present(Variant v, int s) {
    if (v == V_FP32_CUBLAS) return false;
    if (v == V_PERTENSOR) return s == ST_QUANT || s == ST_INT8;
    if (v == V_MIXED_FUSED) return s == ST_SELECT || s == ST_QUANT || s == ST_INT8;
    return true;
}

static void fill_timing(CsvRow& row, Variant variant, const std::vector<RepTimes>& reps) {
    std::vector<double> totals;
    for (const RepTimes& r : reps) totals.push_back(r.total);
    row.total = compute_stats(totals);
    for (int s = 0; s < NUM_STAGES; s++) {
        row.has_stage[s] = stage_present(variant, s);
        if (!row.has_stage[s]) continue;
        std::vector<double> v;
        for (const RepTimes& r : reps) v.push_back(r.ms[s]);
        row.stage_ms[s] = compute_stats(v).median;
    }
}

static void write_c_file(const char* path, const float* C, size_t count) {
    FILE* f = fopen(path, "wb");
    if (f == nullptr) die("cannot open for writing", path);
    if (fwrite(C, sizeof(float), count, f) != count) die("write failed", path);
    if (fclose(f) != 0) die("close failed", path);
    char hex[FNV_HEX_LEN];
    fnv1a_hex(fnv1a(C, sizeof(float) * count), hex);
    printf("C_CHECKSUM %s\n", hex);
}

// Metrics are computed once after timing, from the C of the last timed repetition (SPEC 9).
static void report(const Ctx& c, const Args& a, const HostData& hd, const std::vector<RepTimes>& reps) {
    const Dims& dm = c.dm;
    size_t mn = (size_t)dm.M * dm.N;
    std::vector<float> hostC(mn);
    CUDA_CHECK(cudaMemcpy(hostC.data(), c.d.C, sizeof(float) * mn, cudaMemcpyDeviceToHost));
    CsvRow row = {};
    row.variant = a.variant_name.c_str();
    row.shape = a.shape.c_str();
    row.M = dm.M;
    row.K = dm.K;
    row.N = dm.N;
    row.true_pct = a.true_pct;
    row.has_iso = is_mixed(a.variant);
    row.iso_pct = a.iso_pct;
    row.k = dm.k;
    row.has_tile = (a.variant != V_FP32_CUBLAS);
    row.tile = a.tile;
    row.warmups = a.warmups;
    row.reps = a.reps;
    fill_timing(row, a.variant, reps);
    row.gflops = FLOP_PER_MAC * (double)dm.M * dm.K * dm.N / (row.total.median * GFLOPS_DIVISOR);
    row.rel_err = rel_err(hostC.data(), hd.ref, mn);
    row.has_recall = is_mixed(a.variant);
    if (row.has_recall) row.recall = outlier_recall(c.h.S.data(), dm.k, hd.outliers, hd.n_true);
    write_csv_row(a.out_csv.c_str(), row);
    if (!a.write_c.empty()) write_c_file(a.write_c.c_str(), hostC.data(), mn);
}

int main(int argc, char** argv) {
    Args a = parse_args(argc, argv);
    Ctx c = {};
    c.variant = a.variant;
    c.tile = a.tile;
    c.dm = make_dims(a);
    HostData hd = load_host(a, c.dm);
    alloc_device(c);
    setup_device(c, hd);
    for (int i = 0; i < a.warmups; i++) run_rep(c);
    std::vector<RepTimes> reps;
    for (int i = 0; i < a.reps; i++) reps.push_back(run_rep(c));
    report(c, a, hd, reps);
    return 0;
}
