// cpu/bench_cpu.cpp: runs exactly one configuration (SPEC 11) and appends one CSV row (SPEC 10).
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "checksum.h"
#include "fp32_gemm.h"
#include "kernels.h"
#include "util.h"

constexpr double MS_PER_S = 1000.0;
constexpr double FLOPS_PER_MAC = 2.0;   // SPEC 9: gflops = 2 * M * K * N / (total_ms * 1e6)
constexpr double GFLOPS_DIVISOR = 1e6;

enum Variant {
    V_FP32_SERIAL, V_FP32_OMP, V_PERTENSOR, V_MIXED_SEP_SCALAR, V_MIXED_SEP_AVX2, V_MIXED_FUSED_AVX2
};

struct Config {
    std::string variant_name, shape, true_tag, data_dir, out_csv, write_c;
    Variant variant = V_FP32_SERIAL;
    int M = 0, K = 0, N = 0, threads = 0, warmups = -1, reps = 0;
    double true_pct = 0.0, iso_pct = 0.0;
    bool has_iso = false;
};

struct Data {
    float *X, *W, *R;
    int32_t* truth;
    size_t n_true;
};

struct Workspace {
    float *C, *ci8, *co;
    Select sel;
    Quant q;
    PerTensor pt;
};

struct Measured {
    bool present[NUM_STAGES];
    double stage_median[NUM_STAGES];
    Stats total;
};

static bool is_mixed(Variant v) { return v >= V_MIXED_SEP_SCALAR; }

static Variant parse_variant(const std::string& s) {
    if (s == "fp32_serial") return V_FP32_SERIAL;
    if (s == "fp32_omp") return V_FP32_OMP;
    if (s == "int8_pertensor") return V_PERTENSOR;
    if (s == "mixed_sep_scalar") return V_MIXED_SEP_SCALAR;
    if (s == "mixed_sep_avx2") return V_MIXED_SEP_AVX2;
    if (s == "mixed_fused_avx2") return V_MIXED_FUSED_AVX2;
    die("unknown variant", s.c_str());
}

static int parse_int(const char* flag, const char* s, int min_value) {
    char* end = nullptr;
    const long v = strtol(s, &end, 10);
    if (end == s || *end != '\0' || v < min_value) die("bad integer value for flag", flag);
    return (int)v;
}

static double parse_double(const char* flag, const char* s) {
    char* end = nullptr;
    const double v = strtod(s, &end);
    if (end == s || *end != '\0' || v < 0.0) die("bad value for flag", flag);
    return v;
}

static void check_config(const Config& c, bool have_variant, bool have_true_pct) {
    if (!have_variant || !have_true_pct || c.shape.empty() || c.data_dir.empty() || c.out_csv.empty() ||
        c.M == 0 || c.K == 0 || c.N == 0 || c.threads == 0 || c.warmups < 0 || c.reps == 0)
        die("missing flag", "need --variant --shape --M --K --N --true-pct --threads --warmups --reps --data-dir --out-csv");
    if (is_mixed(c.variant) != c.has_iso) die("--iso-pct is required for mixed_* and not allowed otherwise", c.variant_name.c_str());
    if (c.iso_pct > 100.0) die("--iso-pct is above 100", c.variant_name.c_str());
    if (c.variant == V_FP32_SERIAL && c.threads != 1) die("fp32_serial runs one thread", "--threads 1");
    if (c.variant != V_FP32_SERIAL && c.variant != V_FP32_OMP && c.N % COL_BLOCK != 0)
        die("N is not a multiple of COL_BLOCK (cpu/kernels.h)", c.shape.c_str());
}

static Config parse_args(int argc, char** argv) {
    Config c;
    bool have_variant = false, have_true_pct = false;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) die("flag has no value", argv[i]);
        const std::string f = argv[i];
        const char* v = argv[i + 1];
        if (f == "--variant") { c.variant_name = v; c.variant = parse_variant(c.variant_name); have_variant = true; }
        else if (f == "--shape") c.shape = v;
        else if (f == "--M") c.M = parse_int(argv[i], v, 1);
        else if (f == "--K") c.K = parse_int(argv[i], v, 1);
        else if (f == "--N") c.N = parse_int(argv[i], v, 1);
        else if (f == "--true-pct") {
            c.true_pct = parse_double(argv[i], v);
            c.true_tag = v;  // file tag: "." becomes "p" (SPEC 17.8)
            std::replace(c.true_tag.begin(), c.true_tag.end(), '.', 'p');
            have_true_pct = true;
        }
        else if (f == "--iso-pct") { c.iso_pct = parse_double(argv[i], v); c.has_iso = true; }
        else if (f == "--threads") c.threads = parse_int(argv[i], v, 1);
        else if (f == "--warmups") c.warmups = parse_int(argv[i], v, 0);
        else if (f == "--reps") c.reps = parse_int(argv[i], v, 1);
        else if (f == "--data-dir") c.data_dir = v;
        else if (f == "--out-csv") c.out_csv = v;
        else if (f == "--write-c") c.write_c = v;
        else die("unknown flag", argv[i]);
    }
    check_config(c, have_variant, have_true_pct);
    return c;
}

static Data load_data(const Config& c) {
    const std::string m = std::to_string(c.M), k = std::to_string(c.K), n = std::to_string(c.N);
    const std::string dir = c.data_dir + "/";
    Data d;
    d.X = read_floats((dir + "X_" + m + "x" + k + "_t" + c.true_tag + ".f32").c_str(), (size_t)c.M * c.K);
    d.W = read_floats((dir + "W_" + k + "x" + n + ".f32").c_str(), (size_t)c.K * c.N);
    d.R = read_floats((dir + "ref_" + m + "x" + k + "x" + n + "_t" + c.true_tag + ".f32").c_str(),
                      (size_t)c.M * c.N);
    d.truth = read_int32s((dir + "outliers_K" + k + "_t" + c.true_tag + ".i32").c_str(), &d.n_true);
    return d;
}

static void alloc_mixed(const Problem& p, int k, bool sep, Workspace* w) {
    const size_t M = p.M, K = p.K, N = p.N, nrm = K - (size_t)k, kp = (nrm + 1) / PAIR;
    Select* s = &w->sel;
    s->k = k;
    s->n_nrm = (int)nrm;
    s->s_idx = (int32_t*)alloc_aligned(sizeof(int32_t) * (size_t)k);
    s->nrm_idx = (int32_t*)alloc_aligned(sizeof(int32_t) * nrm);
    for (size_t c = 0; c < nrm; c++) s->nrm_idx[c] = (int32_t)c;  // identity list; only select rewrites it
    s->colmax = (float*)alloc_aligned(sizeof(float) * K);
    s->order = (int32_t*)alloc_aligned(sizeof(int32_t) * K);
    s->is_s = (uint8_t*)alloc_aligned(K);
    Quant* q = &w->q;
    q->kp = (int)kp;
    q->sx = (float*)alloc_aligned(sizeof(float) * M);
    q->sw = (float*)alloc_aligned(sizeof(float) * N);
    q->xq = (int8_t*)alloc_aligned(M * PAIR * kp);
    q->wp = (int8_t*)alloc_aligned(kp * N * PAIR);
    q->xh = (float*)alloc_aligned(sizeof(float) * M * (size_t)k);
    q->wh = (float*)alloc_aligned(sizeof(float) * (size_t)k * N);
    if (sep && k > 0) {
        w->ci8 = (float*)alloc_aligned(sizeof(float) * M * N);
        w->co = (float*)alloc_aligned(sizeof(float) * M * N);
    }
}

static void alloc_pertensor(const Problem& p, Workspace* w) {
    const size_t M = p.M, K = p.K, N = p.N;
    w->pt.xq = (int8_t*)alloc_aligned(M * K);
    w->pt.wq = (int8_t*)alloc_aligned(K * N);
    w->pt.xmax = (float*)alloc_aligned(sizeof(float) * M);
    w->pt.wmax = (float*)alloc_aligned(sizeof(float) * K);
}

static void setup_workspace(const Config& c, const Problem& p, int k, Workspace* w) {
    w->C = (float*)alloc_aligned(sizeof(float) * (size_t)p.M * p.N);
    if (c.variant == V_PERTENSOR) alloc_pertensor(p, w);
    else if (is_mixed(c.variant)) alloc_mixed(p, k, c.variant != V_MIXED_FUSED_AVX2, w);
}

static double ms_since(double t0) { return (omp_get_wtime() - t0) * MS_PER_S; }

static double sum_stages(const double* t) {
    double s = 0.0;
    for (int i = 0; i < NUM_STAGES; i++) s += t[i];
    return s;
}

static double run_fp32(bool omp, const Problem& p, float* C) {
    const double t0 = omp_get_wtime();
    if (omp) fp32_gemm_omp(p.X, p.W, C, p.M, p.K, p.N);
    else fp32_gemm_serial(p.X, p.W, C, p.M, p.K, p.N);
    return ms_since(t0);
}

static double run_pertensor(const Problem& p, Workspace* w, double* t) {
    double t0 = omp_get_wtime();
    quantize_pertensor(p, &w->pt);
    t[ST_QUANT] = ms_since(t0);
    t0 = omp_get_wtime();
    int8_gemm_pertensor(p, w->pt, w->C);
    t[ST_INT8] = ms_since(t0);
    return sum_stages(t);
}

// Select (skipped when k = 0, so its time stays 0.0) and quantize, shared by both mixed families.
static void select_and_quantize(const Problem& p, Workspace* w, double* t) {
    if (w->sel.k > 0) {
        const double t0 = omp_get_wtime();
        select_channels(p, &w->sel);
        t[ST_SELECT] = ms_since(t0);
    }
    const double t1 = omp_get_wtime();
    quantize_mixed(p, w->sel, &w->q);
    t[ST_QUANT] = ms_since(t1);
}

static double run_mixed_sep(bool avx2, const Problem& p, Workspace* w, double* t) {
    select_and_quantize(p, w, t);
    const int k = w->sel.k;
    float* i8_out = (k > 0) ? w->ci8 : w->C;  // k = 0: C = Ci8 (SPEC 6 step 5)
    double t0 = omp_get_wtime();
    if (avx2) int8_gemm_avx2(p, w->q, i8_out);
    else int8_gemm_scalar(p, w->q, i8_out);
    t[ST_INT8] = ms_since(t0);
    if (k > 0) {
        t0 = omp_get_wtime();
        if (avx2) outlier_gemm_avx2(p, w->sel, w->q, w->co);
        else outlier_gemm_scalar(p, w->sel, w->q, w->co);
        t[ST_OUTLIER] = ms_since(t0);
        t0 = omp_get_wtime();
        if (avx2) merge_plain(p, w->ci8, w->co, w->C);
        else merge_scalar(p, w->ci8, w->co, w->C);
        t[ST_MERGE] = ms_since(t0);
    }
    return sum_stages(t);
}

static double run_mixed_fused(const Problem& p, Workspace* w, double* t) {
    select_and_quantize(p, w, t);
    const double t0 = omp_get_wtime();
    fused_gemm_avx2(p, w->sel, w->q, w->C);
    t[ST_INT8] = ms_since(t0);  // steps 3, 4 and 5 together (SPEC 8)
    return sum_stages(t);
}

static double run_once(const Config& c, const Problem& p, Workspace* w, double* t) {
    if (c.variant == V_FP32_SERIAL || c.variant == V_FP32_OMP) return run_fp32(c.variant == V_FP32_OMP, p, w->C);
    if (c.variant == V_PERTENSOR) return run_pertensor(p, w, t);
    if (c.variant == V_MIXED_FUSED_AVX2) return run_mixed_fused(p, w, t);
    return run_mixed_sep(c.variant == V_MIXED_SEP_AVX2, p, w, t);
}

// Stage columns that exist for the variant (SPEC 8); the others are written blank.
static void stage_presence(Variant v, bool* present) {
    for (int s = 0; s < NUM_STAGES; s++) present[s] = (v == V_MIXED_SEP_SCALAR || v == V_MIXED_SEP_AVX2);
    if (v == V_PERTENSOR) present[ST_QUANT] = present[ST_INT8] = true;
    if (v == V_MIXED_FUSED_AVX2) present[ST_SELECT] = present[ST_QUANT] = present[ST_INT8] = true;
}

// SPEC 8: warmups first, then timed repetitions; total = sum of the stage times of each repetition.
static Measured measure(const Config& c, const Problem& p, Workspace* w) {
    double* totals = (double*)alloc_aligned(sizeof(double) * (size_t)c.reps);
    double* samples = (double*)alloc_aligned(sizeof(double) * NUM_STAGES * (size_t)c.reps);
    for (int r = 0; r < c.warmups + c.reps; r++) {
        double t[NUM_STAGES] = {0.0, 0.0, 0.0, 0.0, 0.0};
        const double total = run_once(c, p, w, t);
        if (r < c.warmups) continue;
        const int idx = r - c.warmups;
        totals[idx] = total;
        for (int s = 0; s < NUM_STAGES; s++) samples[(size_t)s * c.reps + idx] = t[s];
    }
    Measured m;
    stage_presence(c.variant, m.present);
    m.total = compute_stats(totals, c.reps);
    for (int s = 0; s < NUM_STAGES; s++) m.stage_median[s] = compute_stats(samples + (size_t)s * c.reps, c.reps).median;
    free(totals);
    free(samples);
    return m;
}

static void write_c_and_checksum(const char* path, const float* C, size_t count) {
    write_floats(path, C, count);
    char hex[FNV_HEX_LEN];
    fnv1a_hex(fnv1a(C, count * sizeof(float)), hex);
    printf("C_CHECKSUM %s\n", hex);
}

static void report(const Config& c, const Data& d, const Workspace& w, const Measured& m) {
    const size_t count = (size_t)c.M * c.N;
    CsvRow r = {};
    r.variant = c.variant_name.c_str();
    r.shape = c.shape.c_str();
    r.M = c.M; r.K = c.K; r.N = c.N;
    r.true_pct = c.true_pct;
    r.has_iso = c.has_iso;
    r.iso_pct = c.iso_pct;
    r.k = w.sel.k;
    r.threads = c.threads; r.warmups = c.warmups; r.reps = c.reps;
    for (int s = 0; s < NUM_STAGES; s++) { r.stage_present[s] = m.present[s]; r.stage_median[s] = m.stage_median[s]; }
    r.total = m.total;
    r.gflops = FLOPS_PER_MAC * c.M * c.K * c.N / (m.total.median * GFLOPS_DIVISOR);
    r.rel_err = rel_err(w.C, d.R, count);
    r.has_recall = is_mixed(c.variant);
    if (r.has_recall) r.recall = outlier_recall(w.sel.s_idx, w.sel.k, d.truth, d.n_true);
    if (!c.write_c.empty()) write_c_and_checksum(c.write_c.c_str(), w.C, count);
    append_csv_row(c.out_csv.c_str(), r);
}

int main(int argc, char** argv) {
    const Config c = parse_args(argc, argv);
    omp_set_num_threads(c.threads);
    const Data d = load_data(c);
    const Problem p = {c.M, c.K, c.N, d.X, d.W};
    // SPEC 2: k = floor(K * p / 100 + 0.5) in double.
    const int k = is_mixed(c.variant) ? (int)std::floor((double)c.K * c.iso_pct / 100.0 + 0.5) : 0;
    Workspace w = {};
    setup_workspace(c, p, k, &w);
    const Measured m = measure(c, p, &w);
    report(c, d, w, m);
    return 0;
}
