// gpu/util.cpp: see util.h.
#include "util.h"

#include <algorithm>
#include <chrono>
#include <cmath>

static const char* const CSV_HEADER =
    "lane,variant,shape,M,K,N,true_out_pct,iso_out_pct,k_isolated,threads,tile,warmups,reps,"
    "t_select_ms,t_quant_ms,t_int8_ms,t_outlier_ms,t_merge_ms,total_ms_median,total_ms_min,"
    "total_ms_max,total_ms_stddev,gflops,rel_err,outlier_recall";

void die(const char* what, const char* detail) {
    fprintf(stderr, "bench_gpu: %s: %s\n", what, detail);
    exit(1);
}

void* checked_malloc(size_t bytes) {
    void* p = malloc(bytes > 0 ? bytes : 1);
    if (p == nullptr) die("out of memory", "malloc failed");
    return p;
}

// A file whose size differs from the expected size is an error, so a wrong shape cannot go unnoticed.
void* read_file_exact(const char* path, size_t bytes) {
    FILE* f = fopen(path, "rb");
    if (f == nullptr) die("cannot open", path);
    if (fseek(f, 0, SEEK_END) != 0) die("seek failed", path);
    long size = ftell(f);
    if (size < 0) die("cannot get file size", path);
    if ((size_t)size != bytes) die("unexpected file size", path);
    if (fseek(f, 0, SEEK_SET) != 0) die("seek failed", path);
    void* buf = checked_malloc(bytes);
    if (fread(buf, 1, bytes, f) != bytes) die("read failed", path);
    if (fclose(f) != 0) die("close failed", path);
    return buf;
}

Stats compute_stats(const std::vector<double>& v) {
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    size_t n = s.size();
    Stats r;
    r.median = (n % 2 == 1) ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
    r.min = s.front();
    r.max = s.back();
    double mean = 0.0;
    for (double x : s) mean += x;
    mean /= (double)n;
    double var = 0.0;
    for (double x : s) var += (x - mean) * (x - mean);
    r.stddev = std::sqrt(var / (double)n);
    return r;
}

// SPEC 17.3 and 17.15: serial, ascending index, two double accumulators, operands widened before subtracting.
double rel_err(const float* C, const float* R, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)C[i] - (double)R[i];
        num += d * d;
        double r = (double)R[i];
        den += r * r;
    }
    return std::sqrt(num) / std::sqrt(den);
}

// Both lists are ascending, so one merge pass counts the intersection.
double outlier_recall(const int* S, int k, const int32_t* true_set, int n_true) {
    int i = 0, j = 0, hit = 0;
    while (i < k && j < n_true) {
        if (S[i] == true_set[j]) {
            hit++;
            i++;
            j++;
        } else if (S[i] < true_set[j]) {
            i++;
        } else {
            j++;
        }
    }
    return (double)hit / (double)n_true;
}

static void put_num(FILE* f, bool present, const char* fmt, double v) {
    if (present) fprintf(f, fmt, v);
}

static void write_row_body(FILE* f, const CsvRow& r) {
    fprintf(f, "gpu,%s,%s,%d,%d,%d,%g,", r.variant, r.shape, r.M, r.K, r.N, r.true_pct);
    put_num(f, r.has_iso, "%g", r.iso_pct);
    fputc(',', f);
    if (r.has_iso) fprintf(f, "%d", r.k);
    fputc(',', f);
    fputc(',', f);  // threads is blank for the GPU lane
    if (r.has_tile) fprintf(f, "%d", r.tile);
    fprintf(f, ",%d,%d,", r.warmups, r.reps);
    for (int s = 0; s < NUM_STAGES; s++) {
        put_num(f, r.has_stage[s], "%.6f", r.stage_ms[s]);
        fputc(',', f);
    }
    fprintf(f, "%.6f,%.6f,%.6f,%.6f,%.3f,%.9e,", r.total.median, r.total.min, r.total.max, r.total.stddev,
            r.gflops, r.rel_err);
    put_num(f, r.has_recall, "%.6f", r.recall);
    fputc('\n', f);
}

void write_csv_row(const char* path, const CsvRow& r) {
    FILE* probe = fopen(path, "r");
    bool exists = (probe != nullptr);
    if (exists) fclose(probe);
    FILE* f = fopen(path, "a");
    if (f == nullptr) die("cannot open for appending", path);
    if (!exists) fprintf(f, "%s\n", CSV_HEADER);
    write_row_body(f, r);
    if (ferror(f) != 0 || fclose(f) != 0) die("write failed", path);
}

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

// SPEC 8: synchronize before starting and before stopping every stage timer.
double stage_start_ms() {
    CUDA_CHECK(cudaDeviceSynchronize());
    return now_ms();
}

double stage_elapsed_ms(double start_ms) {
    CUDA_CHECK(cudaDeviceSynchronize());
    return now_ms() - start_ms;
}
