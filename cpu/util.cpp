// cpu/util.cpp: file I/O, statistics, metrics and CSV output for the CPU benchmark.
#include "util.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

constexpr size_t ALIGN_BYTES = 64;

// SPEC 10, exactly.
static const char* CSV_HEADER =
    "lane,variant,shape,M,K,N,true_out_pct,iso_out_pct,k_isolated,threads,tile,warmups,reps,"
    "t_select_ms,t_quant_ms,t_int8_ms,t_outlier_ms,t_merge_ms,total_ms_median,total_ms_min,"
    "total_ms_max,total_ms_stddev,gflops,rel_err,outlier_recall";

void die(const char* what, const char* detail) {
    fprintf(stderr, "bench_cpu: %s: %s\n", what, detail);
    exit(1);
}

void* alloc_aligned(size_t bytes) {
    if (bytes == 0) bytes = ALIGN_BYTES;  // posix_memalign with size 0 may return null
    void* p = nullptr;
    if (posix_memalign(&p, ALIGN_BYTES, bytes) != 0) die("out of memory", "posix_memalign failed");
    memset(p, 0, bytes);
    return p;
}

static FILE* open_checked(const char* path, const char* mode) {
    FILE* f = fopen(path, mode);
    if (f == nullptr) die("cannot open file", path);
    return f;
}

static size_t file_bytes(FILE* f, const char* path) {
    if (fseek(f, 0, SEEK_END) != 0) die("cannot seek in file", path);
    long n = ftell(f);
    if (n < 0) die("cannot get file size", path);
    if (fseek(f, 0, SEEK_SET) != 0) die("cannot seek in file", path);
    return (size_t)n;
}

static void read_and_close(FILE* f, void* dst, size_t bytes, const char* path) {
    if (fread(dst, 1, bytes, f) != bytes) die("read failed", path);
    if (fclose(f) != 0) die("close failed", path);
}

float* read_floats(const char* path, size_t count) {
    FILE* f = open_checked(path, "rb");
    if (file_bytes(f, path) != count * sizeof(float)) die("file size does not match the shape", path);
    float* data = (float*)alloc_aligned(count * sizeof(float));
    read_and_close(f, data, count * sizeof(float), path);
    return data;
}

int32_t* read_int32s(const char* path, size_t* count) {
    FILE* f = open_checked(path, "rb");
    size_t bytes = file_bytes(f, path);
    if (bytes % sizeof(int32_t) != 0) die("file size is not a multiple of 4 bytes", path);
    int32_t* data = (int32_t*)alloc_aligned(bytes);
    read_and_close(f, data, bytes, path);
    *count = bytes / sizeof(int32_t);
    return data;
}

void write_floats(const char* path, const float* data, size_t count) {
    FILE* f = open_checked(path, "wb");
    if (fwrite(data, sizeof(float), count, f) != count) die("write failed", path);
    if (fclose(f) != 0) die("close failed", path);
}

Stats compute_stats(const double* v, int n) {
    double* s = (double*)malloc(sizeof(double) * (size_t)n);
    if (s == nullptr) die("out of memory", "statistics buffer");
    std::copy(v, v + n, s);
    std::sort(s, s + n);
    Stats st;
    // SPEC 8 does not define the median for an even count; the mean of the two middle values is used.
    st.median = (n % 2 == 1) ? s[n / 2] : (s[n / 2 - 1] + s[n / 2]) / 2.0;
    st.min = s[0];
    st.max = s[n - 1];
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += s[i];
    const double mean = sum / n;
    double sq = 0.0;
    for (int i = 0; i < n; i++) sq += (s[i] - mean) * (s[i] - mean);
    st.stddev = std::sqrt(sq / n);  // population standard deviation
    free(s);
    return st;
}

double rel_err(const float* C, const float* R, size_t count) {
    // SPEC 17.3 and the human's ruling of 2026-11-18 (to be recorded as SPEC 17.15): serial,
    // row-major, the difference is taken in double, two separate double accumulators.
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < count; i++) {
        const double d = (double)C[i] - (double)R[i];
        num += d * d;
        den += (double)R[i] * (double)R[i];
    }
    return std::sqrt(num) / std::sqrt(den);
}

// Both lists are ascending, so one merge pass counts the intersection.
double outlier_recall(const int32_t* s, int k, const int32_t* truth, size_t n_true) {
    size_t a = 0, b = 0, hits = 0;
    while (a < (size_t)k && b < n_true) {
        if (s[a] == truth[b]) {
            hits++;
            a++;
            b++;
        } else if (s[a] < truth[b]) {
            a++;
        } else {
            b++;
        }
    }
    return (double)hits / (double)n_true;
}

static void put_stage(FILE* f, bool present, double ms) {
    if (present) fprintf(f, "%.6f,", ms);
    else fputc(',', f);  // blank: the stage does not exist in this variant (SPEC 8)
}

void append_csv_row(const char* path, const CsvRow& r) {
    FILE* probe = fopen(path, "r");
    const bool exists = (probe != nullptr);
    if (exists) fclose(probe);
    FILE* f = open_checked(path, "a");
    if (!exists) fprintf(f, "%s\n", CSV_HEADER);
    fprintf(f, "cpu,%s,%s,%d,%d,%d,%g,", r.variant, r.shape, r.M, r.K, r.N, r.true_pct);
    if (r.has_iso) fprintf(f, "%g,%d,", r.iso_pct, r.k);
    else fputs(",,", f);
    fprintf(f, "%d,,%d,%d,", r.threads, r.warmups, r.reps);  // the empty field is tile
    for (int s = 0; s < NUM_STAGES; s++) put_stage(f, r.stage_present[s], r.stage_median[s]);
    fprintf(f, "%.6f,%.6f,%.6f,%.6f,%.3f,%.9e,", r.total.median, r.total.min, r.total.max,
            r.total.stddev, r.gflops, r.rel_err);
    if (r.has_recall) fprintf(f, "%.6f\n", r.recall);
    else fputs("\n", f);
    if (ferror(f) != 0 || fclose(f) != 0) die("cannot write CSV", path);
}
