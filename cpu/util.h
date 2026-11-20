// cpu/util.h: file I/O, statistics, metrics and CSV output for the CPU benchmark.
#pragma once
#include <cstddef>
#include <cstdint>

constexpr int NUM_STAGES = 5;
enum Stage { ST_SELECT, ST_QUANT, ST_INT8, ST_OUTLIER, ST_MERGE };  // CSV column order (SPEC 10)

struct Stats {
    double median, min, max, stddev;
};

// One CSV row (SPEC 10). The lane is always "cpu" and tile is always blank here.
struct CsvRow {
    const char* variant;
    const char* shape;
    int M, K, N;
    double true_pct;
    bool has_iso;
    double iso_pct;
    int k;
    int threads, warmups, reps;
    bool stage_present[NUM_STAGES];
    double stage_median[NUM_STAGES];
    Stats total;
    double gflops, rel_err;
    bool has_recall;
    double recall;
};

[[noreturn]] void die(const char* what, const char* detail);

// 64-byte aligned, zero-filled once here (untimed setup, SPEC 8). Exits on failure.
void* alloc_aligned(size_t bytes);

float* read_floats(const char* path, size_t count);
int32_t* read_int32s(const char* path, size_t* count);
void write_floats(const char* path, const float* data, size_t count);

Stats compute_stats(const double* v, int n);
double rel_err(const float* C, const float* R, size_t count);
double outlier_recall(const int32_t* s, int k, const int32_t* truth, size_t n_true);
void append_csv_row(const char* path, const CsvRow& row);
