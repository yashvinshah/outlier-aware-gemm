// gpu/util.h: file reading, statistics, rel_err, outlier_recall, CSV writing, stage timing and the CUDA error check.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

// The only macro in the GPU lane (CLAUDE.md section 4): it needs __FILE__ and __LINE__ of the caller.
#define CUDA_CHECK(call)                                                                         \
    do {                                                                                         \
        cudaError_t err_ = (call);                                                               \
        if (err_ != cudaSuccess) {                                                               \
            fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

// Stages of SPEC 8, in CSV column order.
enum Stage { ST_SELECT, ST_QUANT, ST_INT8, ST_OUTLIER, ST_MERGE, NUM_STAGES };

struct Stats {
    double median, min, max, stddev;  // stddev is the population standard deviation (SPEC 8)
};

// One CSV row (SPEC 10). A field with has_* false is written blank.
struct CsvRow {
    const char* variant;
    const char* shape;
    int M, K, N;
    double true_pct;
    bool has_iso;
    double iso_pct;
    int k;
    bool has_tile;
    int tile;
    int warmups, reps;
    bool has_stage[NUM_STAGES];
    double stage_ms[NUM_STAGES];
    Stats total;
    double gflops;
    double rel_err;
    bool has_recall;
    double recall;
};

[[noreturn]] void die(const char* what, const char* detail);
void* checked_malloc(size_t bytes);
void* read_file_exact(const char* path, size_t bytes);
Stats compute_stats(const std::vector<double>& v);
double rel_err(const float* C, const float* R, size_t n);
double outlier_recall(const int* S, int k, const int32_t* true_set, int n_true);
void write_csv_row(const char* path, const CsvRow& r);
double stage_start_ms();
double stage_elapsed_ms(double start_ms);
