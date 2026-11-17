# SPEC.md: Frozen technical specification

Everything in this file is decided. Both lanes implement exactly this. Anything not covered here is a question for the humans (CLAUDE.md rule 3).

## 1. Goal

Measure, on CPU (OpenMP + AVX2) and on GPU (CUDA), how much GEMM throughput is lost to preserve numerical accuracy in a mixed-precision INT8 GEMM that isolates outlier channels into FP16.

The question this project answers:

> **How much throughput are we losing for accuracy? Specifically, how much speed must we give up to keep 1%, 5%, or 10% of channels in FP16 to preserve numerical accuracy in a mixed-precision INT8 GEMM?**

Wording note: the proposal speaks of an "outlier threshold". We implement it as the isolated percentage p (keep the top p% of channels by magnitude in FP16). This makes "1%, 5%, 10% of channels" directly sweepable. State this in the report.

## 2. Terms

- **GEMM:** C = X * W. X is M x K (activations), W is K x N (weights), C is M x N. All float32, row-major.
- **Channel:** an index c in [0, K). It names column c of X and row c of W.
- **True outlier percentage t:** a property of the generated data. t% of the channels of X are scaled by OUTLIER_SCALE.
- **Isolated percentage p:** an algorithm parameter. The p% of channels with the largest column magnitude in X are computed in FP16. All other channels are computed in INT8.
- **k** = floor(K * p / 100 + 0.5), computed in double. **n_true** = floor(K * t / 100 + 0.5), same formula.
- **S:** the set of k isolated channels. **Nrm:** the other K - k channels ("normal").
- **T (CPU):** the number of physical cores of the allocated node, fixed in Round 0c.

## 3. Repository layout and ownership

```
outlier-gemm/
  CLAUDE.md SPEC.md COVERAGE.md LANE_CPU.md LANE_GPU.md   humans own, frozen
  .gitignore                                              Lane A
  common/     Lane A   params.h, fp32_gemm.h/.cpp, gen_data.cpp, compare_c.cpp, Makefile
  cpu/        Lane A   CPU benchmark program
  gpu/        Lane B   GPU benchmark program
  scripts/cpu/  Lane A     scripts/gpu/  Lane B
  data/       generated on ARC, git-ignored
  results/cpu/  results/gpu/    CSV files, env files, golden checksums, profiler output (committed, they are small)
  docs/       env_<lane>.md and LANE_NOTES_<lane>.md, each written by its own lane
  analysis/   Lane A (Round 6)
  NOTES.md    Lane A (Round 6)
```

## 4. Data

### 4.1 Constants (`common/params.h`, Lane A creates it with exactly this content)

```cpp
// common/params.h: constants fixed by SPEC.md. Do not change without a SPEC change.
#pragma once
#include <cstdint>
constexpr float    OUTLIER_SCALE = 50.0f;   // true outlier channels of X are multiplied by this
constexpr double   W_SIGMA       = 0.02;    // standard deviation of W entries
constexpr uint64_t SEED_X        = 1234;
constexpr uint64_t SEED_W        = 5678;
constexpr uint64_t SEED_OUTLIERS = 91011;
constexpr float    INT8_MAX_F    = 127.0f;  // INT8 scale divisor
constexpr int      DEFAULT_TILE  = 16;      // GPU default
```

### 4.2 Generator algorithm (`common/gen_data.cpp`)

```
rng      = std::mt19937_64(seed)
u01()    = ((rng() >> 11) + 0.5) * (1.0 / 9007199254740992.0)       // double, in (0,1)
pair     : u1 = u01(); u2 = u01(); r = sqrt(-2.0 * log(u1)); a = 2.0 * pi * u2
           z0 = r * cos(a); z1 = r * sin(a)                         // pi = 3.14159265358979323846
fill     : z0, z1 of pair 0, then z0, z1 of pair 1, and so on

X base   : M*K samples, seed SEED_X.   X[i*K + c] = (float)z
W        : K*N samples, seed SEED_W.   W[c*N + j] = (float)(W_SIGMA * z)

True outlier set (depends only on K and t):
  idx = [0, 1, ..., K-1]
  rng = std::mt19937_64(SEED_OUTLIERS)
  for i = K-1 down to 1:  j = rng() % (i + 1);  swap(idx[i], idx[j])
  set = sorted ascending copy of idx[0 .. n_true-1]        (sets for different t are nested)

Outlier scaling: for each c in set and every row i:  X[i*K + c] = X[i*K + c] * OUTLIER_SCALE   (float multiply)

Reference: ref_C = fp32_gemm_omp(X, W), computed after the outlier scaling.
```

### 4.3 Files

Raw little-endian, row-major, no header. The tag `{T}` for the true percentage is `0p5` for 0.5, `2` for 2, `5` for 5.

| File | Content |
|---|---|
| `X_{M}x{K}_t{T}.f32` | X after outlier scaling, float32 |
| `W_{K}x{N}.f32` | W, float32 |
| `ref_{M}x{K}x{N}_t{T}.f32` | reference C, float32 |
| `outliers_K{K}_t{T}.i32` | true outlier channels, int32, ascending |

`gen_data --M --K --N --true-pct --data-dir` writes the four files for one (shape, t).

### 4.4 Checksums

FNV-1a 64-bit over the file bytes: offset basis 14695981039346656037, prime 1099511628211, printed as 16 lowercase hex digits. `gen_data` prints one line per file: `CHECKSUM <filename> <hex>`. Before any benchmarking, the two humans compare these lines between their ARC accounts. Any difference stops everything.

## 5. Shapes and sweep values

| Name | M | K | N |
|---|---|---|---|
| golden | 256 | 256 | 256 |
| sq1024 | 1024 | 1024 | 1024 |
| sq2048 | 2048 | 2048 | 2048 |
| sq4096 | 4096 | 4096 | 4096 |
| skinny16 | 16 | 4096 | 4096 |
| skinny128 | 128 | 4096 | 4096 |

- True percentages t: 0.5, 2, 5. (Golden uses t = 2 only.)
- Isolated percentages p: 0, 0.5, 1, 2, 5, 10, 20. (Golden uses p = 0, 2, 10.)
- Data files are needed for every (shape, t) above, plus W for every (K, N).

## 6. The algorithm (numerics)

Input X (M x K), W (K x N), p. Output C (M x N). The steps and their order are fixed.

```
Step 1  SELECT   (skipped when k = 0)
  colmax[c] = max over rows i of |X[i][c]|, for every channel c.
  Order channels by colmax descending; break ties by channel index ascending.
  S   = the first k channels of that order, then sorted ascending by index.
  Nrm = all other channels, ascending by index.
  (When k = 0: S is empty and Nrm is all channels.)

Step 2  QUANTIZE
  Normal channels:
    sx[i] = (max over c in Nrm of |X[i][c]|) / 127.0f        float division
    sw[j] = (max over c in Nrm of |W[c][j]|) / 127.0f
    If Nrm is empty or the max is 0, the scale is 1.0f.
    Xq[i][c] = (int8) clamp( roundf( X[i][c] / sx[i] ), -127, 127 )     c in Nrm
    Wq[c][j] = (int8) clamp( roundf( W[c][j] / sw[j] ), -127, 127 )     c in Nrm
    roundf rounds half away from zero. The division is IEEE float division.
  Outlier channels (c in S):
    Xh[i][c] = half_rn(X[i][c]),  Wh[c][j] = half_rn(W[c][j])           round-to-nearest-even FP32 -> FP16
  Gathering the channels into compact arrays and any packing or transposing the kernel needs
  is part of this step and is timed with it.

Step 3  INT8 GEMM
  acc[i][j] = sum over c in Nrm of (int32)Xq[i][c] * (int32)Wq[c][j]       exact, any order
  Ci8[i][j] = ((float)acc[i][j] * sx[i]) * sw[j]                           exactly this order of operations

Step 4  OUTLIER GEMM   (skipped when k = 0)
  Co[i][j] = 0.0f
  for c in S in ascending order:  Co[i][j] = Co[i][j] + ((float)Xh[i][c] * (float)Wh[c][j])
  float32 accumulation, ascending channel order, a separate multiply and add.
  (The product of two FP16 values is exact in float32.)

Step 5  MERGE   (skipped when k = 0)
  C[i][j] = Ci8[i][j] + Co[i][j]
  When k = 0: C = Ci8.
```

**Per-tensor variant (`int8_pertensor`):** sx = (max over all of |X|) / 127.0f and sw = (max over all of |W|) / 127.0f, both scalars. Every channel is INT8 (k = 0, no select). Steps 2 and 3 are otherwise identical, including `((float)acc * sx) * sw`.

**Parallelization rules:** a thread owns whole output rows (or whole (row, column-block) tiles). The sum over channels is never split across threads or reordered, so the floating-point order in Step 4 is fixed.

**Consequence:** because there is no floating-point contraction (section 12) and every order of operations is fixed, CPU and GPU outputs are expected to be **bit-identical** for the same inputs, and so are all CPU variants among themselves. A mismatch is a bug to report, never a tolerance to loosen.

## 7. Variants

Variant names are exactly these strings.

**CPU lane**

| Variant | Threads | What it is |
|---|---|---|
| `fp32_serial` | 1 | `fp32_gemm_serial`: loop order i, k, j (i outer, k middle, j inner); one thread; no OpenMP. Its output is the reference. |
| `fp32_omp` | T or sweep | `fp32_gemm_omp`: identical loops with `#pragma omp parallel for` over i. Output bit-identical to `fp32_serial`. |
| `int8_pertensor` | T | Per-tensor INT8 (section 6). Plain loops, OpenMP over rows. |
| `mixed_sep_scalar` | any | Section 6 as five separate stages. The INT8 GEMM, outlier GEMM and merge stages are plain loops in a file compiled without auto-vectorization. |
| `mixed_sep_avx2` | any | Same stages. The INT8 GEMM and outlier GEMM inner loops use explicit AVX2 intrinsics. |
| `mixed_fused_avx2` | any | Steps 3, 4, 5 in one loop nest. `Ci8` and `Co` are never stored as M x N arrays. Final C is written once. Steps 1 and 2 are separate stages as above. |

In all three `mixed_*` variants the select and quantize stages are shared code compiled normally (-O3). Only the GEMM-style stages differ, so the ladder isolates them.

**GPU lane**

| Variant | What it is |
|---|---|
| `fp32_cublas` | cuBLAS SGEMM with math mode `CUBLAS_DEFAULT_MATH` (no TF32). Row-major handled by swapping operands. |
| `int8_pertensor` | Per-tensor INT8 with a `__dp4a` kernel. |
| `mixed_sep` | Select (column-max kernel, then top-k on the host, section 8), quantize/gather kernels, INT8 `__dp4a` GEMM kernel writing a float `Ci8` array, outlier GEMM kernel writing `Co`, merge kernel writing C. |
| `mixed_fused` | Select and quantize as above. One kernel does steps 3, 4, 5. |

All GPU GEMM-style kernels use thread blocks of TILE x TILE threads, one thread per output element, with `TILE` a C++ template parameter in {8, 16, 32}. The host launcher switches on `--tile`. Default 16. No tensor cores, no cuBLASLt, no CUTLASS. cuBLAS is used only for `fp32_cublas`.

**Optimized variants (Round 5):** named `<variant>_opt1`. The original stays in the code.

## 8. Stages and timing

Stages and what each contains:

| Stage | Column | Contents |
|---|---|---|
| select | `t_select_ms` | Step 1: column max, top-k, building the S and Nrm index lists. GPU: the column-max kernel, the copy of `colmax` to the host, the host `std::stable_sort`, and the copy of the index lists back are all included. |
| quantize | `t_quant_ms` | Step 2: scales, quantization, gathering and packing, FP16 conversion of outlier channels. |
| int8 | `t_int8_ms` | Step 3 including the dequantize to float. For fused variants: steps 3, 4 and 5 together. |
| outlier | `t_outlier_ms` | Step 4. |
| merge | `t_merge_ms` | Step 5. |

- A stage column is **blank** if the stage does not exist in that variant (all five for `fp32_*`; select, outlier and merge for `int8_pertensor`; outlier and merge for fused variants). It is **0.0** if the stage exists but was skipped because k = 0.
- Baselines (`fp32_*`) have a single stage, and `total_ms` is that stage.
- All buffers are allocated before timing. File reading is not timed. GPU inputs are already on the device before timing, and copies of X, W and C between host and device are not timed. The small `colmax` and index-list copies inside select are timed.
- CPU timer: `omp_get_wtime()`. GPU timer: host wall clock (`std::chrono::steady_clock`) with `cudaDeviceSynchronize()` before starting and before stopping each stage timer. cudaEvent timing is not used (one mechanism only).
- Per repetition: record each stage time, and `total` = the sum of the stage times of that repetition. Report the median of each stage column and the median, minimum, maximum and population standard deviation of `total`. (Stage medians do not exactly sum to the total median.) GPU totals include the stage-boundary synchronization overhead; state this in the report.
- Defaults: 3 warmups, 10 timed repetitions. Slow runs (`fp32_serial`, `mixed_sep_scalar`) use 1 warmup and 3 repetitions.

## 9. Metrics

- `rel_err` = sqrt( sum (C - R)^2 ) / sqrt( sum R^2 ) over all elements, accumulated in double, where R is the reference file. Computed once after timing, from the C of the last timed repetition.
- `gflops` = 2 * M * K * N / (total_ms_median * 1e6). This is the effective rate against the full dense operation count, for every variant, so curves are comparable across variants.
- `outlier_recall` = |S intersect true set| / |true set|. Filled for `mixed_*` only. For p = 0 it is 0.0.

## 10. CSV

One header row, written once when the file is created, exactly:

```
lane,variant,shape,M,K,N,true_out_pct,iso_out_pct,k_isolated,threads,tile,warmups,reps,t_select_ms,t_quant_ms,t_int8_ms,t_outlier_ms,t_merge_ms,total_ms_median,total_ms_min,total_ms_max,total_ms_stddev,gflops,rel_err,outlier_recall
```

- `lane` is `cpu` or `gpu`. `shape` is the name from section 5. `true_out_pct` is the number (0.5, 2, 5).
- `iso_out_pct` and `k_isolated` are blank for `fp32_*` and `int8_pertensor`.
- `threads` is filled for the CPU lane and blank for GPU. `tile` is filled for the GPU lane and blank for CPU.
- Number formats: times `%.6f`, `gflops` `%.3f`, `rel_err` `%.9e`, `outlier_recall` `%.6f`.
- The program appends one row per run. If the file does not exist it writes the header first.

## 11. Benchmark program interface

Each lane builds one executable, `cpu/bench_cpu` or `gpu/bench_gpu`, that runs exactly one configuration:

```
--variant <name>   --shape <name>   --M <int> --K <int> --N <int>
--true-pct <0.5|2|5>
--iso-pct <float>     required for mixed_* only
--threads <int>       CPU only
--tile <8|16|32>      GPU only, default 16
--warmups <int>  --reps <int>
--data-dir <path>  --out-csv <path>
--write-c <path>      optional: write the final C (raw float32) of the last repetition and print
                      "C_CHECKSUM <hex>" (FNV-1a as in section 4.4)
```

Job scripts loop over configurations. One process per configuration.

## 12. Build flags

- CPU, all files except the scalar GEMM file: `g++ -std=c++17 -O3 -mavx2 -mf16c -fopenmp -ffp-contract=off`
- CPU scalar GEMM file (`cpu/int8_scalar.cpp`): `-std=c++17 -O2 -fno-tree-vectorize -fno-tree-slp-vectorize -fopenmp -ffp-contract=off`
- GPU: `nvcc -std=c++17 -O3 -arch=sm_<CC> --fmad=false -Xcompiler "-O3 -ffp-contract=off" -lcublas`, with `<CC>` taken from the Round 0b output.
- No `-ffast-math`, no `-march=native`, no `-mfma`. If the ARC CPU lacks AVX2 or F16C, or the GPU is below compute capability 6.1 (needed by `__dp4a`), stop and ask.

## 13. Golden cross-check

1. Shape golden, t = 2, p in {0, 2, 10}. Every program run uses `--write-c`.
2. CPU lane runs `int8_pertensor`, `mixed_sep_scalar`, `mixed_sep_avx2`, `mixed_fused_avx2`, each with threads in {1, T}. It also checks that `fp32_omp` output equals `fp32_serial` output. Results go to `results/cpu/golden/checksums_cpu.txt`, one line per run: `<variant> threads=<n> p=<p> <hex>`.
3. GPU lane runs `int8_pertensor`, `mixed_sep`, `mixed_fused`, each with tile in {8, 16, 32}. Results go to `results/gpu/golden/checksums_gpu.txt`, one line per run: `<variant> tile=<n> p=<p> <hex>`.
4. Pass criterion, all exact:
   - all CPU `mixed_*` checksums for the same p are equal to each other;
   - all GPU `mixed_*` checksums for the same p are equal to each other;
   - CPU and GPU `mixed_*` checksums for the same p are equal;
   - CPU and GPU `int8_pertensor` checksums are equal (p is not used);
   - `fp32_omp` equals `fp32_serial`.
5. On any mismatch: stop. Report which pairs differ. The humans commit the small C files, and `common/compare_c` reports the number of differing elements and the maximum absolute difference. Do not loosen the criterion.

## 14. Environment file

Every job script first writes `results/<lane>/env_<tag>.txt` containing: date, hostname, node name, scheduler job id and allocation size (only values the scheduler provides), `nproc`, `lscpu` (CPU lane), `nvidia-smi` (GPU lane), `g++ --version`, `nvcc --version` (GPU lane), loaded modules, the git commit hash of the checkout (`git rev-parse HEAD`, run by the job script on ARC), and the exact build flags in use.

## 15. Out of scope

Tensor cores, cuBLASLt, CUTLASS, INT4, FP8, real model weights or activations, per-batch dynamic thresholds, multi-GPU, MPI, file I/O time, host-to-device transfer time of X, W and C, on-device top-k, more than one outlier magnitude, more than one random seed. Mention the ones that limit conclusions in the report's limitations.

## 16. Change control

SPEC.md changes only by agreement of both humans. After a change, both sessions are told, and any affected results are re-run.

## 17. Clarifications (added Nov 17, 2026; answers to Lane A's spec review; binding on both lanes)

The numbers follow the order of Lane A's review list.

1. **Constants outside `params.h`.** `params.h` stays exactly as in section 4.1. Other fixed constants (pi, 2^53, the integer clamp bound 127, the 64-byte alignment) are named `constexpr` values in the file that uses them. FNV-1a lives in one header, `common/checksum.h` (header-only, inline functions, written by Lane A in the first `common/` drop). `gen_data`, `compare_c`, `cpu/util` and `gpu/util` all include it, so the checksums cannot differ between lanes.
2. **FP16 outlier values.** Rounding an FP32 value to FP16 and storing the result back as a float is allowed. It is numerically identical to storing the half. This is how `cpu/int8_scalar.cpp` (built without `-mf16c`) gets its inputs: the quantize stage rounds and stores floats.
3. **Order of the `rel_err` accumulation.** Serial, on the host, row-major, ascending element index. Two separate double accumulators: numerator = sum of (C - R)^2, denominator = sum of R^2. `rel_err = sqrt(num) / sqrt(den)`. Both lanes use exactly this, so `rel_err` can be compared for exact equality. On the GPU lane, C is copied to the host after timing and the sum is computed on the host.
4. **OpenMP in select and quantize (CPU lane).** Yes. The `--threads` value applies to every stage. No floating-point reductions across threads: column max, row max and column max (scales) are exact, so any order is fine. Quantize and pack are parallel over rows or channels.
5. **Golden checksum lines.** `fp32_*` and `int8_pertensor` use `p=NA`. Output C files go to `data/golden_out/cpu/` and `data/golden_out/gpu/` (git-ignored). They are named `<variant>_threads<n>_p<p>.f32` for CPU and `<variant>_tile<n>_p<p>.f32` for GPU, with `pNA` when p does not apply. Only the checksum files under `results/<lane>/golden/` are committed. On a mismatch, the humans copy only the files in question (SPEC 13 step 5).
6. **Golden data.** The golden shape uses t = 2 only (and W_256x256). Every other shape gets all three t values.
7. **When the helper scripts are written.** `write_env.sh` and `run_golden.sh` are written at the start of Round 2. Round 1 only gives plain per-variant commands for a first small run.
8. **CSV formats for the remaining columns.** `true_out_pct` and `iso_out_pct` are printed with `%g`. `M`, `K`, `N`, `k_isolated`, `threads`, `tile`, `warmups` and `reps` are integers. The file tag for t is the `--true-pct` string with "." replaced by "p" (0.5 -> `0p5`, 2 -> `2`, 5 -> `5`).
9. **Warmups and reps for golden and smoke runs.** Same defaults as SPEC 8 (3 and 10; 1 and 3 for `fp32_serial` and `mixed_sep_scalar`).
10. **S1 (`rel_err` of `fp32_serial` = 0).** It relies on `fp32_serial` and `fp32_omp` being bit-identical. Both live in the same file with the same flags, and each output element sums over k in the same order. Round 2 verifies it. If the value is not exactly 0, stop and report; do not loosen the check.
11. **Run time limits.** Round 0c asks for the scheduler's maximum walltime. `fp32_serial` at sq4096 is about 137 GFLOP on one thread and may be the longest single run. If a group does not fit in the limit, split it into one job per shape.
12. **Which variant gets `_opt1`.** Decided in Round 5 from the profile data and approved by the humans before any code is written.
13. **Git in the env script.** The job scripts may call `git rev-parse HEAD`. The humans run the scripts on ARC. The ban in CLAUDE.md rule 1 applies to Claude Code itself, which never runs git.
14. **Aligned allocation.** `posix_memalign` with 64-byte alignment (CPU lane). Confirm on the first ARC build.
