# COVERAGE.md: Definition of done and validation procedure

This file defines what "complete" means. Round 4 validation is run against it, line by line. Nothing is PASS without evidence you can point to (file and filter).

## 1. Requirements

| # | Requirement | Evidence the validation must find | Lane |
|---|---|---|---|
| 1 | Correct serial FP32 baseline, timed | `fp32_serial` rows in group A, `rel_err` = 0 | A |
| 2 | Per-tensor INT8 as accuracy floor | `int8_pertensor` rows for all shapes and all 3 true levels, with `rel_err` | A, B |
| 3 | Mixed-precision GEMM with configurable isolated % | `mixed_*` rows for every p in the sweep (group B) | A, B |
| 4 | CPU: OpenMP plus explicit AVX2 INT8 inner loop | group C: scalar vs AVX2 vs AVX2+OpenMP | A |
| 5 | GPU: `__dp4a` INT8 kernel plus FP16-path outliers, cuBLAS FP32 alongside | GPU group A and B rows | B |
| 6 | Separate-pass vs fused merge | `mixed_sep*` and `mixed_fused*` rows for identical configurations | A, B |
| 7 | Seeded synthetic data with outlier statistics | identical data checksums in both lanes; t in {0.5, 2, 5} | A, B |
| 8 | Correctness | golden checksums all equal (SPEC 13); `fp32_omp` equals `fp32_serial`; `rel_err` on every row | A, B |
| 9 | Sweep of parallel resources and problem size | CPU thread scaling (group D); GPU tile sweep (group C); five shapes in group B | A, B |
| 10 | Profiler-driven optimization with before/after | group E rows, profiler output under `results/<lane>/profile/`, the decision recorded in `docs/LANE_NOTES_<lane>.md` | A, B |
| 11 | Machine, thread and GPU counts | `env_*.txt` files and `docs/env_<lane>.md` | A, B |
| 12 | Timing method | SPEC 8 followed; min, max, stddev columns present | A, B |
| 13 | Where the time goes | stage columns filled for `mixed_*` rows | A, B |
| 14 | The answer to the project question | table T3 (section 6) | A |

## 2. Run matrix

Shape names, t levels and p levels are from SPEC 5. Defaults: CPU `threads` = T, GPU `tile` = 16. The thread list L is the powers of two from 1 up to T, plus T if T is not a power of two.

### CPU lane

| Group | Variants | Configurations | Rows |
|---|---|---|---|
| A baselines | `fp32_serial`, `fp32_omp` | all 5 non-golden shapes, t = 2 | 10 |
| | `int8_pertensor` | all 5 shapes, t in {0.5, 2, 5} | 15 |
| B sweep | `mixed_sep_avx2`, `mixed_fused_avx2` | 5 shapes x 3 t x 7 p | 210 |
| C ladder | `mixed_sep_scalar` (threads 1), `mixed_sep_avx2` (threads 1), `mixed_sep_avx2` (threads T) | shapes sq1024 and sq2048, t = 2, p in {0, 5} | 12 |
| D scaling | `fp32_omp`, `mixed_sep_avx2`, `mixed_fused_avx2` | threads in L, shapes sq2048 and sq4096, t = 2, p = 5 (mixed only) | 6 x \|L\| |
| E optimization | `<variant>_opt1` | sq2048 and sq4096, t = 2, p in {1, 5, 10} | 6 |
| G golden | see SPEC 13 | golden shape | n/a |

### GPU lane

| Group | Variants | Configurations | Rows |
|---|---|---|---|
| A baselines | `fp32_cublas` | all 5 shapes, t = 2 | 5 |
| | `int8_pertensor` | all 5 shapes, t in {0.5, 2, 5} | 15 |
| B sweep | `mixed_sep`, `mixed_fused` | 5 shapes x 3 t x 7 p | 210 |
| C tile sweep | `mixed_sep`, `mixed_fused` | tile in {8, 16, 32}, shapes sq2048 and sq4096, t = 2, p = 5 | 12 |
| E optimization | `<variant>_opt1` | sq2048 and sq4096, t = 2, p in {1, 5, 10} | 6 |
| G golden | see SPEC 13 | golden shape | n/a |

Group C of the GPU lane has no scalar ladder; the GPU ladder is `int8_pertensor` then `mixed_sep` then `mixed_fused`, taken from groups A and B.

Each CSV is written by the group's job script to `results/<lane>/<group>.csv`, for example `results/cpu/B_sweep.csv`.

## 3. Defaults for repetitions

3 warmups and 10 repetitions, except `fp32_serial` and `mixed_sep_scalar`, which use 1 warmup and 3 repetitions.

## 4. Validation procedure (Round 4)

For each lane, in this order:

1. Confirm every expected file exists and every CSV header equals the header in SPEC 10 exactly.
2. Count rows per group and compare with section 2. List each missing configuration. Flag duplicates.
3. Run the sanity checks below on every row.
4. Produce a verdict table: requirement number, PASS / FAIL / PARTIAL, evidence (file, filter, count).
5. For everything missing or flagged, output the exact commands to produce it (using the rules in CLAUDE.md section 2).
6. Never mark PASS without evidence. Never explain away an anomaly; report it.

### Sanity checks

| Id | Check | If violated |
|---|---|---|
| S1 | `rel_err` of `fp32_serial` and `fp32_omp` is exactly 0 | stop; the reference or the harness is wrong |
| S2 | `rel_err` of `fp32_cublas` is below 1e-5 (about 1e-3 would mean TF32 is active) | stop; fix math mode |
| S3 | For each (lane, shape, t): `rel_err` at p = 0 is at least `rel_err` at the largest p | flag |
| S4 | `outlier_recall` = 1.0 whenever p is at least t (the x50 outliers are well separated) | flag |
| S5 | `total_ms_stddev` / `total_ms_median` above 0.10 | flag for re-run |
| S6 | `mixed_sep*` and `mixed_fused*` have identical `rel_err` for the same configuration | flag |
| S7 | CPU and GPU `rel_err` match for the same (variant family, shape, t, p) | flag |
| S8 | Sum of the stage medians is within 5% of `total_ms_median` | flag |
| S9 | `gflops` equals 2*M*K*N / (total_ms_median * 1e6) within rounding | flag |
| S10 | `k_isolated` equals floor(K * p / 100 + 0.5) | flag |

## 5. Group E (profile-driven optimization)

- The profiling commands, the profiler output (saved under `results/<lane>/profile/`), and the reasoning go into `docs/LANE_NOTES_<lane>.md`.
- The optimization must not change output bits. Re-run the golden check for `<variant>_opt1`. If the optimization cannot keep bit-identity, ask the human first.
- The report needs before and after numbers for the same configurations: the originals come from group B, the new ones from group E.

## 6. Final assets (Round 6)

Lane A writes `analysis/make_assets.py` (pandas and matplotlib, run only with the human's permission) that reads `results/cpu/*.csv` and `results/gpu/*.csv` and writes to `analysis/out/`:

- `headline_numbers.csv`: every number that will be quoted in NOTES.md, with its source file and filter. NOTES.md may only contain numbers that appear here or in a CSV.
- Tables (CSV and Markdown): T1 environment; T2 run counts per group against section 2; **T3 headline**: for each (lane, shape, t) and for p in {1, 5, 10}, the `rel_err`, `rel_err` relative to p = 0 and to `int8_pertensor`, the absolute GFLOP/s, the throughput cost as a percentage relative to p = 0, and relative to `int8_pertensor`; plus the knee (below). T4 stage breakdown at p in {1, 5, 10}; T5 before/after of the optimization.
- Figures (PNG and PDF, readable at half-column width, axes labeled with units, same color for the same variant across figures):
  - F1 GFLOP/s vs p per shape (t = 2), one figure per lane, with `int8_pertensor` and the fp32 baseline as horizontal reference lines.
  - F2 `rel_err` vs p (log y), one line per t, with the `int8_pertensor` level as a reference line.
  - F3 trade-off scatter: `rel_err` (x, log) vs GFLOP/s (y), points labeled by p, for sq2048 and sq4096, t = 2, both lanes.
  - F4 stacked stage times vs p for `mixed_sep`, sq4096, t = 2, per lane.
  - F5 CPU thread scaling: speedup and parallel efficiency vs threads (group D).
  - F6 ladder: CPU scalar, AVX2, AVX2+OpenMP (group C) and GPU `int8_pertensor`, `mixed_sep`, `mixed_fused`.
  - F7 GPU tile sweep.
  - F8 skinny vs square: throughput relative to p = 0, vs p.
  - F9 `outlier_recall` vs p, one line per t.
  - F10 CPU vs GPU: throughput relative to p = 0, vs p.
  - F11 before/after of the optimization.
- Headline set for the 4-minute talk (separate files, large fonts): F3, F4, F1 (sq4096), F6.

**Knee definition (analysis convention):** the knee for a (lane, shape, t) is the smallest p in the sweep whose `rel_err` is at most 1.25 times the minimum `rel_err` over all p for that (lane, shape, t). The humans may change this convention; if so, state the change in the report.

**Throughput cost** at p = (1 - gflops(p) / gflops(p = 0)) x 100%, using `mixed_sep` and `mixed_fused` separately.

### NOTES.md (Lane A writes it)

Sections, in order: 1) findings in one paragraph per question, 2) environment, 3) what each variant does in plain words (from the two LANE_NOTES files), 4) results by question with the tables, 5) the optimization before/after, 6) anomalies and flagged checks, 7) limitations and claims we must not make (synthetic data, one outlier magnitude x50, one seed, host-side top-k on GPU, stage-boundary sync in GPU timings, node sharing status), 8) index of every figure and table with its source CSV, 9) exact commands to reproduce.

Rules for NOTES.md: every number carries a source tag in the form `[file | filter]`. Observations and hypotheses are labeled separately. Anything unexplained is listed as unexplained. No sentence claims more than the data shows.

### LANE_NOTES_<cpu|gpu>.md (each lane writes its own)

Plain-language description of each implemented variant (what the loops or kernels do), the exact build flags used, the Round 5 decision (profiler output summary, chosen optimization, why), every anomaly encountered, and known limitations. Written so a teammate can turn it into report text.
