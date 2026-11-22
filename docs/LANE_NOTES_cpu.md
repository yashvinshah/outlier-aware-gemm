# LANE_NOTES_cpu.md: Lane A (CPU) notes

This file is written in pieces; the full set of sections required by COVERAGE section 6 is completed in Round 6. So far it holds the decisions that must not be lost.

## Known limitations

- **`int8_pertensor` is a plain-loop kernel, the mixed variants use AVX2 intrinsics.** SPEC 7 specifies it this way. In the smoke run (`data/smoke/smoke.csv`, sq2048, t = 2, 16 threads) `mixed_sep_avx2` at p = 0 reached 885.670 GFLOP/s and `int8_pertensor` 471.169 GFLOP/s, so any throughput comparison of a mixed variant against `int8_pertensor` compares different kernel styles and can show a negative cost. Human ruling of 2026-11-20: SPEC stays as is; the main throughput reference on both lanes is p = 0; `int8_pertensor` is the accuracy floor, and its CPU throughput is reported only with this plain-loop caveat. Whether to add an AVX2 per-tensor variant is undecided.
- **sq1024 timing noise.** At about 2.3 to 3.2 ms total per repetition (`B_sweep.csv`, `shape=sq1024`, total median column), 31 of the 42 sq1024 rows of group B have `total_ms_stddev` / `total_ms_median` between 0.103 and 0.148 (COVERAGE S5 threshold 0.10), so the sq1024 GFLOP/s values carry that noise. A second run of the sq1024 slice flagged the same shape again (human decision 2026-11-20: accept, no third run).

## Anomalies

COVERAGE S5 flags (stddev / median above 0.10), accepted as reported by the human on 2026-11-20 after one re-run; 33 rows in the final CSVs. The cause is unexplained.

- `A_baselines.csv`: `fp32_omp`, sq1024, t = 2, 16 threads (ratio 0.141). 1 row.
- `B_sweep.csv`, shape sq1024, 16 threads, 31 rows:
  - t = 0.5: `mixed_sep_avx2` and `mixed_fused_avx2` at p = 0.5, 1, 2, 5, 10, 20 (12 rows).
  - t = 2: `mixed_sep_avx2` at p = 0, 0.5, 2, 5, 10, 20 and `mixed_fused_avx2` at p = 2, 5, 10, 20 (10 rows).
  - t = 5: `mixed_sep_avx2` at p = 0, 5, 10, 20 and `mixed_fused_avx2` at p = 0, 1, 5, 10, 20 (9 rows).
- `B_sweep.csv`: `mixed_fused_avx2`, skinny16, t = 5, p = 1, 16 threads, ratio 0.491 (median 6.536 ms, max 17.291 ms); it was also flagged before the re-run (ratio 0.123). 1 row.
- Groups C and D: no flagged rows.

## Round 5 profile (`run_profile.sh` on c5, sq4096, t = 2, 16 threads; user-space `perf`, paranoid level 2)

Facts from the pasted `perf stat` and `perf report` output (files `results/cpu/profile/stat_*.txt`, `report_*.txt`):

| Configuration | cycles:u | instructions:u | IPC | cache-misses (share of refs) | `int8_tile_avx2` | `outlier_tile_avx2` |
|---|---|---|---|---|---|---|
| `mixed_sep_avx2` p = 5 | 99.290e9 | 327.178e9 | 3.30 | 2.381e9 (6.598%) | 78.99% | 11.99% |
| `mixed_fused_avx2` p = 5 | 106.984e9 | 327.176e9 | 3.06 | 2.121e9 (5.901%) | 79.36% | 12.51% |
| `mixed_sep_avx2` p = 0 | 99.251e9 | 327.154e9 | 3.30 | 0.445e9 (1.429%) | 88.25% | not listed |
| `mixed_sep_avx2` p = 0.5 | 99.195e9 | 327.747e9 | 3.30 | 0.562e9 (1.749%) | 85.62% | 0.81% |

- Fused and sep execute the same number of instructions (327.176e9 vs 327.178e9); the fused run took more cycles (IPC 3.06 vs 3.30). The merge pass is 0.90% of samples in sep (`merge_plain`) and 0.13% in fused (`fused_gemm_avx2`), so fusing can remove at most about 1% of the work. Both variants call `int8_tile_avx2` and `outlier_tile_avx2` as separate symbols.
- Process-to-process spread (same configuration, separate processes, `data/perf/repeat_runs.csv`, not committed): `mixed_sep_avx2` p = 0 total medians 153.712 to 165.398 ms over 7 processes (int8 stage 139.292 to 150.971 ms); p = 5 totals 163.808 to 169.963 ms over 5 processes (int8 stage 131.443 to 138.098 ms). The `B_sweep.csv` row at p = 0, t = 2 (total 175.515 ms, int8 161.859 ms) is above all 7 of the p = 0 repeats. No systematic p = 0 penalty was found; the cause of that single high row is unexplained (hypotheses only: process-level variation, drift over the job).
- Decision (optimization), approved by the human: `mixed_sep_avx2_opt1`, a 4-row x 16-column register-blocked INT8 micro-kernel with the X channel pairs pre-packed as 32-bit words (`cpu/int8_avx2_opt1.cpp`). The dominant cost it targets is `int8_tile_avx2` (79 to 88% of samples). `mixed_sep_avx2` and the other variants are unchanged. The `pack_x_pairs` time is counted in the quantize stage. The gain is reported only as measured (group E against group B), not as an estimate. Status: code written, not yet built or run.
- Group E job design (human ruling): each opt1 run is followed in the same job by the base `mixed_sep_avx2` run of the same configuration (`results/cpu/E_base.csv`); the before/after comparison uses these same-job base rows, and `B_sweep.csv` rows are context only. `E_opt.csv` keeps the six opt1 rows.
- Round 5 result for `mixed_sep_avx2_opt1` (bit identity: golden lines for p in {0, 2, 10}, threads in {1, 16} equal the base lines; `compare_c` on sq1024 and skinny16, p = 5, 0 differing elements). Same-job rows (`E_opt.csv` vs `E_base.csv`): INT8 stage 4.4% to 10.9% faster in 5 of 6 rows and 2.7% slower in sq2048 p = 5; one process per row, so most of this is inside the process-to-process spread (3.7% to 7.4% at sq4096 p = 5 and p = 0). Round 5 stops here with this one optimization.
- Profile of `mixed_sep_avx2_opt1` (`results/cpu/profile/stat_mixed_sep_avx2_opt1_p5.txt`, `report_mixed_sep_avx2_opt1_p5.txt`; sq4096, t = 2, p = 5, 16 threads) against `mixed_sep_avx2` (`stat_mixed_sep_avx2_p5.txt`): instructions:u 268.639e9 vs 327.178e9 (-17.9%), cycles:u 112.500e9 vs 99.290e9 (+13.3%), IPC 2.39 vs 3.30, task-clock 32,734.29 vs 33,503.70 ms (-2.3%), elapsed 2.230 vs 2.304 s (-3.2%), cache-references 22.746e9 vs 36.084e9, cache-misses 2.487e9 vs 2.381e9. Observation: perf reports 3.437 GHz for the opt1 run and 2.964 GHz for the base run, so cycles are not a time proxy between these two runs. Hypotheses, none tested: (1) the INT8 stage is limited by data supply (a 16-column micro-tile touches half a cache line of packed W per channel pair), not by instruction issue; (2) register pressure or loop structure in `micro_tile`; (3) the clock-rate difference is a node effect, not a kernel effect.
- p = 0 reference rows: kept as they are (human ruling: option a). The human may later propose replicate jobs of groups A and B with the median per configuration; no code for that exists.
- `data/perf/perf_runs.csv` and `repeat_runs.csv` are to be copied into `results/cpu/profile/` on ARC by the human, so that they are in git.
