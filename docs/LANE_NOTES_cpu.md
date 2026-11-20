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
