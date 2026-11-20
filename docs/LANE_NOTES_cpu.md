# LANE_NOTES_cpu.md: Lane A (CPU) notes

This file is written in pieces; the full set of sections required by COVERAGE section 6 is completed in Round 6. So far it holds the decisions that must not be lost.

## Known limitations

- **`int8_pertensor` is a plain-loop kernel, the mixed variants use AVX2 intrinsics.** SPEC 7 specifies it this way. In the smoke run (`data/smoke/smoke.csv`, sq2048, t = 2, 16 threads) `mixed_sep_avx2` at p = 0 reached 885.670 GFLOP/s and `int8_pertensor` 471.169 GFLOP/s, so any throughput comparison of a mixed variant against `int8_pertensor` compares different kernel styles and can show a negative cost. Human ruling of 2026-11-20: SPEC stays as is; the main throughput reference on both lanes is p = 0; `int8_pertensor` is the accuracy floor, and its CPU throughput is reported only with this plain-loop caveat. Whether to add an AVX2 per-tensor variant is undecided.
