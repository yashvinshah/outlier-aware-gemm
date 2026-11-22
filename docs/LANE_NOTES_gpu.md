# LANE_NOTES_gpu.md: Lane B (GPU) notes

Written for a teammate who will turn it into report text. Every number has a source tag in the form `[file | filter]`; nothing here comes from memory of a chat. Observations and hypotheses are labeled. Files are under `results/gpu/` unless a path says otherwise.

## 1. What each variant does

All variants read the same data files as the CPU lane and run on one GPU (RTX 4060 Ti, compute capability 8.9, `docs/env_gpu.md` facts 12 and 13). The matrix product is C = X (M x K) times W (K x N), row-major float32.

- **`fp32_cublas`** (`gpu/fp32_cublas.cu`). One call of cuBLAS `cublasSgemm` with `CUBLAS_DEFAULT_MATH`, so TF32 is not used. cuBLAS is column-major, so the operands are swapped (W first, then X) to get a row-major C. The handle is created once before timing.
- **`int8_pertensor`** (`gpu/pertensor.cu`, `gpu/quant.cu`, `gpu/int8_gemm.cu`). One scale for all of X and one for all of W, each the global max of the absolute values divided by 127 (1.0 if the max is 0). Every channel is quantized to INT8 and packed four per 32-bit word. The INT8 GEMM then uses `__dp4a` and dequantizes each output as `(acc * sx) * sw`. This is the accuracy floor. The global-max kernels are counted in the quantize stage.
- **`mixed_sep`** (five stages, one timed stage each):
  1. *select* (`gpu/select.cu`): a kernel computes the per-channel column max of |X|; the host copies it back, does a stable sort (descending, ties by ascending channel index) and takes the top k as the isolated set S (k = floor(K * p / 100 + 0.5)); the rest are the normal channels Nrm; both lists go back to the device. Top-k is on the host.
  2. *quantize* (`gpu/quant.cu`): per-row scale for X and per-column scale for W, each the max over the normal channels divided by 127; the normal channels are quantized (division, `roundf`, clamp to plus or minus 127) and packed four per word (W stored transposed); the isolated channels are rounded to FP16 and gathered into compact arrays.
  3. *int8* (`gpu/int8_gemm.cu`): a tiled kernel, one thread per output element, `__dp4a` over the packed words, exact integer sums, dequantized as above.
  4. *outlier* (`gpu/outlier_merge.cu`): a tiled float32 GEMM over the k isolated channels, walked in ascending channel order, with a separate multiply and add (no fused multiply-add).
  5. *merge*: C = INT8 result + outlier result. When k = 0, stages 4 and 5 are skipped and the INT8 result is C.
- **`mixed_fused`** (`gpu/fused.cu`). Same select and quantize stages. Then one kernel does the INT8 loop, then the outlier loop, then writes C once, so the two M x N intermediate arrays are never stored. The loops are copies of the separate kernels, so the bits are identical.
- **`mixed_sep_opt1`** (`gpu/int8_gemm_opt1.cu`). Same as `mixed_sep`, except that the shared-memory tile of W in the INT8 kernel has one padding word per row (`WS_PAD = 1`). See section 3.

The tile size (8, 16 or 32) is the block edge of the two-dimensional kernels. Default 16.

## 2. Build flags

Source: `gpu/Makefile` and `results/gpu/env_golden.txt` (section "build flags").

```
nvcc -std=c++17 -O3 -arch=sm_89 --fmad=false --prec-div=true -Xcompiler "-O3 -ffp-contract=off" -lcublas
```

No fast-math flag is used. `--fmad=false`, `--prec-div=true` and `-ffp-contract=off` are what make CPU and GPU outputs bit-identical. Toolchain in the env files: `nvcc` 13.4, V13.4.92, `g++` 12.4.0 [`env_golden.txt` | sections `nvcc --version`, `g++ --version`].

## 3. Round 5: profile, decision and result

**Profile target.** `mixed_sep`, sq4096, t = 2, p = 5, tile 16, profiled with `nsys` and `ncu` [`profile/nsys_mixed_sep_sq4096_p5.txt`, `profile/ncu_int8_gemm_sq4096_p5.txt`]. The profile run used 1 warmup and 3 repetitions (profile only; `profile/profile_runs.csv`).

**Where the time goes** [`profile/nsys_mixed_sep_sq4096_p5.txt` | section `cuda_gpu_kern_sum`, 4 instances per kernel]: `int8_gemm_kernel<16>` 80.7% of kernel time (30.41 ms per launch on average), `outlier_kernel<16>` 12.3% (4.64 ms), `merge_kernel<16>` 2.1% (0.80 ms); the select, quantize, pack and gather kernels are each 0.8% or less.

**Why the INT8 kernel is slow** [`profile/ncu_int8_gemm_sq4096_p5.txt`]:
- L1/TEX throughput 97.13% and memory throughput 97.10%, compute (SM) throughput 29.50%, DRAM throughput 1.03%.
- Shared loads have a 9.0-way bank conflict: 1,535,115,264 bank conflicts, 66.65% of 2,303,260,125 shared-load wavefronts. `ncu` rates this finding at an estimated 64.74% speedup (an estimate by `ncu`, not a measurement).
- The top stall is the MIO queue: 44.5 of 82.2 cycles per issued instruction (54.1%).
- Occupancy is not the limit (99.78% achieved) and there is no register spilling.
- Global loads are not fully coalesced (22.2 of 32 bytes per sector used, estimated speedup 28.74 to 29.59%). Left alone.

**Cause, from reading the code** (`gpu/int8_gemm.cu` line 22, the read `ws[tx][w]`): with a row length of TILE words, the threads of a warp that differ in `tx` fall on only a few banks.

**Decision (approved by the human, 2026-11-21):** pad the `ws` tile to `[TILE][TILE + 1]` in a new variant `mixed_sep_opt1`; `mixed_sep` is untouched. One optimization only. `mixed_fused` was not profiled and not changed. Reason for `mixed_sep`: the INT8 kernel is 80.7% of its kernel time. The change keeps the integer arithmetic and its order, so output bits should not change.

**Bit-identity check** [`golden/checksums_gpu.txt` | lines starting `mixed_sep_opt1`]: 9 lines (tiles 8, 16, 32; p = 0, 2, 10), equal to the `mixed_sep` checksums for the same p: p=0 `25df7ddb655c5cb7`, p=2 `c611d2aaa379885a`, p=10 `5211561ff29d762e`. `rel_err` is also identical on all six group E configurations (string comparison of `E_opt.csv` and `B_sweep.csv`).

**Before and after, official sources** [before: `B_sweep.csv` | `variant==mixed_sep`, t=2, tile 16; after: `E_opt.csv`; medians in ms]:

| Shape | p | `t_int8_ms` before | after | Total before | after |
|---|---|---|---|---|---|
| sq2048 | 1 | 3.749 | 2.430 | 4.873 | 3.341 |
| sq2048 | 5 | 3.884 | 2.348 | 5.501 | 3.717 |
| sq2048 | 10 | 3.633 | 2.196 | 5.819 | 4.127 |
| sq4096 | 1 | 28.800 | 19.351 | 32.975 | 23.182 |
| sq4096 | 5 | 27.387 | 16.831 | 34.879 | 23.951 |
| sq4096 | 10 | 26.161 | 16.160 | 37.779 | 27.475 |

**Same-node control** [`E_base.csv` (`mixed_sep`) and `E_pair_opt.csv` (`mixed_sep_opt1`), one job, node `c23` per `env_E_base.txt`; the two variants alternate per configuration]: `t_int8_ms` improves 1.65x to 1.70x on all six configurations (for example sq4096 p=5: 27.382 to 16.556 ms); total time improves 1.37x to 1.53x. The control was added because group B ran on `c44` and group E on `c1`.

**Not explained:** the gain in the INT8 stage (1.5x to 1.7x) is smaller than the `ncu` estimate for the conflict fix (64.74%); the new kernel was not profiled, so what limits it now is unknown.

## 4. Anomalies

Each one is reported as found; none is explained unless stated.

1. **Three nodes for three groups.** Group A ran on `c41`, B on `c44`, C on `c23` and the golden run on `c37`; E ran on `c1` [`env_A_baselines.txt`, `env_B_sweep.txt`, `env_C_tiles.txt`, `env_golden.txt`, `env_E_opt.txt` | `SLURMD_NODENAME`]. The jobs were not pinned. All env files show an RTX 4060 Ti.
2. **Driver versions differ.** 615.71.09 on `c3` during Round 0 (`docs/env_gpu.md` fact 12) against 580.82.07 in all later env files (`docs/env_gpu.md` fact 21). Cause unknown.
3. **16 CPUs per job** (`SLURM_CPUS_ON_NODE=16`, `nproc` 16 [`env_golden.txt`]) although `-c 4` was requested for the interactive allocation.
4. **Timing at `p = 0` varies across t on sq2048, and on one skinny16 row** [`B_sweep.csv` | `total_ms_median`, p=0]: `mixed_sep` sq2048 4.462 / 4.179 / 4.461 ms for t = 0.5 / 2 / 5; `mixed_fused` sq2048 4.432 / 4.358 / 4.047 ms; `mixed_sep` skinny16 0.912 / 0.952 / 0.952 ms. The same work with different input values. At sq4096, sq1024 and skinny128 the three values agree within about 1%. The CPU lane reports a similar pattern at sq4096. Cause unknown. It makes "cost relative to p = 0" less reliable at small p on sq2048.
5. **sq2048 p = 1 base `t_int8_ms` differs by node:** 3.749 ms on `c44` [`B_sweep.csv`] against 4.014 ms on `c23` [`E_base.csv`] (about 7%). The other five group E configurations agree within 0.1% between the two nodes. Cause unknown.
6. **sq4096 p = 1 group E row is slower than its paired control:** 19.351 ms [`E_opt.csv`, node `c1`] against 17.446 ms [`E_pair_opt.csv`, node `c23`]. The `E_opt.csv` row has the largest relative stddev in the file (0.762 on a median total of 23.18 ms, about 3.3%). Hypothesis (not tested): run-to-run noise on that job.
7. **INT8 path slower than cuBLAS FP32.** At sq4096, t = 2: `fp32_cublas` 12,473.528 GFLOP/s, `int8_pertensor` 4,466.539 GFLOP/s [`A_baselines.csv`]. Observation, not a defect: the INT8 kernel is the simple Round 1 version. Not investigated beyond the profile in section 3.
8. **Tile 32 is the slowest.** `mixed_sep`, sq2048, t = 2, p = 5: 3,604.949 / 3,211.604 / 1,960.154 GFLOP/s for tiles 8 / 16 / 32 [`C_tiles.csv`]. Unexplained.
9. **`mixed_fused` is faster than `mixed_sep` at p = 5 in the tile sweep** (for example 4,221.341 against 3,604.949 GFLOP/s at sq2048, tile 8 [`C_tiles.csv`]). Observation only.
10. **`mixed_sep` p = 0 `rel_err` is below `int8_pertensor`** (sq2048, t = 2: 4.018e-02 against 6.808e-02 [`B_sweep.csv`, `A_baselines.csv`]). Plausible, since the mixed variants use per-row and per-column scales; not tested separately.

## 5. Validation summary (GPU lane)

Checks run on the three group CSVs and `E_opt.csv`, all on 2026-11-20 and 2026-11-21 [`A_baselines.csv`, `B_sweep.csv`, `C_tiles.csv`, `E_opt.csv`]:

- Headers equal SPEC 10; rows 20 (A), 210 (B), 12 (C), 6 (E); no duplicate configurations.
- S2: `fp32_cublas` largest `rel_err` 1.150e-06 (limit 1e-5) [`A_baselines.csv`].
- S3: 0 violations in 30 (variant, shape, t) groups. S4: `outlier_recall` is 1 on all 130 rows with p at least t. S5: 0 flags on GPU. S6: sep and fused `rel_err` identical on 105 pairs (B) and 6 pairs (C). S8, S9, S10: 0 flags.
- S7 (CPU against GPU `rel_err`): 225 keys, 0 mismatches, as reported by the CPU lane and re-checked here on `results/cpu/A_baselines.csv`, `results/cpu/B_sweep.csv`.
- Golden checksums: all GPU lines equal the CPU lines (`golden/checksums_gpu.txt`, `results/cpu/golden/checksums_cpu.txt`). Data checksums: 64 lines, identical in both lanes (`data_checksums.txt`).

## 6. Known limitations

- **Top-k is on the host.** The select stage copies the column maxima to the host, sorts there and copies the lists back, so its time includes two device-host copies and a host sort. Observation: it is small next to the INT8 kernel at sq4096 (0.610 ms of a 34.517 ms total, `mixed_sep` sq4096 p = 5 tile 16 [`C_tiles.csv`, `t_select_ms`, `total_ms_median`]); I did not check it at the smaller shapes.
- **Timing uses a device synchronization before starting and before stopping each stage** (SPEC 8). The stage times therefore include launch latency, and `total` is the sum of the stage times, not one run-through timing. `mixed_fused` has no separate outlier or merge stage (those columns are blank).
- **One seed, one outlier magnitude (x50), synthetic data.** Results say nothing about real data.
- **`mixed_fused` has the same shared-memory read pattern as the original INT8 kernel** (`gpu/fused.cu`, `wq_s[tx][w]`) and was not profiled or optimized. Reading the code suggests the same bank conflict; this is not verified.
- **The optimization covers `mixed_sep` only**, at tile 16 in group E. The tile sweep (group C) uses the original kernels.
- **Node variation.** Groups ran on different nodes (anomaly 1). Exclusive nodes were requested in every job script (`#SBATCH --exclusive`); the env files do not show whether another job shared the GPU.
- **`fp32_cublas` uses the default math mode.** The golden check excludes it (SPEC 13); its `rel_err` (S2) is the only correctness evidence.
- **Throughput cost relative to p = 0 at small p is affected by anomaly 4.**
