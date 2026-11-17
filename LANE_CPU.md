# LANE_CPU.md: Lane A (CPU)

You are Lane A. You write the shared tools in `common/` first, then the CPU benchmark.

**You own:** `.gitignore`, `common/`, `cpu/`, `scripts/cpu/`, `results/cpu/`, `analysis/`, `NOTES.md`, `docs/env_cpu.md`, `docs/LANE_NOTES_cpu.md`.
**You must not edit:** `gpu/`, `scripts/gpu/`, `results/gpu/`, `docs/*_gpu.md`, `SPEC.md`, `COVERAGE.md`, `CLAUDE.md`.

Lane B cannot test anything until your generator exists. In Round 1, write and hand over `common/` first, then tell the human to push it before you continue.

## Round 0a: base commands (login node, read-only)

Output exactly this block, adding commands only with a stated reason (read-only ones only). Commands that may not exist on ARC are marked CHECK.

```
uname -a
cat /etc/os-release
echo "$SHELL"
which gcc g++ python3 git perf sbatch srun salloc qsub
gcc --version
g++ --version
python3 --version
python3 -c "import pandas, matplotlib; print(pandas.__version__, matplotlib.__version__)"
module avail 2>&1 | head -100                      # CHECK: the module command may not exist
sinfo                                              # CHECK: only if the scheduler is Slurm
df -h $HOME
quota -s                                           # CHECK
nproc
lscpu | grep -E "Model name|Socket|Core|Thread|MHz|L1d|L2|L3|NUMA"
lscpu | grep -o -w -E "avx2|f16c|fma|avx512f" | sort -u
free -h
```

Ask the human to paste the whole output.

## Round 0b: compute node

From the 0a output, write the minimal command(s) that run `lscpu`, `nproc`, `free -h`, the `avx2|f16c` flag check, and `g++ --version` inside a CPU allocation on a compute node. Also ask how to get an exclusive node on this scheduler (read the 0a output; if it is not answered there, ask the human). Stop.

## Round 0c: facts and questions

Write `docs/env_cpu.md`. Questions you must ask at this point, each with a proposed default: the value of `REPO_DIR` on ARC; where `data/` should live given the quota; the value of T (physical cores of the node, not hyperthreads); the scheduler and the way jobs are submitted; whether the node is exclusive; the scheduler's maximum walltime (SPEC 17.11).

## Round 1: files, in this order

1. `.gitignore`: ignore `data/`, object files and the `bench_cpu`, `gen_data`, `compare_c` binaries. Do not ignore `results/`.
2. `common/params.h`: exactly the content in SPEC 4.1. Also `common/checksum.h`: header-only inline FNV-1a, included by `gen_data`, `compare_c`, `cpu/util` and `gpu/util` (SPEC 17.1).
3. `common/fp32_gemm.h` and `common/fp32_gemm.cpp`: `fp32_gemm_serial` and `fp32_gemm_omp` (SPEC 7). Loop order i, k, j. OpenMP only over i. Compiled with `-ffp-contract=off`.
4. `common/gen_data.cpp`: SPEC 4.2 to 4.4. Takes `--M --K --N --true-pct --data-dir`.
5. `common/compare_c.cpp`: `compare_c a.f32 b.f32 <count>` prints the number of elements whose bit patterns differ, the maximum absolute difference, and the FNV-1a checksum of each file.
6. `common/Makefile`: builds `gen_data` and `compare_c` with the SPEC 12 flags.
7. `scripts/cpu/gen_all_data.sh <DATA_DIR>`: loops over every shape and every t of SPEC 5 and prints the CHECKSUM lines (the humans save them with `tee`).

**Stop here.** Tell the human to push `common/` so Lane B can start testing, and give the commands to build and run the generator on ARC. Wait for the human to say continue.

8. `cpu/util.h` and `cpu/util.cpp`: file reading, statistics (median, min, max, population standard deviation), CSV writing, `rel_err` (SPEC 17.3), `outlier_recall`. FNV-1a comes from `common/checksum.h` (SPEC 17.1).
9. `cpu/select.cpp`: Step 1.
10. `cpu/quant.cpp`: Step 2 for the normal channels, the outlier channels, and the per-tensor case. Gathering and packing in whatever layout your AVX2 loop needs.
11. `cpu/pertensor.cpp`: `int8_pertensor`.
12. `cpu/int8_scalar.cpp`: Steps 3, 4, 5 as plain loops. This file gets the special build flags in SPEC 12.
13. `cpu/int8_avx2.cpp`: Steps 3 and 4 with intrinsics, Step 5 as a plain loop.
14. `cpu/fused_avx2.cpp`: Steps 3, 4, 5 in one loop nest.
15. `cpu/bench_cpu.cpp`: command line (SPEC 11), loading the data files, running the variant with the warmup and timing loop (SPEC 8), stage timers, the CSV row, `--write-c`.
16. `cpu/Makefile`.
17. `scripts/cpu/write_env.sh` and `scripts/cpu/run_golden.sh` are written at the start of Round 2 (SPEC 17.7), and the group scripts of Round 3 in Round 3, not now.

Give the build commands and plain per-variant commands for one small run on the golden shape (SPEC 17.7). Ask the human to paste the build output and the output of one small run per variant on the golden shape.

## Technical notes (all of these are facts; follow them)

- For signed INT8 x signed INT8, do **not** use `_mm256_maddubs_epi16`. It multiplies unsigned bytes by signed bytes and saturates. Sign-extend the bytes to 16 bits and use `_mm256_madd_epi16`, which multiplies signed 16-bit pairs and adds adjacent products into 32-bit lanes.
- `_mm256_madd_epi16` consumes pairs of adjacent channels. The number of normal channels K - k can be odd. Handle the tail correctly (zero padding does not change the sum). Packing W into channel pairs belongs to the quantize stage and is timed there.
- FP16 conversion: with `-mf16c`, `<immintrin.h>` provides `_cvtss_sh`, `_cvtsh_ss`, `_mm256_cvtps_ph`, `_mm256_cvtph_ps`. Use round-to-nearest-even. AVX2 has no FP16 arithmetic, so the outlier path converts FP16 values back to float and multiplies and adds in float32 (SPEC 6 step 4).
- Step 4 order must stay ascending over channels, per output element. Vectorize across output columns j, never across channels.
- Do not use OpenMP reductions on floating-point values. A thread owns whole rows or (row, column-block) tiles. For skinny shapes (M = 16 or 128) tiles over (i, j-block) keep more threads busy.
- Set the thread count with `omp_set_num_threads(--threads)`. The job scripts set `OMP_PROC_BIND=close` and `OMP_PLACES=cores`.
- Allocate aligned buffers (64 bytes) before timing. Allocation, zero-fills done once at allocation or setup, and any one-time setup are outside every timed region; anything done in every repetition to produce the packed, gathered or transposed arrays is timed inside `t_quant_ms` (SPEC 8).
- Never rely on compiler auto-vectorization in `int8_scalar.cpp`; its flags disable it on purpose.

## Round 3: job scripts

Groups A, B, C, D of COVERAGE section 2 (CPU lane), one script each: `scripts/cpu/run_A_baselines.sh`, `run_B_sweep.sh`, `run_C_ladder.sh`, `run_D_scaling.sh`. Each script: writes the env file first (SPEC 14), loops over configurations, and appends to `results/cpu/<group>.csv` (file names: COVERAGE section 2). Add the submission wrapper that matches the scheduler found in Round 0. Start with a tiny smoke job (golden shape, a few configurations) before the full groups.

## Round 5 (CPU specifics)

Profiling tools only if Round 0 found them (`perf` is the likely candidate). If no profiler exists, the stage columns plus compiler vectorization reports are the evidence; say so and ask the human before relying on that. The group E script is `scripts/cpu/run_E_opt.sh`. Re-run the golden check for `<variant>_opt1` and with the base variant's parameters (p in {0, 2, 10}, threads in {1, T}) and append its lines to `checksums_cpu.txt` (SPEC 13 step 6).

## Round 6

You write `analysis/make_assets.py`, run it (with the human's permission), write `NOTES.md` (rules in COVERAGE section 6), and write `docs/LANE_NOTES_cpu.md`.

## Done when

Every item in COVERAGE section 1 is PASS for your lane, and `NOTES.md` and `docs/LANE_NOTES_cpu.md` exist.
