# LANE_GPU.md: Lane B (GPU)

You are Lane B. You write the CUDA benchmark.

**You own:** `gpu/`, `scripts/gpu/`, `docs/env_gpu.md`, `docs/LANE_NOTES_gpu.md`.
**You must not edit:** `common/`, `cpu/`, `scripts/cpu/`, `analysis/`, `NOTES.md`, `.gitignore`, `docs/*_cpu.md`, `SPEC.md`, `COVERAGE.md`, `CLAUDE.md`.

Lane A writes `common/` (data generator, `compare_c`, `params.h`). Until Lane A's code is pushed and pulled, you can write your code but cannot test it. `common/params.h` has the exact content shown in SPEC 4.1; your code includes it from `../common/params.h`.

## Round 0a: base commands (login node, read-only)

Output exactly this block, adding commands only with a stated reason (read-only ones only). Commands that may not exist on ARC are marked CHECK.

```
uname -a
cat /etc/os-release
echo "$SHELL"
which gcc g++ nvcc python3 git ncu nsys nvprof sbatch srun salloc qsub
g++ --version
nvcc --version                                     # CHECK: may need a module load first
module avail 2>&1 | head -100                      # CHECK: the module command may not exist
sinfo                                              # CHECK: only if the scheduler is Slurm
df -h $HOME
quota -s                                           # CHECK
```

Ask the human to paste the whole output.

## Round 0b: GPU compute node

From the 0a output, write the minimal command(s) that, inside a GPU allocation on a compute node, run:

```
nvidia-smi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
nvcc --version
```

The `compute_cap` query field does not exist on older drivers. If the second command errors, tell the human to rely on the plain `nvidia-smi` output and ask what the GPU model's compute capability is rather than guessing it. Also ask how to get an exclusive GPU node on this scheduler (read the 0a output; if it is not answered there, ask). Stop.

## Round 0c: facts and questions

Write `docs/env_gpu.md`. Questions you must ask, each with a proposed default: the value of `REPO_DIR` on ARC; where `data/` should live given the quota; the exact `-arch=sm_<CC>` value; the scheduler and how GPU jobs are submitted; the CUDA module to load; whether the node is exclusive. If the GPU's compute capability is below 6.1, stop: `__dp4a` is not available (SPEC 12).

## Round 1: files, in this order

1. `gpu/util.h` and `gpu/util.cpp`: file reading, FNV-1a, statistics (median, min, max, population standard deviation), CSV writing, `rel_err`, `outlier_recall`, the CUDA error-check macro.
2. `gpu/select.cu`: Step 1: a column-max kernel, copy of `colmax` to the host, `std::stable_sort` on the host (colmax descending, index ascending on ties), building S and Nrm, copy of the index lists back to the device.
3. `gpu/quant.cu`: Step 2: scale kernels, quantize and gather kernels, FP16 gather kernels, including the packing the `__dp4a` kernel needs.
4. `gpu/int8_gemm.cu`: Step 3, the `__dp4a` kernel, templated on TILE in {8, 16, 32}, output float `Ci8`.
5. `gpu/outlier_merge.cu`: Step 4 kernel (writes `Co`) and Step 5 kernel (writes C), same TILE.
6. `gpu/fused.cu`: one kernel for Steps 3, 4, 5.
7. `gpu/pertensor.cu`: `int8_pertensor`.
8. `gpu/fp32_cublas.cu`: `fp32_cublas`.
9. `gpu/bench_gpu.cu`: command line (SPEC 11), loading the data files and copying them to the device before timing, the warmup and timing loop (SPEC 8), stage timers, the CSV row, `--write-c`.
10. `gpu/Makefile` with the SPEC 12 flags.

Do not write job scripts yet. Give the build and golden-test commands for ARC. Ask the human to paste the build output and the output of one small run per variant on the golden shape.

## Technical notes (all of these are facts; follow them)

- `__dp4a(int a, int b, int c)` treats `a` and `b` as four packed signed 8-bit values each and returns `c` plus their 4-element dot product. It needs compute capability 6.1 or higher. Use the signed (int) overload.
- `__dp4a` needs 4 consecutive channels packed into one 32-bit word for both operands. So X rows are stored with channels contiguous (4 per word), and W is stored transposed (N rows of channels, 4 per word). Pad K - k up to a multiple of 4 with zeros in both operands; zeros do not change the sum. Packing and transposing belong to the quantize stage and are timed there.
- FP16: include `cuda_fp16.h`; use `__float2half_rn` and `__half2float`. The outlier path converts back to float and multiplies and accumulates in float32 in ascending channel order (SPEC 6 step 4).
- cuBLAS is column-major. To compute row-major C = X * W, call SGEMM on the transposed problem with the operands swapped. Call `cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH)`. Expect `rel_err` of `fp32_cublas` around 1e-7 to 1e-6; a value near 1e-3 means TF32 is active, which is a bug (COVERAGE S2).
- Build with `--fmad=false`. Division and `roundf` must be the standard (IEEE) ones; do not use fast-math intrinsics.
- Timing: host wall clock with `cudaDeviceSynchronize()` before starting and before stopping each stage timer (SPEC 8). No cudaEvent timing.
- All device buffers are allocated before timing. Inputs are already on the device before timing.
- Kernel design: one thread per output element, blocks of TILE x TILE threads, shared-memory tiles over the packed channel dimension. No register blocking, no warp tricks, no tensor cores, no cuBLASLt, no CUTLASS. Simplicity is the goal in version 1.
- The fused kernel does the INT8 loop first, then the outlier loop in ascending channel order, then writes `((float)acc * sx[i]) * sw[j]` plus the outlier sum, so the bits equal the separate version.
- Skinny shapes (M = 16 or 128): compute the grid from M, N and TILE, and make sure partial tiles at the edges are handled (N is a multiple of 8 for all shapes; M is not a multiple of 32 for skinny16).

## Round 2

Run the golden procedure of SPEC 13 for the GPU lane and write `scripts/gpu/run_golden.sh`, which writes `results/gpu/golden/checksums_gpu.txt`. The human exchanges the checksum files with Lane A (via the repo or by pasting) and both sessions compare.

## Round 3: job scripts

Groups A, B, C of COVERAGE section 2 (GPU lane), one script each: `scripts/gpu/run_A_baselines.sh`, `run_B_sweep.sh`, `run_C_tiles.sh`. Each script: writes the env file first (SPEC 14), loops over configurations, and appends to `results/gpu/<group>.csv`. Add the submission wrapper that matches the scheduler found in Round 0. Start with a tiny smoke job (golden shape, a few configurations) before the full groups.

## Round 5 (GPU specifics)

Use `ncu` or `nsys` only if Round 0 found them (`nvprof` only if the GPU generation supports it; ask). If no profiler exists, the stage columns are the evidence; say so and ask the human before relying on that. Save profiler output under `results/gpu/profile/`.

## Round 6

You write `docs/LANE_NOTES_gpu.md` (contents in COVERAGE section 6). Lane A produces the analysis and NOTES.md from both lanes' CSVs.

## Done when

Every item in COVERAGE section 1 is PASS for your lane, and `docs/LANE_NOTES_gpu.md` exists.
