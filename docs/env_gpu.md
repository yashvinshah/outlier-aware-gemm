# env_gpu.md: ARC facts for the GPU lane (Round 0c)

Every fact below comes from output the human pasted in Round 0a or Round 0b. Source tags: **0a** = login node block, **0b** = compute node. Nothing here is assumed.

## Facts

| # | Fact | Value | Source |
|---|---|---|---|
| 1 | Login node | `login2`, kernel `4.18.0-553.184.1.el8_10.x86_64`, shell `/bin/bash` | 0a: `uname -a`, `echo $SHELL` |
| 2 | OS release | only `PLATFORM_ID="platform:el8"` was printed | 0a: `cat /etc/os-release` |
| 3 | Host compiler | `g++ (GCC) 12.4.0` at `/opt/ohpc/pub/compiler/gcc/12.4.0/bin/g++` (`gcc` in the same directory) | 0a: `which`, `g++ --version` |
| 4 | Other tools | `python3` at `/usr/bin/python3`, `git` at `/usr/bin/git` | 0a: `which` |
| 5 | Scheduler | Slurm: `sbatch`, `srun`, `salloc` in `/usr/bin`; `qsub` absent | 0a: `which`, `sinfo` |
| 6 | Module system | exists; module `cuda` is loaded by default and sets `CUDA_HOME` and `CUDA_PATH` to `/usr/local/cuda` | 0a: `module avail`; 0b: `module show cuda` |
| 7 | nvcc on the login node | not found; `/usr/local/cuda/bin` does not exist there | 0a: `nvcc --version`; 0b: `ls` |
| 8 | GPU partition used | `rtx4060ti16g`: GRES `(null)`, `MaxTime=1-00:00:00`, `DefaultTime=NONE`, `OverSubscribe=NO`, `ExclusiveUser=NO`, nodes `c[0-3,23-24,37,39-44,54-56,60-79]` | 0b: `sinfo`, `scontrol show partition` |
| 9 | GPU request syntax | `--gres=gpu:1` is rejected on this partition ("Invalid generic resource (gres) specification") | 0b: `srun`, `salloc` errors |
| 10 | Interactive job rules | interactive jobs on this partition are limited to 6 hours; `--mem=16G` was rejected ("Memory specification can not be satisfied") | 0b: `salloc` errors |
| 11 | Allocation that worked | `salloc -p rtx4060ti16g -N1 -c 4 -t 06:00:00`, job 342489, node `c3` | 0b |
| 12 | GPU | `NVIDIA GeForce RTX 4060 Ti`, one GPU listed (index 0), `memory.total` 16380 MiB, driver 615.71.09, no processes running at the time of the query | 0b: `nvidia-smi` on `c3` |
| 13 | Compute capability | `8.9` (from the `compute_cap` query field) | 0b |
| 14 | CUDA toolkit | `nvcc` release 13.4, V13.4.92, build `cuda_13.4.r13.4/compiler.38855100_0`; `/usr/local/cuda/bin` on `c3` contains `nvcc`, `ncu`, `nsys`, `compute-sanitizer`, `cuobjdump`; no `nvprof` in that listing | 0b: `nvcc --version`, `ls` on `c3` |
| 15 | nvcc flags exist | `--prec-div {true|false}` and `--fmad {true|false}` are listed by `nvcc --help` | 0b |
| 16 | Profilers on the login PATH | `ncu`, `nsys`, `nvprof` not found | 0a: `which` |
| 17 | Disk and quota | `/home`: 91T size, 79T available; user quota 39936M soft, 40960M limit, 12220M used | 0a: `df -h $HOME`, `quota -s` |
| 18 | Date on the node | Tue Nov 17 13:09:43 2026 | 0b: `nvidia-smi` header |

Not captured, so unknown: whether `nvcc` is on PATH in a non-interactive `sbatch` shell, which `g++` the compute node uses, whether `-lcublas` links, whether `ncu` can read hardware counters, whether other jobs can share the GPU on a node.

## Derived from the facts (not extra assumptions)

- Compute capability 8.9 is at least 6.1, so the `__dp4a` requirement of SPEC 12 is met.
- The `-arch` value implied by SPEC 12 and fact 13 is `sm_89`. It is proposed below for confirmation.

## Open questions (each with a proposed default)

1. **`REPO_DIR` on ARC.** No default: please give the full path of the cloned repo.
2. **`-arch=sm_<CC>`.** Proposed: `sm_89`, from fact 13. The first build on ARC (run by you) confirms that nvcc 13.4 accepts it, together with `-std=c++17` and `-lcublas` (SPEC 12).
3. **`data/` location.** Proposed: `$REPO_DIR/data` (git-ignored). My estimate from the SPEC 4.3 and 5 file sizes is under 1 GiB in total, against about 27.7 GiB of quota headroom (39936M minus 12220M, fact 17).
4. **Scheduler and GPU job submission.** Slurm. Interactive: the `salloc` line in fact 11. Proposed for job scripts: `sbatch -p rtx4060ti16g -N1 -c 4 -t <time>` with no `--gres` and no `--mem` (facts 9 and 10). `CHECK`: whether batch jobs have a time limit below the partition `MaxTime` of 1 day, as interactive jobs do.
5. **Maximum walltime (SPEC 17.11).** Partition `MaxTime` is 1-00:00:00 (fact 8). Interactive jobs are limited to 6 hours (fact 10). Please confirm the batch limit.
6. **CUDA module.** No explicit `module load` was needed on `c3` for `nvcc --version` to run, and `cuda` is loaded by default (fact 6). Proposed: no `module load` in job scripts, and the env file records `module list` (SPEC 14). `CHECK`: that `nvcc` is on PATH inside an `sbatch` job.
7. **Exclusive node.** Not established. The partition has no GRES (fact 9), so Slurm may not stop another job from using the same GPU. The partition shows `OverSubscribe=NO` and `ExclusiveUser=NO`, which does not settle whole-node exclusivity. Proposed: add `--exclusive` (standard Slurm option, `CHECK`: not in any pasted output and may not be allowed here), and have the env file record the `nvidia-smi` process list. Please confirm whether `--exclusive` is allowed.
8. **Same GPU model for every run.** The partition lists 36 nodes, and only `c3` was queried. Proposed: do not pin a node; the env file records `nvidia-smi` for each job, and Round 4 flags any differences.
9. **Profilers.** `ncu` and `nsys` exist in `/usr/local/cuda/bin` on `c3` (fact 14); `nvprof` does not. Proposed: use `ncu` and `nsys` in Round 5. `CHECK`: whether `ncu` can access hardware counters for your account.

## Decisions (human answers, 2026-11-17)

Answers to the questions above. They are the human's word, not pasted output.

- Q2: `-arch=sm_89` accepted as proposed; the first ARC build still confirms it.
- Q3: `data/` at `$REPO_DIR/data`, as proposed.
- Q4, Q5: job scripts use `sbatch` with no `--gres` and no `--mem`, as proposed. The human has not seen batch time limits cause problems and does not want every scenario analysed. The `-t` value is set in Round 3.
- Q6: no `module load` in job scripts; the env file records `module list`. Whether `nvcc` is on PATH inside an `sbatch` job is confirmed by the first batch job.
- Q7: `--exclusive` is allowed and is used for timing runs.
- Q8: no node pinning.
- Q9: no answer needed now. Profiler use is decided in Round 5.
- Q1 (`REPO_DIR`): `/home/ynshah/outlier-aware-gemm` (from `pwd` after the clone on the login node, 2026-11-17). Remote branches seen by `git branch -a`: `main`, `cpu`, `yashvi_gpu`.

## Facts added in Rounds 2 and 3 (from `results/gpu/env_*.txt`)

Each row is read from the env file named in the Source column. These add to the table above; none replaces it.

| # | Fact | Value | Source |
|---|---|---|---|
| 19 | Nodes used | golden run: `c37` (job 343498); group A: `c41` (job 344194); group B: `c44` (job 344195); group C: `c23` (job 344196). The group jobs were not pinned, so they ran on three different nodes | `env_golden.txt`, `env_A_baselines.txt`, `env_B_sweep.txt`, `env_C_tiles.txt` |
| 20 | GPU on those nodes | `NVIDIA GeForce RTX 4060 Ti` in all four env files | the `nvidia-smi` section of each |
| 21 | Driver on those nodes | 580.82.07, CUDA 13.0 shown by `nvidia-smi` in all four env files. Fact 12 recorded 615.71.09 on `c3`; the two values differ and the difference is not explained | the `nvidia-smi` section of each |
| 22 | CPUs per job | `SLURM_CPUS_ON_NODE=16` and `nproc` 16 (golden env file) although `-c 4` was requested for the interactive allocation | `env_golden.txt`; the group jobs show `SLURM_CPUS_ON_NODE=16` |
| 23 | Toolchain inside Slurm jobs | `g++ (GCC) 12.4.0` and `nvcc` 13.4, V13.4.92 are found inside the batch jobs without a `module load` (all four env files). This settles open question 6 | the `g++ --version` and `nvcc --version` sections |
| 24 | Batch time limit | `sbatch` accepted `-t 1-00:00:00` in the three group scripts and the jobs ran. This settles the time-format `CHECK` | job ids 344194 to 344196 and the logs in `results/gpu/logs/` (logs are not in the tree) |
| 25 | Scheduler variables | `SLURM_JOB_ID`, `SLURM_JOB_PARTITION`, `SLURMD_NODENAME`, `SLURM_JOB_NODELIST`, `SLURM_JOB_NUM_NODES` and `SLURM_CPUS_ON_NODE` all resolved; `module list` worked inside the script | `env_golden.txt` |
| 26 | Loaded modules | `autotools`, `prun/2.2`, `gnu12/12.4.0`, `hwloc/2.7.2`, `ucx/1.15.0`, `libfabric/1.19.0`, `openmpi4/4.1.6`, `cuda`, `ohpc` | `env_golden.txt` |
