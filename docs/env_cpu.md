# env_cpu.md: ARC environment facts for Lane A (CPU)

Every row comes from output the human pasted. Source column names the command and, where known, the Slurm job. Nothing here is inferred except where the Notes column says "computed".

## Facts

| # | Fact | Value | Source |
|---|---|---|---|
| 1 | Login node kernel | Linux 4.18.0-553.184.1.el8_10.x86_64 | `uname -a` on login2 |
| 2 | OS release line printed | `PLATFORM_ID="platform:el8"` (only this line was pasted) | `cat /etc/os-release` |
| 3 | Shell | /bin/bash | `echo "$SHELL"` |
| 4 | gcc, g++ | 12.4.0, path /opt/ohpc/pub/compiler/gcc/12.4.0/bin | `which`, `gcc --version`, `g++ --version` (login); `g++ --version` in job 342486 |
| 5 | git | /usr/bin/git | `which` |
| 6 | Scheduler | Slurm: `sbatch`, `srun`, `salloc` exist; `qsub` does not | `which` |
| 7 | `perf` | not found in PATH on the login node; `/usr/bin/perf`, `/usr/bin/gprof` and `/usr/bin/valgrind` exist on c3 (row 27) | `which` on login2; batch job 342493 |
| 8 | Python | 3.6.8 at /usr/bin/python3 | `python3 --version`, `which` |
| 9 | pandas | not installed on ARC (`ModuleNotFoundError`); Round 6 analysis runs on the human's own machine, not on ARC | `python3 -c "import pandas, matplotlib ..."`; human answer 8 |
| 10 | Module system | works on the login node; `gnu12/12.4.0`, `ohpc`, `autotools`, `cuda`, `hwloc/2.7.2`, `libfabric/1.19.0`, `ucx/1.15.0`, `prun/2.2`, `openmpi4/4.1.6` marked loaded | `module avail` |
| 11 | Profiling-related modules listed | `likwid/5.2.2` and `papi/6.0.0` are listed by `module avail` on the login node (listed only; not loaded or tested) | `module avail 2>&1 \| grep -i -E "likwid\|papi"` on login2 |
| 12 | Partition `normal` | TIMELIMIT 1-00:00:00 (batch limit as listed by sinfo) | `sinfo` |
| 13 | Interactive `srun` on `normal` | limited to 6 hours | `slurm_job_submit` message from every srun |
| 14 | Slurm flags accepted on `normal` | `--partition`, `--exclusive`, `--time` (minutes), `--nodes=1` | `sbatch --help`; jobs 342481 to 342487 |
| 15 | Node-selection flags that exist | `-w, --nodelist=hosts...`, `-C, --constraint=list`, `-x, --exclude=hosts...` | `sbatch --help \| grep -i -E "nodelist\|constraint\|exclude"` |
| 16 | Home filesystem | 10.1.1.10:/home, 91T size, 79T available | `df -h $HOME` |
| 17 | Home quota | 943M used; quota 39936M; limit 40960M; 2118 files | `quota -s` |
| 18 | Login node CPU | Intel Xeon Silver 4110, 2 sockets x 8 cores, 1 thread per core, nproc 16; avx2, avx512f, f16c, fma present; 93Gi RAM | `lscpu`, `nproc`, `free -h` on login2 |
| 19 | Compute node c3 CPU | AMD EPYC 9115 16-Core, 1 socket x 16 cores, 1 thread per core, 1 NUMA node, nproc 16, L1d 48K, L2 1024K, L3 32768K | `lscpu`, `nproc` in job 342487 |
| 20 | Compute node c1 | nproc 16, 125Gi RAM total | jobs 342481 (hostname), 342483 (nproc), 342484 (free) |
| 21 | CPU flags on a compute node | avx2, avx512f, f16c, fma present (node name not printed by that job); confirmed on c3 in row 27 | job 342485 |
| 22 | Nodes differ | login2 is a Xeon with 93Gi; c3 is an EPYC; the jobs ran on at least two different nodes (c1, c3) | rows 18 to 20 |
| 23 | Exclusive allocation | accepted: `--exclusive` ran without error on `normal` | jobs 342481 to 342487 |
| 24 | c3 in sinfo | c3 is listed in partitions `normal*`, `max`, `all`, `class`, `turin`, `rtx4060ti16g`; STATE `allocated`; CPUS 16 | `sinfo -N -n c3 -o "%N %P %T %c"` on login2 |
| 25 | Job 342492 (pinned to c3 with `--nodelist=c3`) | printed "queued and waiting for resources", then `Killed`; no hostname, nproc or lscpu output | `srun --partition=normal --nodelist=c3 --exclusive --time=1 --nodes=1 bash -c '...'`. Cause unknown. |
| 26 | Job 342482 | allocated, then `Killed` (lscpu on a compute node, no output). `squeue` later showed it still running on c1 after 22:22 despite `--time=5`; 342492 was still pending. Both were cancelled by the human. The `Killed` message came from the `srun` client; the allocation outlived it. Cause unknown. | `srun ... lscpu`, `squeue -u apethan` |
| 27 | Batch probe pinned to c3 (job 342493: `sbatch --partition=normal --nodelist=c3 --exclusive --time=1 --nodes=1 --wrap=...`) | hostname c3; nproc 16; 1 thread per core, 16 cores per socket, 1 socket, 1 NUMA node (CPUs 0-15); AMD EPYC 9115 16-Core; flags avx2, avx512f, f16c, fma present; `/usr/bin/perf`, `/usr/bin/gprof`, `/usr/bin/valgrind` exist. The job waited in the queue (reason `ReqNodeNotAvail`, then `Resources`) before running. | `cat $HOME/c3_probe.out` |
| 28 | `--wrap`, `--output` and `--nodelist` | accepted by `sbatch` (job 342493 ran and wrote the output file) | job 342493 |

## Answers from the human (Round 0c)

| # | Answer |
|---|---|
| 1 | `REPO_DIR=$HOME/outlier-aware-gemm` |
| 2 | `data/` at `$REPO_DIR/data` |
| 3 | T = 16, valid only on c3 |
| 4 | Pin all CPU jobs to c3; wait limit before reconsidering is 30 minutes. Probe finished (rows 15, 24, 27). |
| 5 | `--exclusive` on every job |
| 6 | Submit each group with `sbatch` and `--partition=normal --exclusive --nodes=1`; set `OMP_PROC_BIND=close` and `OMP_PLACES=cores` inside the script; record CPU affinity (`nproc`) in each env file |
| 7 | Split a group per shape only if it exceeds its estimate or the limit. Request each job's walltime from the smoke-run extrapolation (about 2x margin), not the 1-day maximum. One job per group. |
| 8 | Round 6 analysis runs on the human's own machine; all ARC scripts are bash-only; pandas and matplotlib are confirmed locally in Round 6 |
| 9 | Deferred; the addendum probe covers which profiler tools exist (row 27: `perf`, `gprof`, `valgrind` exist on c3) |

## Still open

1. c3 is shared with other users' jobs (row 24), so every `--exclusive` job pinned to c3 may wait in the queue before it starts. I cannot see how long.
2. Use batch jobs (`sbatch`), not interactive `srun`, for anything that may queue: a killed `srun` client left allocations running (row 26).
3. Whether `perf` works for this account on c3 (counter permissions) is not tested; it is only known to exist.
