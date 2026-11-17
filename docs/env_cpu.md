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
| 7 | `perf` | not found in PATH on the login node | `which` |
| 8 | Python | 3.6.8 at /usr/bin/python3 | `python3 --version`, `which` |
| 9 | pandas | not installed (`ModuleNotFoundError`); matplotlib not tested because the import failed first | `python3 -c "import pandas, matplotlib ..."` |
| 10 | Module system | works; `gnu12/12.4.0`, `ohpc`, `autotools`, `cuda`, `hwloc/2.7.2`, `libfabric/1.19.0`, `ucx/1.15.0`, `prun/2.2`, `openmpi4/4.1.6` marked loaded on the login node | `module avail` |
| 11 | Modules listed that might matter for profiling | `likwid/5.2.2`, `papi/6.0.0`, `tau/2.31.1`, `valgrind/3.27.1` (listed only; not tested) | `module avail` |
| 12 | Partition `normal` | TIMELIMIT 1-00:00:00 (batch limit as listed by sinfo) | `sinfo` |
| 13 | Interactive `srun` on `normal` | limited to 6 hours | `slurm_job_submit` message from every srun |
| 14 | Slurm flags accepted on `normal` | `--partition`, `--exclusive`, `--time` (minutes), `--nodes=1` | `sbatch --help`; jobs 342481 to 342487 |
| 15 | Home filesystem | 10.1.1.10:/home, 91T size, 79T available | `df -h $HOME` |
| 16 | Home quota | 943M used; quota 39936M; limit 40960M; 2118 files | `quota -s` |
| 17 | Login node CPU | Intel Xeon Silver 4110, 2 sockets x 8 cores, 1 thread per core, nproc 16; avx2, avx512f, f16c, fma present; 93Gi RAM | `lscpu`, `nproc`, `free -h` on login2 |
| 18 | Compute node c3 CPU | AMD EPYC 9115 16-Core, 1 socket x 16 cores, 1 thread per core, 1 NUMA node, nproc 16, L1d 48K, L2 1024K, L3 32768K | `lscpu`, `nproc` in job 342487 |
| 19 | Compute node c1 | nproc 16, 125Gi RAM total | jobs 342481 (hostname), 342483 (nproc), 342484 (free) |
| 20 | CPU flags on a compute node | avx2, avx512f, f16c, fma present | job 342485 (node name not printed by that job) |
| 21 | Nodes differ | login2 is a Xeon with 93Gi; c3 is an EPYC; the jobs ran on at least two different nodes (c1, c3) | rows 17 to 19 |
| 22 | Exclusive allocation | accepted: `--exclusive` ran without error on `normal` | jobs 342481 to 342487 |

## Open questions (proposed defaults)

1. **`REPO_DIR` on ARC.** [No default; I will not guess it. Please give the path.]
2. **Where `data/` lives.** The generated files add up to about 0.59 GiB (computed from SPEC 4.3 sizes: sum over the 5 shapes and 3 t values of X and ref, plus W per (K, N), golden excluded). Home quota is 39936M. [`$REPO_DIR/data`, which SPEC 3 and `.gitignore` already assume.]
3. **Value of T.** Node c3 has 16 physical cores, 1 thread per core (row 18). [T = 16, valid only if every run happens on a node with 16 physical cores.]
4. **Node consistency.** The nodes differ (row 21), and job 342485 does not say which node had avx2 and f16c. Mixed node types would make GFLOP/s incomparable between rows. [Pin all CPU jobs to one node, c3, using a Slurm node-selection flag. Which flag does ARC accept (`--nodelist=c3`)? Not in any pasted output, so CHECK. The job scripts also print `hostname` and re-check avx2 and f16c on that node.]
5. **Node sharing.** `--exclusive` was accepted (row 22). [Use `--exclusive` for every job.]
6. **How jobs are submitted.** [`sbatch --partition=normal --exclusive --nodes=1 --time=<minutes>` per group script, with `OMP_PROC_BIND=close` and `OMP_PLACES=cores` set inside the script (LANE_CPU Technical notes).]
7. **Maximum walltime (SPEC 17.11).** The `normal` limit is 1 day (row 12); interactive jobs 6 hours (row 13). [Treat 1 day as the batch maximum; one job per group; if a group does not fit, split into one job per shape.]
8. **Python for Round 6.** pandas is missing (row 9), Python is 3.6.8. [Decide in Round 6: either you install pandas and matplotlib for your account (I will give a `CHECK` command then) or the analysis script runs on a machine where both exist. Nothing needed now.]
9. **Profiler for Round 5.** `perf` is not in PATH (row 7). [Fallback in LANE_CPU: stage columns plus compiler vectorization reports. `likwid` and `papi` are listed as modules (row 11) but untested; in Round 5 I will ask before relying on either.]
