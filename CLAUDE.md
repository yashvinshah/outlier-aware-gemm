# CLAUDE.md: Rules for Claude Code in this repository

This is a two-person graduate course project (CSC 548 Parallel Systems, NC State). Goal: measure how much GEMM throughput is lost to keep accuracy in outlier-aware mixed-precision INT8 GEMM (see SPEC.md section 1).

Two Claude Code sessions work in parallel on two machines:

- **Lane A (CPU)** owns `common/`, `cpu/`, `scripts/cpu/`, `results/cpu/`, `analysis/`, `NOTES.md`, `.gitignore`, `docs/env_cpu.md`, `docs/LANE_NOTES_cpu.md`.
- **Lane B (GPU)** owns `gpu/`, `scripts/gpu/`, `results/gpu/`, `docs/env_gpu.md`, `docs/LANE_NOTES_gpu.md`.

Before doing anything, read SPEC.md, COVERAGE.md and your lane file (LANE_CPU.md or LANE_GPU.md). If you are not sure which lane you are, ask.

## 1. Hard rules

1. **No git.** Do not run any git command (no pull, push, commit, add, checkout, stash, reset, status, log). The humans do all git. You only create and edit files in this working tree. Job scripts that the humans run on ARC may call `git rev-parse HEAD` (SPEC.md 17.13).
2. **Ask before running anything locally.** This covers compiling, running a program, running Python, installing packages, and any command that changes something. Before any local run, state the exact command and why, then wait for an explicit "yes". Reading files and listing directories is allowed. Editing files inside this repo is allowed. Never touch files outside this repo.
3. **Make no assumptions.** If something is not written in SPEC.md, COVERAGE.md or your lane file (a design decision, a parameter, a path, a flag, a library, a fact about the ARC cluster), stop and ask. Collect all open questions into one numbered list, give your proposed default next to each, and wait for answers. Do not continue past a question that has not been answered.
4. **Never invent facts.** Do not make up file paths, module names, compiler flags, library functions, command-line options, command output, or numbers. If you are not sure that a command, flag or API exists, say so, mark it `CHECK`, and say how to verify it. Never say a test passed, a build worked, or a result looks good unless the human pasted the output (or the file is in this tree and you read it). Every number you write must come from pasted output or from a file you read.
5. **Do exactly what is asked.** No extra features, options, abstractions, "future-proofing", or refactors. If you see something worth improving that was not requested, mention it in one line and leave it alone.
6. **SPEC.md, COVERAGE.md and this file are frozen.** Do not edit them. If you think something in them is wrong, ambiguous or impossible, tell the human (what, why, proposed fix) and wait. A spec change needs both lanes to agree, so the human will tell both sessions.
7. **Stay in your lane.** Do not edit files owned by the other lane. If you need something from the other lane, ask the human to relay it.

## 2. How we work together

The loop: you edit files here, the human commits and pushes, the human pulls on ARC and builds and runs using commands you provide, the human brings output back (pasted text, or files copied into `results/<lane>/` in this tree), and you validate it against COVERAGE.md and say what to do next.

Rules for command blocks that the human will run on ARC:

- Copy-paste ready. One command per line. Start with `REPO_DIR=<value>` (ask the human for the value; never guess it).
- Say where each block runs: login node, or inside an allocation.
- Use only scheduler syntax and module names that appear in pasted discovery output. Mark anything else `CHECK`.
- Say exactly what output to paste back.
- Redirect long output to a file and say which file to bring back.

### Git workflow (humans only)

- One shared `main` plus branches `lane-cpu` and `lane-gpu`.
- Changes to SPEC, COVERAGE, CLAUDE and the LANE files are made on `main` only, then merged into each lane branch.
- Each lane merges into `main` at milestones: golden passes, sweeps validated, Round 5 optimization.
- Before `git pull` on ARC, move any untracked `results/<lane>/` folder aside.
- Claude Code still runs no git.

## 3. Rounds

Work one round at a time. At the end of each round, stop and wait for the human.

- **Round 0a, discovery on the login node.** Output one block of read-only commands (the base list is in your lane file; you may add commands only with a stated reason and only if they are read-only). Stop.
- **Round 0b, discovery on a compute node.** Using the scheduler syntax from the 0a output, write the minimal command(s) that run the hardware queries on a compute node. Stop.
- **Round 0c, facts and questions.** Write `docs/env_<cpu|gpu>.md` containing a table of facts, each taken from pasted output only. List open questions with proposed defaults. Stop.
- **Round 1, code.** Write all code for your lane as described in SPEC.md and your lane file, in the order the lane file gives. Do not run anything. Give: the list of files with one-line purposes, the ARC commands (pull, load modules, build, one small first run per variant; `write_env.sh` and `run_golden.sh` come at the start of Round 2, SPEC.md 17.7), and what to paste back. Stop. If the human pastes compile or run errors, fix them with the smallest possible change, explain the cause in one or two sentences, and stop again.
- **Round 2, cross-check.** Run the golden procedure in SPEC.md section 13. Compare checksums between lanes. Do not move on until every pair matches exactly.
- **Round 3, sweeps.** Write the job scripts for the run groups in COVERAGE.md. The human submits them and brings back CSVs and env files. Stop.
- **Round 4, validation.** Validate the results against COVERAGE.md using its procedure. Output the verdict table and the exact commands for anything missing or flagged. Repeat until every item is PASS.
- **Round 5, profile and optimize.** Give profiling commands. After the output is pasted, identify the dominant cost from the data, propose exactly one optimization, and wait for approval. Then implement it as a new variant named `<variant>_opt1`, keeping the original untouched. Re-run the affected configurations (COVERAGE.md group E) and the golden check for `<variant>_opt1` with the same parameters as the base variant (SPEC.md section 13), and validate.
- **Round 6, final assets.** Described in COVERAGE.md section 6. Lane A writes the analysis script and NOTES.md. Both lanes write `docs/LANE_NOTES_<cpu|gpu>.md`.

Do not skip or merge rounds. Speed matters, but correctness comes first.

## 4. Code style

- C++17. Plain functions and structs. No inheritance, no classes unless a struct is not enough, no exceptions, no operator overloading.
- No templates, except the GPU `TILE` template parameter that SPEC.md section 7 requires.
- Allowed libraries: the C++ standard library, OpenMP, `<immintrin.h>` (CPU lane), the CUDA runtime and `cuda_fp16.h` (GPU lane), cuBLAS (only for `fp32_cublas`). Python is only for `analysis/`, only pandas and matplotlib, and only after confirming they are installed. Nothing else.
- One purpose per file. Keep files under about 300 lines and functions under about 40 lines.
- Comments explain why, not what. Where code implements a numbered step of SPEC.md section 6, put a comment like `// SPEC 6 step 2`.
- Constants fixed by the spec come from `common/params.h`. Other fixed constants (pi, 2^53, the integer clamp bound 127, the 64-byte alignment) are named `constexpr` values in the file that uses them (SPEC.md 17.1). No other magic numbers.
- Check every file read/write, every allocation, and every CUDA/cuBLAS call. On failure print one clear message and `exit(1)`. The only macro allowed is the CUDA error-check macro.
- No dead code, no commented-out code, no unused flags.
- Version 1 of every variant is the simplest correct implementation. Optimize only in Round 5.
- Never use `-ffast-math` or any flag that reorders floating-point operations.

## 5. How to answer

- Start each reply with the round and step you are in.
- Be brief. Put code in files, not in chat.
- End with: files changed, commands for the human, what to paste back, open questions.
- When validating results, name the file and the filter you used for every claim.
