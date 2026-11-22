"""analysis/headline.py: headline_numbers.csv and the facts sheet for the talk.

Every number carries its source files and filter, its n (processes) and, for the CPU, its min-max range.
Throughput costs use the median over the processes of each row, including the p = 0 reference, and are
labelled by the rule in data.cost.
"""
import glob
import os
import re

import pandas as pd

from data import NOISE_NOTE, ROOT, baseline, cost, s5_counts, sel, sweep

T = 2.0
SHAPE = "sq4096"


def short(row):
    """Short file tag: the original CSV plus the replicate numbers that were merged into the row."""
    files = row["sources"].split(" + ")
    base = [os.path.relpath(f, "results") for f in files if "/rep/" not in f]
    reps = [m.group(1) for f in files for m in [re.search(r"_rep(\d+)\.csv", f)] if m]
    return base[0] + (f"+rep{','.join(reps)}" if reps else "")


def rng(row, col, digits=1):
    """'median [min-max] n=3' (or 'median n=1' when there is one process)."""
    if row["n"] > 1:
        return f"{row[col]:.{digits}f} [{row[col + '_min']:.{digits}f}-{row[col + '_max']:.{digits}f}] n={int(row['n'])}"
    return f"{row[col]:.{digits}f} n=1"


def flt(row, **kw):
    return ", ".join([row["variant"], row["shape"]] + [f"{k}={v}" for k, v in kw.items()])


class Sheet:
    """Collects the CSV rows and the text lines of the facts sheet."""

    def __init__(self):
        self.rows, self.lines = [], []

    def line(self, text=""):
        self.lines.append(text)

    def add(self, section, name, value, unit, row=None, col=None, filt=""):
        entry = {"section": section, "name": name, "value": value, "unit": unit, "n": "", "min": "", "max": "",
                 "source": "", "filter": filt}
        if row is not None:
            entry["source"] = row["sources"]
            entry["n"] = int(row["n"])
            if col and row["n"] > 1:
                entry["min"], entry["max"] = row[col + "_min"], row[col + "_max"]
        self.rows.append(entry)


def cpu_b(agg, variant, p):
    return sel(agg, lane="cpu", group="B", variant=variant, shape=SHAPE, true_out_pct=T, iso_out_pct=float(p),
               threads=16.0, tile=None)


def section_throughput(agg, sh):
    sh.line("(1) CPU, sq4096, t = 2, 16 threads: GFLOP/s, median [min-max] n; cost vs p = 0 (median of p vs median of p = 0)")
    for variant in ("mixed_sep_avx2", "mixed_fused_avx2"):
        ref = cpu_b(agg, variant, 0)
        sh.line(f"  {variant}  p=0: {rng(ref, 'gflops')}  [{short(ref)} | {flt(ref, t=2, p=0)}]")
        sh.add(1, f"cpu {variant} p=0 gflops", ref["gflops"], "GFLOP/s", ref, "gflops", flt(ref, t=2, p=0))
        for p in (1, 5, 10):
            row = cpu_b(agg, variant, p)
            value, low, high, verdict = cost(row, ref)
            sh.line(f"    p={p}: {rng(row, 'gflops')}; cost {value:.1f}% (allowed {low:.1f} to {high:.1f}%): {verdict}")
            sh.add(1, f"cpu {variant} p={p} gflops", row["gflops"], "GFLOP/s", row, "gflops", flt(row, t=2, p=p))
            sh.add(1, f"cpu {variant} p={p} cost vs p=0", value, "%", row, None,
                   f"{flt(row, t=2, p=p)}; allowed {low:.1f} to {high:.1f}%; {verdict}")


def section_rel_err(agg, sh):
    sh.line(f"(2) rel_err, sq4096, t = 2 (identical in all processes), factor = int8_pertensor / mixed_sep")
    pts = {lane: baseline(agg, lane, "int8_pertensor", SHAPE) for lane in ("cpu", "gpu")}
    for lane in ("cpu", "gpu"):
        sh.add(2, f"{lane} int8_pertensor rel_err", pts[lane]["rel_err"], "ratio", pts[lane], None, flt(pts[lane], t=2))
        sh.line(f"  {lane.upper()} int8_pertensor: {float(pts[lane]['rel_err']):.3e}  [{short(pts[lane])} | {flt(pts[lane], t=2)}]")
    for lane in ("cpu", "gpu"):
        d = sweep(agg, lane, "mixed_sep", SHAPE).set_index("iso_out_pct")
        parts = []
        for p in (0, 1, 5, 10):
            row = d.loc[float(p)]
            f =float(pts[lane]["rel_err"]) / float(row["rel_err"])
            parts.append(f"p={p}: {float(row['rel_err']):.3e} (x{f:.1f})")
            sh.add(2, f"{lane} mixed_sep p={p} rel_err", row["rel_err"], "ratio", row, None, f"{row['variant']}, {SHAPE}, t=2, p={p}")
            sh.add(2, f"{lane} factor pertensor/mixed_sep p={p}", f, "x", row, None, f"{row['variant']}, {SHAPE}, t=2, p={p}")
        sh.line(f"  {lane.upper()} mixed_sep  " + "; ".join(parts) + f"  [{short(sweep(agg, lane, 'mixed_sep', SHAPE).iloc[0])} | sq4096 t=2]")


def section_stages(agg, sh):
    sh.line("(3) stage breakdown, mixed_sep, sq4096, t = 2, p = 5: ms, median [min-max]; share of the stage sum")
    for lane in ("cpu", "gpu"):
        d = sweep(agg, lane, "mixed_sep", SHAPE)
        row = d[d["iso_out_pct"] == 5.0].iloc[0]
        stages = ["t_select_ms", "t_quant_ms", "t_int8_ms", "t_outlier_ms", "t_merge_ms"]
        total = sum(row[c] for c in stages)
        parts = []
        for col, name in zip(stages, ["select", "quant", "INT8", "outlier", "merge"]):
            parts.append(f"{name} {rng(row, col, 2)} ({row[col] / total * 100:.1f}%)")
            sh.add(3, f"{lane} mixed_sep p=5 {name} ms", row[col], "ms", row, col, flt(row, t=2, p=5))
        sh.line(f"  {lane.upper()}: " + "; ".join(parts))
        sh.line(f"      stage sum {total:.2f} ms; total_ms_median {rng(row, 'total_ms', 2)}  [{short(row)} | {flt(row, t=2, p=5)}]")


def section_ladder(agg, sh):
    sh.line("(4) ladder, sq2048, t = 2, p = 5: GFLOP/s median [min-max] n")
    scalar = None
    for label, variant, threads in [("scalar 1 thread", "mixed_sep_scalar", 1), ("AVX2 1 thread", "mixed_sep_avx2", 1),
                                    ("AVX2 + OpenMP 16 threads", "mixed_sep_avx2", 16)]:
        r = sel(agg, lane="cpu", group="C", variant=variant, shape="sq2048", true_out_pct=T, iso_out_pct=5.0,
                threads=float(threads))
        scalar = r if scalar is None else scalar
        speed = r["gflops"] / scalar["gflops"]
        sh.line(f"  CPU {label}: {rng(r, 'gflops')}, x{speed:.1f} over scalar  [{short(r)} | {flt(r, t=2, p=5, threads=threads)}]")
        sh.add(4, f"cpu ladder {label} gflops", r["gflops"], "GFLOP/s", r, "gflops", flt(r, t=2, p=5, threads=threads))
        sh.add(4, f"cpu ladder {label} speedup over scalar", speed, "x", r, None, flt(r, t=2, p=5, threads=threads))
    gpu = [("int8_pertensor", baseline(agg, "gpu", "int8_pertensor", "sq2048"))]
    for fam in ("mixed_sep", "mixed_fused"):
        d = sweep(agg, "gpu", fam, "sq2048")
        gpu.append((fam, d[d["iso_out_pct"] == 5.0].iloc[0]))
    for label, r in gpu:
        sh.line(f"  GPU {label}: {rng(r, 'gflops')}  [{short(r)} | {flt(r, t=2, p=5)}]")
        sh.add(4, f"gpu ladder {label} gflops", r["gflops"], "GFLOP/s", r, "gflops", flt(r, t=2, p=5))


def overlap_text(opt, base, col):
    if opt[col + "_max"] < base[col + "_min"]:
        return "opt1 range entirely below base range"
    if opt[col + "_min"] > base[col + "_max"]:
        return "opt1 range entirely above base range"
    return "ranges overlap"


def section_opt1(agg, sh):
    sh.line("(5) opt1 (mixed_sep_avx2_opt1) vs mixed_sep_avx2, same job, t = 2, 16 threads, n = 4 each: total_ms and INT8 stage, median [min-max]")
    for shape in ("sq2048", "sq4096"):
        for p in (1, 5, 10):
            kw = dict(lane="cpu", group="E", shape=shape, true_out_pct=T, iso_out_pct=float(p), threads=16.0, tile=None)
            base, opt = sel(agg, variant="mixed_sep_avx2", **kw), sel(agg, variant="mixed_sep_avx2_opt1", **kw)
            for col, name in (("total_ms", "total"), ("t_int8_ms", "INT8 stage")):
                change = (opt[col] / base[col] - 1.0) * 100.0
                note = overlap_text(opt, base, col)
                sh.line(f"  {shape} p={p} {name}: base {rng(base, col, 2)} -> opt1 {rng(opt, col, 2)}: {change:+.1f}%, {note}")
                sh.add(5, f"cpu {shape} p={p} {name} opt1 vs base", change, "%", opt, col, f"{flt(opt, t=2, p=p)} vs base; {note}")
    sh.line("  sources: cpu/E_opt.csv+rep1,2,3 (opt1) and cpu/E_base.csv+rep1,2,3 (base)")
    section_opt1_gpu(agg, sh)


def section_opt1_gpu(agg, sh):
    """GPU group E: the paired control of one job is the main comparison; E_opt.csv is a second job."""
    note = "n = 1, no noise range: indicative, not a finding"
    configs = [(shape, p) for shape in ("sq2048", "sq4096") for p in (1, 5, 10)]
    kw = dict(lane="gpu", true_out_pct=T, threads=None, tile=16.0)
    sh.line(f"  GPU paired control, same job: mixed_sep (E_base.csv) vs mixed_sep_opt1 (E_pair_opt.csv), t = 2, tile 16; {note}")
    for shape, p in configs:
        base = sel(agg, group="E_base", variant="mixed_sep", shape=shape, iso_out_pct=float(p), **kw)
        opt = sel(agg, group="E_pair_opt", variant="mixed_sep_opt1", shape=shape, iso_out_pct=float(p), **kw)
        for col, name in (("total_ms", "total"), ("t_int8_ms", "INT8 stage")):
            change = (opt[col] / base[col] - 1.0) * 100.0
            sh.line(f"    {shape} p={p} {name}: base {rng(base, col, 2)} -> opt1 {rng(opt, col, 2)}: {change:+.1f}%")
            sh.add(5, f"gpu {shape} p={p} {name} paired opt1 vs base", change, "%", opt, col,
                   f"{flt(opt, t=2, p=p)} vs gpu/E_base.csv; {note}")
    sh.line("    sources: gpu/E_base.csv and gpu/E_pair_opt.csv")
    sh.line(f"  GPU job-to-job difference, same variant mixed_sep_opt1: E_opt.csv relative to E_pair_opt.csv; {note}")
    for shape, p in configs:
        a = sel(agg, group="E_opt", variant="mixed_sep_opt1", shape=shape, iso_out_pct=float(p), **kw)
        b = sel(agg, group="E_pair_opt", variant="mixed_sep_opt1", shape=shape, iso_out_pct=float(p), **kw)
        parts = []
        for col, name in (("total_ms", "total"), ("t_int8_ms", "INT8 stage")):
            diff = (a[col] / b[col] - 1.0) * 100.0
            parts.append(f"{name} {a[col]:.2f} vs {b[col]:.2f} ({diff:+.1f}%)")
            sh.add(5, f"gpu {shape} p={p} {name} E_opt vs E_pair_opt", diff, "%", a, col,
                   f"{flt(a, t=2, p=p)}, E_opt.csv relative to E_pair_opt.csv; {note}")
        sh.line(f"    {shape} p={p}: " + "; ".join(parts))
    sh.line("    sources: gpu/E_opt.csv and gpu/E_pair_opt.csv")


def read_golden(path):
    out = []
    for line in open(path).read().splitlines():
        parts = line.split()
        out.append((parts[0], parts[2].split("=")[1], parts[3]))  # variant, p, checksum
    return out


def section_correctness(agg, sh):
    sh.line("(6) correctness")
    cpu = read_golden(os.path.join(ROOT, "results", "cpu", "golden", "checksums_cpu.txt"))
    gpu = read_golden(os.path.join(ROOT, "results", "gpu", "golden", "checksums_gpu.txt"))
    for p in ("0", "2", "10"):
        c = {h for v, q, h in cpu if q == p and v.startswith("mixed_") and not v.endswith("_opt1")}
        o = {h for v, q, h in cpu if q == p and v.endswith("_opt1")}
        g = {h for v, q, h in gpu if q == p and v.startswith("mixed_") and not v.endswith("_opt1")}
        go = {h for v, q, h in gpu if q == p and v.endswith("_opt1")}
        nc = sum(1 for v, q, h in cpu if q == p and v.startswith("mixed_") and not v.endswith("_opt1"))
        ng = sum(1 for v, q, h in gpu if q == p and v.startswith("mixed_") and not v.endswith("_opt1"))
        ngo = sum(1 for v, q, h in gpu if q == p and v.endswith("_opt1"))
        same = "equal" if c == g == o and len(c) == 1 else "NOT equal"
        same_opt = "equal" if go == o and len(o) == 1 else "NOT equal"
        sh.line(f"  golden p={p}: CPU {nc} runs, {len(c)} distinct; opt1 {len(o)} distinct; GPU {ng} runs, {len(g)} distinct; CPU vs GPU vs opt1: {same}; "
                f"GPU opt1 {ngo} runs, {len(go)} distinct, equal to CPU opt1: {same_opt}")
        sh.add(6, f"golden p={p} CPU/GPU/opt1 checksums", same, "", None, None,
               "results/cpu/golden/checksums_cpu.txt and results/gpu/golden/checksums_gpu.txt")
        sh.add(6, f"golden p={p} GPU opt1 equal to CPU opt1", same_opt, "", None, None,
               "results/cpu/golden/checksums_cpu.txt and results/gpu/golden/checksums_gpu.txt; GPU mixed_sep_opt1 lines")
    pc = {h for v, q, h in cpu if v == "int8_pertensor"}
    pg = {h for v, q, h in gpu if v == "int8_pertensor"}
    fp = {v: h for v, q, h in cpu if v.startswith("fp32_")}
    sh.line(f"  golden int8_pertensor: CPU {len(pc)} distinct, GPU {len(pg)} distinct, {'equal' if pc == pg else 'NOT equal'}; "
            f"fp32_omp vs fp32_serial: {'equal' if fp['fp32_omp'] == fp['fp32_serial'] else 'NOT equal'}")
    sh.line(s7_text(agg, sh))


def s7_text(agg, sh):
    keep = agg[agg["group"].isin(["A", "B"]) & (agg["family"].str.startswith("mixed") | (agg["family"] == "int8_pertensor"))]
    cols = ["family", "shape", "true_out_pct", "iso_out_pct"]
    m = keep[keep["lane"] == "cpu"].merge(keep[keep["lane"] == "gpu"], on=cols, suffixes=("_cpu", "_gpu"))
    equal = int((m["rel_err_cpu"] == m["rel_err_gpu"]).sum())
    varying = int((keep["rel_err_distinct"] > 1).sum())
    sh.add(6, "rel_err keys equal CPU vs GPU", equal, "keys", None, None,
           f"results/cpu/{{A,B}}_*.csv(+rep) vs results/gpu/{{A,B}}_*.csv; {len(m)} keys compared")
    return (f"  rel_err CPU = GPU (exact text) on {equal} of {len(m)} keys (mixed_sep, mixed_fused, int8_pertensor); "
            f"keys with differing rel_err between processes: {varying}  [cpu/{{A,B}}_*.csv+rep1,2 | gpu/{{A,B}}_*.csv]")


def grab(text, pattern):
    m = re.search(pattern, text, re.M)
    return m.group(1).strip() if m else "not found"


def section_machine(agg, runs, sh):
    cpu = open(os.path.join(ROOT, "results", "cpu", "env_B_sweep.txt")).read()
    gpu = open(os.path.join(ROOT, "results", "gpu", "env_B_sweep.txt")).read()
    threads = sorted(runs[(runs["lane"] == "cpu") & (runs["group"] == "B")]["threads"].unique())
    tiles_b = sorted(runs[(runs["lane"] == "gpu") & (runs["group"] == "B")]["tile"].unique())
    tiles_c = sorted(runs[(runs["lane"] == "gpu") & (runs["group"] == "C")]["tile"].unique())
    # Patterns are kept out of the f-strings: before Python 3.12 an f-string expression cannot hold a backslash.
    model = grab(cpu, r"^cpu_model: Model name:\s*(.+)$")
    node = grab(cpu, r"^hostname: (\S+)")
    nproc = grab(cpu, r"^nproc: (\d+)")
    cores = grab(cpu, r"^Core\(s\) per socket:\s*(\d+)")
    smt = grab(cpu, r"^Thread\(s\) per core:\s*(\d+)")
    gxx = grab(cpu, r"^g\+\+ \(GCC\) (\S+)")
    flags = grab(cpu, r"^(g\+\+ -std=c\+\+17 -O3 -mavx2 -mf16c -fopenmp -ffp-contract=off) -I")
    gname = grab(gpu, r"\|\s*\d+\s+(NVIDIA GeForce .+?)\s+(?:On|Off)\s*\|")
    gmem = grab(gpu, r"/\s*(\d+MiB)")
    gnode = grab(gpu, r"== hostname\n(\S+)")
    gpart = grab(gpu, r"SLURM_JOB_PARTITION=(\S+)")
    driver = grab(gpu, r"Driver Version: (\S+)")
    nvcc = grab(gpu, r"release ([\d.]+)")
    nvflags = grab(gpu, r"NVCCFLAGS = (.+)$")
    sh.line("(7) machine")
    sh.line(f"  CPU [cpu/env_B_sweep.txt]: {model}, node {node}, nproc {nproc}, {cores} cores x {smt} thread per core, g++ {gxx}")
    sh.line(f"      flags: {flags}; threads in group B: {', '.join(str(int(t)) for t in threads)}")
    sh.line(f"  GPU [gpu/env_B_sweep.txt]: {gname}, {gmem}, node {gnode}, partition {gpart}, driver {driver}, nvcc release {nvcc}")
    sh.line(f"      NVCCFLAGS: {nvflags}; tile in group B: {', '.join(str(int(t)) for t in tiles_b)}; "
            f"tiles in group C: {', '.join(str(int(t)) for t in tiles_c)}")
    gpu_env_files(sh)


def gpu_env_files(sh):
    """Node and driver of every GPU env file, read from the files; a disagreement between files is shown."""
    sh.line("  GPU env files [results/gpu/env_*.txt | hostname and nvidia-smi sections]:")
    by_driver = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "gpu", "env_*.txt"))):
        text = open(path).read()
        name = os.path.basename(path)
        driver = grab(text, r"Driver Version: (\S+)")
        if driver == "not found":  # the newer nvidia-smi layout prints the kernel-mode driver version instead
            driver = grab(text, r"KMD Version: (\S+)")
        node = grab(text, r"== hostname\n(\S+)")  # kept out of the f-string: no backslash allowed there before Python 3.12
        sh.line(f"      {name}: node {node}, driver {driver}")
        by_driver.setdefault(driver, []).append(name)
    summary = "; ".join(f"{d} in {', '.join(names)}" for d, names in by_driver.items())
    sh.line(f"      distinct driver values: {len(by_driver)}: {summary}")


def section_caveats(runs, sh):
    s5 = s5_counts(runs)
    flagged, rows = s5["sq1024"]
    other_f = sum(f for s, (f, r) in s5.items() if s != "sq1024")
    other_r = sum(r for s, (f, r) in s5.items() if s != "sq1024")
    sh.line("(8) caveats for the slides")
    for text in [
        "synthetic data, one outlier magnitude (x50), one seed (COVERAGE.md section 6, list of claims we must not make)",
        "GPU rows are single processes (n = 1): no noise range, so no GPU cost is a finding; CPU ranges are n = 3 (n = 4 for E), only a rough estimate",
        "int8_pertensor on the CPU is a plain-loop kernel, the mixed variants use AVX2: a CPU comparison against it compares kernel styles (docs/LANE_NOTES_cpu.md)",
        f"sq1024 is noisy: {flagged} of {rows} sq1024 CPU process rows (groups A, B, C, original + replicates) have stddev/median above 0.10; "
        f"all other shapes together {other_f} of {other_r} rows (groups A, B, C, E)",
        "GPU: host-side top-k selection and stage-boundary sync in the GPU timings; GPU node sharing status (COVERAGE.md section 6; not checked in GPU code)",
        NOISE_NOTE]:
        sh.line(f"  - {text}")
    sh.add(8, "sq1024 CPU process rows above S5", flagged, "rows", None, None, f"{rows} sq1024 rows in cpu A, B, C original + rep")


def build(agg, runs):
    sh = Sheet()
    sh.line("FACTS SHEET: format is value [min-max] n=processes [file | filter]; machines are in section (7)")
    for fn in (section_throughput, section_rel_err, section_stages, section_ladder, section_opt1, section_correctness):
        fn(agg, sh)
        sh.line()
    section_machine(agg, runs, sh)
    sh.line()
    section_caveats(runs, sh)
    return pd.DataFrame(sh.rows), "\n".join(sh.lines) + "\n"
