"""analysis/data.py: loads every result CSV of both lanes and aggregates the processes of each configuration.

One CSV row is one process (SPEC 11). The CPU lane has 3 processes per row of groups A, B and C (the original
run plus replicates 1 and 2, results/cpu/rep/) and 4 per row of group E (original plus replicates 1 to 3).
The GPU lane has 1 process per row. The aggregate holds, per configuration, the median over its processes
and the minimum and maximum, plus n. The range of 3 or 4 processes is only a rough noise estimate.
"""
import glob
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LANES = ["cpu", "gpu"]
STAGE_COLS = ["t_select_ms", "t_quant_ms", "t_int8_ms", "t_outlier_ms", "t_merge_ms"]
KEY = ["lane", "group", "variant", "shape", "true_out_pct", "iso_out_pct", "threads", "tile"]
NOISE_NOTE = ("Noise range = min to max over the processes of the same configuration: n = 3 (groups A, B, C) or "
              "n = 4 (group E) on the CPU, n = 1 on the GPU (no range). This is only a rough noise estimate.")
S5_RATIO = 0.10  # COVERAGE section 4, check S5


def _read(path, lane, group):
    df = pd.read_csv(path, dtype={"rel_err": str})  # rel_err stays text so lanes are compared exactly
    df["lane"] = lane
    df["group"] = group
    df["file"] = os.path.relpath(path, ROOT)
    return df


def load_runs():
    """Every row of every process, with the group letter and the file it came from."""
    frames = []
    for lane in LANES:
        base = os.path.join(ROOT, "results", lane)
        for path in sorted(glob.glob(base + "/[ABC]_*.csv") + glob.glob(base + "/rep/[ABC]_*_rep*.csv")):
            frames.append(_read(path, lane, os.path.basename(path)[0]))
    base = os.path.join(ROOT, "results", "cpu")
    for name in ("E_opt", "E_base"):
        for path in sorted(glob.glob(f"{base}/{name}.csv") + glob.glob(f"{base}/rep/{name}_rep*.csv")):
            frames.append(_read(path, "cpu", "E"))
    # GPU group E files each keep their own group name (E_base, E_opt, E_pair_opt): E_opt and E_pair_opt hold
    # the same variant in two different jobs and must not be merged into one configuration.
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "gpu", "E_*.csv"))):
        frames.append(_read(path, "gpu", os.path.basename(path)[:-len(".csv")]))
    return pd.concat(frames, ignore_index=True)


def aggregate(runs):
    """Median, min, max and n per configuration; 'family' drops _avx2 so the lanes line up."""
    spec = {"n": ("gflops", "size"), "gflops": ("gflops", "median"), "gflops_min": ("gflops", "min"),
            "gflops_max": ("gflops", "max"), "total_ms": ("total_ms_median", "median"),
            "total_ms_min": ("total_ms_median", "min"), "total_ms_max": ("total_ms_median", "max"),
            "rel_err": ("rel_err", "first"), "rel_err_distinct": ("rel_err", "nunique"),
            "k": ("k_isolated", "first"), "recall": ("outlier_recall", "first"),
            "sources": ("file", lambda s: " + ".join(sorted(set(s))))}
    for col in STAGE_COLS:
        spec[col] = (col, "median")
        spec[col + "_min"] = (col, "min")
        spec[col + "_max"] = (col, "max")
    agg = runs.groupby(KEY, dropna=False).agg(**spec).reset_index()
    agg["family"] = agg["variant"].str.replace("_avx2", "", regex=False)
    return agg


def sel(agg, **kw):
    """The one configuration matching the keywords (None matches an empty cell); exits if not exactly one."""
    mask = pd.Series(True, index=agg.index)
    for key, value in kw.items():
        mask &= agg[key].isna() if value is None else (agg[key] == value)
    found = agg[mask]
    if len(found) != 1:
        raise SystemExit(f"data.py: expected one configuration for {kw}, found {len(found)}")
    return found.iloc[0]


def sweep(agg, lane, family, shape, t=2.0):
    """Group B rows of one series ordered by p (the GPU B rows are all at the default tile 16)."""
    d = agg[(agg["lane"] == lane) & (agg["group"] == "B") & (agg["family"] == family)
            & (agg["shape"] == shape) & (agg["true_out_pct"] == t)]
    return d.sort_values("iso_out_pct")


def baseline(agg, lane, family, shape, t=2.0):
    """The group A row (fp32 or int8_pertensor) of one lane."""
    return sel(agg, lane=lane, group="A", variant=family, shape=shape, true_out_pct=t)


def cost(row, ref):
    """Throughput cost of row against ref in percent: (value, lowest, highest, verdict).

    lowest and highest pair the best of one range with the worst of the other, so they bound the cost that
    the min-max ranges allow. The verdict is "above noise" only when that whole interval lies on one side of
    zero. With n = 1 there is no range and the verdict says so."""
    value = (1.0 - row["gflops"] / ref["gflops"]) * 100.0
    if min(row["n"], ref["n"]) < 2:
        return value, float("nan"), float("nan"), "n = 1, no noise range: not a finding"
    low = (1.0 - row["gflops_max"] / ref["gflops_min"]) * 100.0
    high = (1.0 - row["gflops_min"] / ref["gflops_max"]) * 100.0
    if low > 0:
        return value, low, high, "above noise (cost)"
    if high < 0:
        return value, low, high, "above noise (gain)"
    return value, low, high, "within noise: not a finding"


def s5_counts(runs):
    """CPU rows with total_ms_stddev / total_ms_median above the S5 threshold, per shape: {shape: (flagged, rows)}."""
    cpu = runs[runs["lane"] == "cpu"]
    flagged = cpu["total_ms_stddev"] / cpu["total_ms_median"] > S5_RATIO
    return {s: (int(flagged[cpu["shape"] == s].sum()), int((cpu["shape"] == s).sum())) for s in sorted(cpu["shape"].unique())}
