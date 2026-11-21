"""analysis/make_assets.py: talk figures F3, F4, F1 (sq4096), F6 and headline_numbers.csv (COVERAGE section 6).

Reads results/cpu/*.csv, results/gpu/*.csv and results/cpu/profile/repeat_runs.csv; writes PNG and PDF
figures and headline_numbers.csv to analysis/out/. Run from anywhere: paths are relative to this file.
Only pandas and matplotlib are used.

Noise rule: the process-to-process spread of the same configuration is measured only for the CPU at
sq4096, t = 2 (repeat_runs.csv, 5 separate processes each at p = 0 and p = 5). A cost or a gain is called
"above noise" only when it is larger than that spread. The GPU has no replicate processes, so its costs
are written with the verdict "noise not measured" and are not findings.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "analysis", "out")
LANES = ["cpu", "gpu"]
P_LEVELS = [0, 0.5, 1, 2, 5, 10, 20]
STAGES = [("t_select_ms", "select"), ("t_quant_ms", "quantize"), ("t_int8_ms", "INT8 GEMM"),
          ("t_outlier_ms", "outlier GEMM"), ("t_merge_ms", "merge")]

# Categorical slots 1 to 3 of the validated reference palette (light mode); neutral grey for the FP32 line.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
COLOR = {"mixed_sep": "#2a78d6", "mixed_fused": "#eb6834", "int8_pertensor": "#1baf7a", "fp32": "#52514e"}
STAGE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
LABEL = {"mixed_sep": "mixed, separate passes", "mixed_fused": "mixed, fused merge",
         "int8_pertensor": "int8_pertensor", "fp32": "FP32 baseline"}

NOISE_LIMITS = ("Spread measured at sq4096 only, at p = 0 and p = 5 only, from 5 separate processes each; "
                "assumed for p = 1 and p = 10; not measured for other shapes or for the GPU.")

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "legend.fontsize": 13,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2.0, "lines.markersize": 8,
})


def load():
    """All CSVs of both lanes in one table; 'family' drops the _avx2 suffix so the lanes line up."""
    frames = []
    for lane in LANES:
        folder = os.path.join(ROOT, "results", lane)
        for name in sorted(os.listdir(folder)):
            if name.endswith(".csv") and name[0] in "ABC" and name[1] == "_":
                df = pd.read_csv(os.path.join(folder, name))
                df["file"] = f"results/{lane}/{name}"
                frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["family"] = df["variant"].str.replace("_avx2", "", regex=False)
    return df


def noise_table():
    """Spread of total_ms_median over separate processes: {p: (min_ms, max_ms, spread_pct)}."""
    rep = pd.read_csv(os.path.join(ROOT, "results", "cpu", "profile", "repeat_runs.csv"))
    rep = rep[(rep["variant"] == "mixed_sep_avx2") & (rep["shape"] == "sq4096")]
    table = {}
    for p, g in rep.groupby("iso_out_pct"):
        lo, hi, med = g["total_ms_median"].min(), g["total_ms_median"].max(), g["total_ms_median"].median()
        table[float(p)] = (lo, hi, 100.0 * (hi - lo) / med, len(g), g["gflops"].min(), g["gflops"].max())
    return table


def pick(df, lane, family, shape, t=2.0, tile=None):
    d = df[(df["lane"] == lane) & (df["family"] == family) & (df["shape"] == shape) & (df["true_out_pct"] == t)]
    if tile is not None:
        d = d[d["tile"] == tile]
    return d


def sweep(df, lane, family, shape, t=2.0):
    """B_sweep rows of one series, ordered by p (GPU B rows are at the default tile 16)."""
    d = pick(df, lane, family, shape, t)
    return d[d["file"].str.endswith("B_sweep.csv")].sort_values("iso_out_pct")


def single(df, lane, family, shape, t=2.0):
    d = pick(df, lane, family, shape, t)
    d = d[d["file"].str.endswith("A_baselines.csv")]
    return d.iloc[0]


def baseline_family(lane):
    return "fp32_omp" if lane == "cpu" else "fp32_cublas"


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def xpos(p_values):
    return [P_LEVELS.index(p) for p in p_values]


def set_p_axis(ax):
    ax.set_xticks(range(len(P_LEVELS)))
    ax.set_xticklabels([f"{p:g}" for p in P_LEVELS])
    ax.set_xlabel("isolated channels p (%)")


def fig_f1(df, noise):
    """F1: GFLOP/s vs p at sq4096, t = 2, one panel per lane, with reference lines and the CPU noise bars."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, lane in zip(axes, LANES):
        for fam in ("mixed_sep", "mixed_fused"):
            d = sweep(df, lane, fam, "sq4096")
            ax.plot(xpos(d["iso_out_pct"]), d["gflops"], marker="o", color=COLOR[fam], label=LABEL[fam])
        pt = single(df, lane, "int8_pertensor", "sq4096")["gflops"]
        fp = single(df, lane, baseline_family(lane), "sq4096")["gflops"]
        pt_label = "int8_pertensor (plain loop)" if lane == "cpu" else "int8_pertensor"
        ax.axhline(pt, color=COLOR["int8_pertensor"], linestyle="--", label=pt_label)
        ax.axhline(fp, color=COLOR["fp32"], linestyle=":", label=LABEL["fp32"])
        if lane == "cpu":
            for p, (_, _, _, n, g_lo, g_hi) in noise.items():
                ax.vlines(P_LEVELS.index(p), g_lo, g_hi, color=INK, linewidth=6, alpha=0.35,
                          label=f"spread over {n} processes" if p == 0 else None)
        ax.set_title(f"{lane.upper()}, sq4096, t = 2")
        ax.set_ylabel("GFLOP/s")
        ax.set_yscale("log" if lane == "gpu" else "linear")
        set_p_axis(ax)
        ax.legend(loc="center right" if lane == "cpu" else "lower left")
    fig.tight_layout()
    fig.text(0.01, -0.02, "Grey bars (CPU): range over 5 separate processes at p = 0 and p = 5 only (sq4096). "
             "The GPU has no replicate processes.", fontsize=11, color=INK2)
    save(fig, "F1_gflops_vs_p_sq4096")


def fig_f3(df):
    """F3: accuracy (rel_err, log x) against throughput, points labelled by p."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for r, shape in enumerate(["sq2048", "sq4096"]):
        for c, lane in enumerate(LANES):
            ax = axes[r][c]
            for fam in ("mixed_sep", "mixed_fused"):
                d = sweep(df, lane, fam, shape)
                ax.plot(d["rel_err"], d["gflops"], marker="o", color=COLOR[fam], label=LABEL[fam])
                if fam == "mixed_sep":
                    for _, row in d.iterrows():
                        ax.annotate(f"{row['iso_out_pct']:g}", (row["rel_err"], row["gflops"]),
                                    textcoords="offset points", xytext=(6, 6), fontsize=12, color=INK2)
            pt = single(df, lane, "int8_pertensor", shape)
            ax.plot(pt["rel_err"], pt["gflops"], marker="*", markersize=16, linestyle="none",
                    color=COLOR["int8_pertensor"], label="int8_pertensor")
            ax.set_xscale("log")
            ax.set_title(f"{lane.upper()}, {shape}, t = 2 (labels: p in %)")
            ax.set_xlabel("relative error vs FP32 (log)")
            ax.set_ylabel("GFLOP/s")
            if r == 0 and c == 0:
                ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "F3_tradeoff")


def fig_f4(df):
    """F4: stage times of mixed_sep at sq4096, t = 2, stacked per p."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, lane in zip(axes, LANES):
        d = sweep(df, lane, "mixed_sep", "sq4096")
        x = xpos(d["iso_out_pct"])
        bottom = [0.0] * len(d)
        for (col, name), color in zip(STAGES, STAGE_COLORS):
            vals = d[col].fillna(0.0).tolist()
            ax.bar(x, vals, bottom=bottom, color=color, edgecolor=SURFACE, linewidth=1.5, label=name, width=0.7)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_title(f"{lane.upper()}, mixed_sep, sq4096, t = 2")
        ax.set_ylabel("time per repetition (ms)")
        set_p_axis(ax)
        ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "F4_stage_times")


def ladder_bars(df):
    """(label, GFLOP/s, family) for the CPU ladder (group C, sq2048, t = 2, p = 5) and the GPU ladder."""
    cpu = df[(df["lane"] == "cpu") & (df["shape"] == "sq2048") & (df["iso_out_pct"] == 5)
             & df["file"].str.endswith("C_ladder.csv")]
    rows = []
    for label, variant, threads in [("scalar\n1 thread", "mixed_sep_scalar", 1), ("AVX2\n1 thread", "mixed_sep_avx2", 1),
                                    ("AVX2 + OpenMP\n16 threads", "mixed_sep_avx2", 16)]:
        rows.append(("cpu", label, cpu[(cpu["variant"] == variant) & (cpu["threads"] == threads)].iloc[0]["gflops"], "mixed_sep"))
    for label, fam in [("int8_pertensor", "int8_pertensor"), ("mixed,\nseparate", "mixed_sep"), ("mixed,\nfused", "mixed_fused")]:
        if fam == "int8_pertensor":
            g = single(df, "gpu", fam, "sq2048")["gflops"]
        else:
            g = sweep(df, "gpu", fam, "sq2048").query("iso_out_pct == 5").iloc[0]["gflops"]
        rows.append(("gpu", label, g, fam))
    return rows


def fig_f6(df):
    """F6: the ladder at sq2048, t = 2, p = 5, log y because CPU and GPU differ by orders of magnitude."""
    rows = ladder_bars(df)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, lane in zip(axes, LANES):
        sel = [r for r in rows if r[0] == lane]
        ax.bar(range(len(sel)), [r[2] for r in sel], color=[COLOR[r[3]] for r in sel], width=0.6)
        for i, r in enumerate(sel):
            ax.text(i, r[2] * 1.08, f"{r[2]:,.0f}", ha="center", fontsize=13, color=INK)
        ax.set_xticks(range(len(sel)))
        ax.set_xticklabels([r[1] for r in sel])
        ax.set_yscale("log")
        ax.set_ylim(top=max(r[2] for r in sel) * 2.2)
        ax.set_ylabel("GFLOP/s (log)")
        ax.set_title(f"{lane.upper()}, sq2048, t = 2" + (", p = 5" if lane == "cpu" else ", p = 5 (mixed)"))
    fig.tight_layout()
    save(fig, "F6_ladder")


def verdict(cost, lane, noise_pct):
    if lane != "cpu":
        return "noise not measured (no replicate processes): not a finding"
    return "above noise" if abs(cost) > noise_pct else "within noise: not a finding"


def headline(df, noise):
    """Every quoted number with its source file and filter. Costs carry the noise rule."""
    rows = []

    def add(name, value, unit, source, filt):
        rows.append({"name": name, "value": value, "unit": unit, "source": source, "filter": filt})

    for p, (lo, hi, pct, n, g_lo, g_hi) in noise.items():
        src, filt = "results/cpu/profile/repeat_runs.csv", f"mixed_sep_avx2, sq4096, iso_out_pct={p:g}, {n} processes"
        add(f"cpu sq4096 p={p:g} total_ms_median min over processes", lo, "ms", src, filt)
        add(f"cpu sq4096 p={p:g} total_ms_median max over processes", hi, "ms", src, filt)
        add(f"cpu sq4096 p={p:g} spread (max-min)/median", pct, "%", src, filt)
    noise_pct = max(v[2] for v in noise.values())
    add("cpu noise threshold used for costs", noise_pct, "%", "results/cpu/profile/repeat_runs.csv",
        "larger of the p=0 and p=5 spreads. " + NOISE_LIMITS)
    for lane in LANES:
        base = single(df, lane, baseline_family(lane), "sq4096")
        add(f"{lane} sq4096 fp32 baseline gflops", base["gflops"], "GFLOP/s", base["file"], f"{base['variant']}, t=2")
        pt = single(df, lane, "int8_pertensor", "sq4096")
        add(f"{lane} sq4096 int8_pertensor gflops", pt["gflops"], "GFLOP/s", pt["file"], "t=2")
        add(f"{lane} sq4096 int8_pertensor rel_err", pt["rel_err"], "ratio", pt["file"], "t=2")
        for fam in ("mixed_sep", "mixed_fused"):
            d = sweep(df, lane, fam, "sq4096").set_index("iso_out_pct")
            g0 = d.loc[0, "gflops"]
            for p in (0, 1, 5, 10):
                row, filt = d.loc[p], f"{row_variant(d, p)}, sq4096, t=2, iso_out_pct={p}"
                add(f"{lane} {fam} sq4096 p={p} gflops", row["gflops"], "GFLOP/s", row["file"], filt)
                add(f"{lane} {fam} sq4096 p={p} rel_err", row["rel_err"], "ratio", row["file"], filt)
                if p > 0:
                    cost = (1.0 - row["gflops"] / g0) * 100.0
                    add(f"{lane} {fam} sq4096 p={p} throughput cost vs p=0", cost, "%", row["file"], filt + "; 1 - g(p)/g(0)")
                    add(f"{lane} {fam} sq4096 p={p} cost verdict", verdict(cost, lane, noise_pct), "", row["file"], filt)
    for lane, label, g, fam in ladder_bars(df):
        add(f"{lane} ladder sq2048 p=5 {label.replace(chr(10), ' ')}", g, "GFLOP/s",
            f"results/{lane}/" + ("C_ladder.csv" if lane == "cpu" else "A_baselines.csv and B_sweep.csv"), "t=2")
    return pd.DataFrame(rows)


def row_variant(d, p):
    return d.loc[p, "variant"]


def main():
    df = load()
    noise = noise_table()
    fig_f1(df, noise)
    fig_f3(df)
    fig_f4(df)
    fig_f6(df)
    os.makedirs(OUT, exist_ok=True)
    headline(df, noise).to_csv(os.path.join(OUT, "headline_numbers.csv"), index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
