"""analysis/figures.py: the talk figures F1 (sq4096), F3, F4 and F6 (COVERAGE section 6).

CPU points are medians over the processes of a configuration with min-max bars (n = 3); GPU points are one
process each and carry no bars. Colors are slots 1 to 3 of the validated reference palette (light mode),
the same for the same variant in every figure.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from data import LANES, ROOT, STAGE_COLS, baseline, sel, sweep

OUT = os.path.join(ROOT, "analysis", "out")
P_LEVELS = [0, 0.5, 1, 2, 5, 10, 20]
STAGES = list(zip(STAGE_COLS, ["select", "quantize", "INT8 GEMM", "outlier GEMM", "merge"]))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
COLOR = {"mixed_sep": "#2a78d6", "mixed_fused": "#eb6834", "int8_pertensor": "#1baf7a", "fp32": "#52514e"}
STAGE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
LABEL = {"mixed_sep": "mixed, separate passes", "mixed_fused": "mixed, fused merge",
         "int8_pertensor": "int8_pertensor", "fp32": "FP32 baseline"}
# Offsets (points) of the p labels in F3, chosen so neighbouring points do not collide.
LABEL_OFFSET = {0: (10, -20), 0.5: (-34, 8), 1: (10, 8), 2: (-30, 8), 5: (12, -4), 10: (12, 2), 20: (12, -6)}

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "legend.fontsize": 13,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2.0, "lines.markersize": 8,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def set_p_axis(ax):
    ax.set_xticks(range(len(P_LEVELS)))
    ax.set_xticklabels([f"{p:g}" for p in P_LEVELS])
    ax.set_xlabel("isolated channels p (%)")


def xpos(d, shift=0.0):
    return [P_LEVELS.index(p) + shift for p in d["iso_out_pct"]]


def range_bars(ax, x, d, col, color):
    """Min-max bars of the processes behind each point; only where there is more than one process."""
    multi = d["n"] > 1
    if multi.any():
        ax.vlines([xi for xi, m in zip(x, multi) if m], d[col + "_min"][multi], d[col + "_max"][multi],
                  color=color, linewidth=3, alpha=0.45)


def fig_f1(agg):
    """F1: GFLOP/s vs p at sq4096, t = 2; reference lines are the int8_pertensor and FP32 baselines."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, lane in zip(axes, LANES):
        for fam, shift in (("mixed_sep", -0.06), ("mixed_fused", 0.06)):
            d = sweep(agg, lane, fam, "sq4096")
            x = xpos(d, shift)
            range_bars(ax, x, d, "gflops", COLOR[fam])
            ax.plot(x, d["gflops"], marker="o", color=COLOR[fam], label=LABEL[fam])
        pt = baseline(agg, lane, "int8_pertensor", "sq4096")["gflops"]
        fp = baseline(agg, lane, "fp32_omp" if lane == "cpu" else "fp32_cublas", "sq4096")["gflops"]
        ax.axhline(pt, color=COLOR["int8_pertensor"], linestyle="--",
                   label="int8_pertensor (plain loop)" if lane == "cpu" else "int8_pertensor")
        ax.axhline(fp, color=COLOR["fp32"], linestyle=":", label=LABEL["fp32"])
        ax.set_title(f"{lane.upper()}, sq4096, t = 2")
        ax.set_ylabel("GFLOP/s")
        ax.set_yscale("log" if lane == "gpu" else "linear")
        set_p_axis(ax)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    fig.tight_layout()
    fig.text(0.01, -0.12, "CPU: median of 3 processes per point, bars = min to max (only a rough noise estimate). "
             "GPU: one process per point, no range.", fontsize=11, color=INK2)
    save(fig, "F1_gflops_vs_p_sq4096")


def fig_f3(agg):
    """F3: accuracy (rel_err, log x) against throughput, points labelled by p."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for r, shape in enumerate(["sq2048", "sq4096"]):
        for c, lane in enumerate(LANES):
            ax = axes[r][c]
            for fam in ("mixed_sep", "mixed_fused"):
                d = sweep(agg, lane, fam, shape)
                x = d["rel_err"].astype(float)
                range_bars(ax, list(x), d, "gflops", COLOR[fam])
                ax.plot(x, d["gflops"], marker="o", color=COLOR[fam], label=LABEL[fam])
                if fam == "mixed_sep":
                    for xi, yi, p in zip(x, d["gflops"], d["iso_out_pct"]):
                        ax.annotate(f"{p:g}", (xi, yi), textcoords="offset points", xytext=LABEL_OFFSET[p],
                                    fontsize=12, color=INK2)
            pt = baseline(agg, lane, "int8_pertensor", shape)
            ax.plot(float(pt["rel_err"]), pt["gflops"], marker="*", markersize=16, linestyle="none",
                    color=COLOR["int8_pertensor"], label="int8_pertensor")
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())  # the default minor labels overlap
            ax.set_title(f"{lane.upper()}, {shape}, t = 2 (labels: p in %)")
            ax.set_xlabel("relative error vs FP32 (log)")
            ax.set_ylabel("GFLOP/s")
            if r == 0 and c == 0:
                ax.legend(loc="lower left")
    fig.tight_layout()
    fig.text(0.01, -0.01, "CPU: median of 3 processes, bars = min to max. GPU: one process per point.",
             fontsize=11, color=INK2)
    save(fig, "F3_tradeoff")


def fig_f4(agg):
    """F4: stage times of mixed_sep at sq4096, t = 2, stacked per p (each stage is a median over processes)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, lane in zip(axes, LANES):
        d = sweep(agg, lane, "mixed_sep", "sq4096")
        x = xpos(d)
        bottom = [0.0] * len(d)
        for (col, name), color in zip(STAGES, STAGE_COLORS):
            vals = d[col].fillna(0.0).tolist()
            ax.bar(x, vals, bottom=bottom, color=color, edgecolor=SURFACE, linewidth=1.5, label=name, width=0.7)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_title(f"{lane.upper()}, mixed_sep, sq4096, t = 2")
        ax.set_ylabel("time per repetition (ms)")
        set_p_axis(ax)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    fig.tight_layout()
    fig.text(0.01, -0.1, "CPU: each segment is the median over 3 processes. GPU: one process.", fontsize=11, color=INK2)
    save(fig, "F4_stage_times")


def ladder_bars(agg):
    """(lane, label, GFLOP/s median, color key) for the CPU ladder (group C) and the GPU ladder, sq2048, t = 2, p = 5."""
    rows = []
    for label, variant, threads in [("scalar\n1 thread", "mixed_sep_scalar", 1), ("AVX2\n1 thread", "mixed_sep_avx2", 1),
                                    ("AVX2 + OpenMP\n16 threads", "mixed_sep_avx2", 16)]:
        r = sel(agg, lane="cpu", group="C", variant=variant, shape="sq2048", true_out_pct=2.0, iso_out_pct=5.0,
                threads=float(threads))
        rows.append(("cpu", label, r["gflops"], "mixed_sep"))
    rows.append(("gpu", "int8_pertensor", baseline(agg, "gpu", "int8_pertensor", "sq2048")["gflops"], "int8_pertensor"))
    for label, fam in (("mixed,\nseparate", "mixed_sep"), ("mixed,\nfused", "mixed_fused")):
        d = sweep(agg, "gpu", fam, "sq2048")
        rows.append(("gpu", label, d[d["iso_out_pct"] == 5.0].iloc[0]["gflops"], fam))
    return rows


def fig_f6(agg):
    """F6: the ladder at sq2048, t = 2, p = 5. Both panels start at zero. CPU: speedup over the scalar kernel."""
    rows = ladder_bars(agg)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, lane in zip(axes, LANES):
        sel_rows = [r for r in rows if r[0] == lane]
        scale = 1.0 / sel_rows[0][2] if lane == "cpu" else 1.0  # on the CPU the first bar is the scalar kernel
        values = [r[2] * scale for r in sel_rows]
        ax.bar(range(len(sel_rows)), values, color=[COLOR[r[3]] for r in sel_rows], width=0.6)
        for i, v in enumerate(values):
            ax.text(i, v + max(values) * 0.02, f"{v:,.1f}x" if lane == "cpu" else f"{v:,.0f}",
                    ha="center", fontsize=13, color=INK)
        ax.set_xticks(range(len(sel_rows)))
        ax.set_xticklabels([r[1] for r in sel_rows])
        ax.set_ylim(0, max(values) * 1.12)
        ax.set_ylabel("speedup over scalar, 1 thread (x)" if lane == "cpu" else "GFLOP/s")
        ax.set_title(f"{lane.upper()}, sq2048, t = 2, p = 5" + ("" if lane == "cpu" else " (mixed)"))
    fig.tight_layout()
    fig.text(0.01, -0.03, "CPU: median of 3 processes per bar. GPU: one process per bar.", fontsize=11, color=INK2)
    save(fig, "F6_ladder")
