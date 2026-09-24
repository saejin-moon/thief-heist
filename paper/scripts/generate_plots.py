#!/usr/bin/env python3
"""
generate_plots.py
Publication-grade vector PDF plots for NeurIPS 2026.

Design system
-------------
* Restrained, colorblind-safe palette: a single crimson accent for the final
  method (THIEF), blue shades for the ablation lineage, neutral grays for
  external baselines.
* Legends ALWAYS outside the plotting axes (below), never overlapping data.
* Figure physical size matches the NeurIPS \\textwidth (5.5 in) exactly, so
  fonts render 1:1 at their nominal point sizes in the compiled paper.
* In-figure titles are omitted (captions carry that information).
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Global typography: serif (Times-like) to match the NeurIPS text font.
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 7.2,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.0,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#3C4043",
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "grid.color": "#C8CDD2",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.color": "#3C4043",
    "ytick.color": "#3C4043",
    "axes.labelcolor": "#202124",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.columnspacing": 1.0,
    "legend.handletextpad": 0.4,
})

# Restrained, colorblind-safe styling.
#   Lineage : one crimson accent + blue family (dark -> light)
#   Baselines: neutral grays, distinct dash patterns
STYLE_CONFIG = {
    "thief":  {"label": "THIEF (ours)",              "color": "#B2182B", "ls": "-",   "marker": "o", "lw": 1.7, "ms": 4.2, "z": 10},
    "ecoop":  {"label": "E-COOP (iter 3)",           "color": "#2166AC", "ls": "-",   "marker": "s", "lw": 1.4, "ms": 3.8, "z": 9},
    "coop":   {"label": "COOP (iter 2)",             "color": "#4393C3", "ls": "-",   "marker": "^", "lw": 1.3, "ms": 3.8, "z": 8},
    "marc":   {"label": "MARC (iter 1)",             "color": "#92C5DE", "ls": "-",   "marker": "D", "lw": 1.2, "ms": 3.4, "z": 7},
    "mappo":  {"label": "MAPPO",                     "color": "#4D4D4D", "ls": "--",  "marker": "x", "lw": 1.2, "ms": 3.8, "z": 6},
    "hmappo": {"label": "H-MAPPO",                   "color": "#878787", "ls": "-.",  "marker": "v", "lw": 1.1, "ms": 3.6, "z": 5},
    "coma":   {"label": "COMA",                      "color": "#BDBDBD", "ls": ":",   "marker": "+", "lw": 1.2, "ms": 4.2, "z": 4},
}

TEXT_DARK = "#202124"


def load_eval_data(filepath=None):
    candidates = [
        filepath,
        "results/eval/eval_summary.json",
        "results/eval/multiseed_full_eval_n1000.json",
        "results/multiseed_summary.json",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            try:
                with open(p, "r") as f:
                    return json.load(f)
            except Exception:  # noqa: BLE001, S112
                continue

    parquet_path = "results/stage_results.parquet"
    if filepath is None and os.path.exists(parquet_path):
        try:
            import polars as pl
            df = pl.read_parquet(parquet_path)
            return df.to_dicts()
        except Exception:  # noqa: BLE001, S110
            pass

    print("Warning: No evaluation data found.")
    return None


def seed_stats(grouped, algo, stage, key, scale=1.0):
    runs = grouped[algo][stage]
    if not runs:
        return None, None
    vals = [r[key] * scale for r in runs]
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean, std


def plot_curriculum_curves(data, out_dir="paper/figures"):
    """Two side-by-side panels (win rate, return); legend below the panels.

    H-MAPPO and COMA are omitted from the curves because both collapse to
    near-0% on Stages 2-4; plotting them compresses the competitive regime
    into a few pixels. Their full numbers appear in Table 2 (main results).
    """
    os.makedirs(out_dir, exist_ok=True)
    algos_order = ["thief", "ecoop", "coop", "marc", "mappo"]
    stages = [0, 1, 2, 3, 4]
    stage_names = ["0\n$11^2$", "1\n$17^2$", "2\n$25^2$", "3\n$35^2$", "4\n$50^2$"]

    grouped = defaultdict(lambda: defaultdict(list))
    for row in data:
        grouped[row["algo"]][row["stage"]].append(row)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.55))

    for a in algos_order:
        cfg = STYLE_CONFIG[a]
        wr_m, wr_s, ret_m, ret_s = [], [], [], []
        for s in stages:
            m, sd = seed_stats(grouped, a, s, "win_rate", 100.0)
            wr_m.append(m if m is not None else 0.0)
            wr_s.append(sd if sd is not None else 0.0)
            m, sd = seed_stats(grouped, a, s, "mean_return")
            ret_m.append(m if m is not None else 0.0)
            ret_s.append(sd if sd is not None else 0.0)

        for ax, means, stds in ((ax1, wr_m, wr_s), (ax2, ret_m, ret_s)):
            ax.errorbar(
                stages, means, yerr=stds,
                label=cfg["label"], color=cfg["color"],
                linestyle=cfg["ls"], marker=cfg["marker"],
                linewidth=cfg["lw"], markersize=cfg["ms"],
                markeredgewidth=0.9, markerfacecolor=cfg["color"],
                capsize=1.6, capthick=0.7, elinewidth=0.7,
                zorder=cfg["z"],
            )

    for ax, ylab in ((ax1, "Win rate (%)"), (ax2, "Episodic return")):
        ax.set_xticks(stages)
        ax.set_xticklabels(stage_names)
        ax.set_xlabel("Curriculum stage (grid size)", labelpad=2.0)
        ax.set_ylabel(ylab, labelpad=1.5)
        ax.set_xlim(-0.35, 4.35)
        ax.tick_params(axis="x", which="major", pad=1.5)

    ax1.set_ylim(-4, 108)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax2.set_ylim(-7, 22)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        fontsize=6.6,
        handlelength=1.7,
    )

    fig.tight_layout(rect=[0, 0.17, 1, 1], w_pad=1.6)
    path = os.path.join(out_dir, "curriculum_combined.pdf")
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"Saved {path}")


def plot_stage3_subtasks(data, out_dir="paper/figures"):
    """Grouped bars for Stage 3 subtasks; legend below the axes, no title."""
    os.makedirs(out_dir, exist_ok=True)
    display_algos = ["thief", "ecoop", "coop", "marc", "mappo"]
    subtasks = ["Scout tag", "Hacker hack", "Muscle neutralize", "Extractor loot"]
    keys = ["scout_tag_rate", "hacker_hack_rate", "muscle_neutralize_rate", "extractor_loot_rate"]

    grouped = defaultdict(lambda: defaultdict(list))
    for row in data:
        grouped[row["algo"]][row["stage"]].append(row)

    fig, ax = plt.subplots(figsize=(5.5, 2.0))
    x = np.arange(len(subtasks))
    width = 0.15

    for i, a in enumerate(display_algos):
        cfg = STYLE_CONFIG[a]
        runs = grouped[a][3]
        vals = [float(np.mean([r[k] * 100 for r in runs])) if runs else 0.0 for k in keys]
        offset = (i - len(display_algos) / 2 + 0.5) * width
        ax.bar(
            x + offset, vals, width * 0.92,
            label=cfg["label"],
            color=cfg["color"],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )

    ax.set_ylabel("Completion rate (%)", labelpad=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(subtasks)
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.55, len(subtasks) - 0.45)
    ax.tick_params(axis="x", which="major", pad=1.5)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=5,
        fontsize=6.8,
    )

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    path = os.path.join(out_dir, "stage3_subtasks.pdf")
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    eval_data = load_eval_data()
    if eval_data:
        plot_curriculum_curves(eval_data)
        plot_stage3_subtasks(eval_data)