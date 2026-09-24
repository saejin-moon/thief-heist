#!/usr/bin/env python3
"""
generate_plots.py
Publication-grade vector PDF plots for NeurIPS 2026.

Design system
-------------
* Restrained, colorblind-safe palette: crimson accent for the final method (THIEF),
  purple for ROMA, blues for the ablation lineage, amber for RODE, neutral grays for baselines.
* Legends ALWAYS outside the plotting axes (below), never overlapping data.
* Figure physical size matches the NeurIPS \\textwidth (5.5 in) exactly, so
  fonts render 1:1 at their nominal point sizes in the compiled paper.
"""

import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Global typography: serif (Times-like) to match the NeurIPS text font.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 7.8,
    "lines.linewidth": 1.6,
    "lines.markersize": 5.0,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#3C4043",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linewidth": 0.5,
    "grid.linestyle": "--",
    "grid.color": "#C8CDD2",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
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
STYLE_CONFIG = {
    "thief":  {"label": "THIEF (ours)",      "color": "#B2182B", "ls": "-",  "marker": "o", "lw": 2.2, "ms": 5.8, "z": 10},
    "roma":   {"label": "ROMA",              "color": "#762A83", "ls": "--", "marker": "p", "lw": 1.7, "ms": 5.2, "z": 9},
    "coop":   {"label": "COOP (iter 2)",     "color": "#2166AC", "ls": "-",  "marker": "^", "lw": 1.6, "ms": 4.8, "z": 8},
    "ecoop":  {"label": "E-COOP (iter 3)",   "color": "#4393C3", "ls": "-.", "marker": "s", "lw": 1.5, "ms": 4.8, "z": 7},
    "marc":   {"label": "MARC (iter 1)",     "color": "#92C5DE", "ls": ":",  "marker": "D", "lw": 1.4, "ms": 4.2, "z": 6},
    "rode":   {"label": "RODE",              "color": "#E08214", "ls": "--", "marker": "v", "lw": 1.5, "ms": 4.8, "z": 5},
    "mappo":  {"label": "MAPPO",             "color": "#4D4D4D", "ls": "-",  "marker": "x", "lw": 1.6, "ms": 5.2, "z": 4},
    "hmappo": {"label": "H-MAPPO",           "color": "#878787", "ls": "-.", "marker": "v", "lw": 1.2, "ms": 4.0, "z": 3},
    "coma":   {"label": "COMA",              "color": "#BDBDBD", "ls": ":",  "marker": "+", "lw": 1.2, "ms": 4.5, "z": 2},
}


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


def plot_curriculum_curves(data, out_dir="paper/figures"):
    """Two side-by-side panels: (a) Win rate scaling across Stages 1-4,
    and (b) Relative win rate advantage over MAPPO (%).
    Accentuates the late-stage divergence where THIEF widens its lead (+101.5%).
    Legend placed below the panels.
    """
    os.makedirs(out_dir, exist_ok=True)
    stages = [1, 2, 3, 4]
    stage_labels = ["St. 1\n($17^2$)", "St. 2\n($25^2$)", "St. 3\n($35^2$)", "St. 4\n($50^2$)"]
    algos_order = ["thief", "roma", "coop", "ecoop", "marc", "rode", "mappo"]

    grouped = defaultdict(lambda: defaultdict(list))
    for row in data:
        grouped[row["algo"]][row["stage"]].append(row)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.55))

    # Panel (a): Absolute Win Rate Scaling
    for a in algos_order:
        cfg = STYLE_CONFIG[a]
        wrs, wr_errs = [], []
        for s in stages:
            runs = grouped[a][s]
            vals = [r["win_rate"] * 100.0 for r in runs]
            wrs.append(float(np.mean(vals)))
            wr_errs.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        ax1.errorbar(
            stages, wrs, yerr=wr_errs,
            label=cfg["label"], color=cfg["color"],
            linestyle=cfg["ls"], marker=cfg["marker"],
            linewidth=cfg["lw"], markersize=cfg["ms"],
            capsize=2.0, capthick=0.7, elinewidth=0.7,
            zorder=cfg["z"],
        )

    ax1.set_xticks(stages)
    ax1.set_xticklabels(stage_labels)
    ax1.set_xlabel("Curriculum stage (grid size)", labelpad=2.0)
    ax1.set_ylabel("Win rate (%)", labelpad=2.0)
    ax1.set_ylim(-3, 105)
    ax1.set_yticks([0, 25, 50, 75, 100])
    ax1.set_title("(a) Win Rate Scaling (Stages 1--4)", fontsize=9.2, pad=4)

    # Panel (b): Relative Advantage over MAPPO (%)
    mappo_wrs = {s: float(np.mean([r["win_rate"] * 100.0 for r in grouped["mappo"][s]])) for s in stages}
    for a in ["thief", "roma", "coop", "ecoop", "marc", "rode"]:
        cfg = STYLE_CONFIG[a]
        rel_gains = []
        for s in stages:
            runs = grouped[a][s]
            a_wr = float(np.mean([r["win_rate"] * 100.0 for r in runs]))
            m_wr = mappo_wrs[s]
            rel_gains.append(((a_wr - m_wr) / m_wr) * 100.0)
        ax2.plot(
            stages, rel_gains,
            label=cfg["label"], color=cfg["color"],
            linestyle=cfg["ls"], marker=cfg["marker"],
            linewidth=cfg["lw"], markersize=cfg["ms"],
            zorder=cfg["z"],
        )

    ax2.axhline(0, color="#4D4D4D", linestyle="-", linewidth=1.1, alpha=0.7)
    ax2.set_xticks(stages)
    ax2.set_xticklabels(stage_labels)
    ax2.set_xlabel("Curriculum stage (grid size)", labelpad=2.0)
    ax2.set_ylabel("Win rate gain vs. MAPPO (%)", labelpad=2.0)
    ax2.set_ylim(-28, 115)
    ax2.set_title("(b) Relative Gain over Baseline", fontsize=9.2, pad=4)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        fontsize=7.2,
        columnspacing=1.0,
        handletextpad=0.3,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0.16, 1, 1], w_pad=2.0)
    path = os.path.join(out_dir, "curriculum_combined.pdf")
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"Saved {path}")


def plot_stage3_subtasks(data, out_dir="paper/figures"):
    """Grouped bars for Stage 3 subtasks showing the critical coordination bottleneck.
    Includes THIEF, ROMA, COOP, MAPPO, and H-MAPPO to show the steep contrast
    with hierarchical policy collapse. Data labels above bars for clarity.
    """
    os.makedirs(out_dir, exist_ok=True)
    display_algos = ["thief", "roma", "coop", "mappo", "hmappo"]
    subtasks = ["Scout Tag", "Hacker Hack", "Muscle Neut.", "Extractor Loot"]
    keys = ["scout_tag_rate", "hacker_hack_rate", "muscle_neutralize_rate", "extractor_loot_rate"]

    sub_styles = {
        "thief":  {"label": "THIEF (ours)",  "color": "#B2182B"},
        "roma":   {"label": "ROMA",          "color": "#762A83"},
        "coop":   {"label": "COOP",          "color": "#2166AC"},
        "mappo":  {"label": "MAPPO",         "color": "#4D4D4D"},
        "hmappo": {"label": "H-MAPPO",       "color": "#878787"},
    }

    grouped = defaultdict(lambda: defaultdict(list))
    for row in data:
        grouped[row["algo"]][row["stage"]].append(row)

    fig, ax = plt.subplots(figsize=(5.5, 2.3))
    x = np.arange(len(subtasks))
    width = 0.16

    for i, a in enumerate(display_algos):
        cfg = sub_styles[a]
        runs = grouped[a][3]
        vals = [float(np.mean([r[k] * 100 for r in runs])) if runs else 0.0 for k in keys]
        offset = (i - len(display_algos) / 2 + 0.5) * width
        rects = ax.bar(
            x + offset, vals, width * 0.90,
            label=cfg["label"],
            color=cfg["color"],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        for rect in rects:
            h = rect.get_height()
            if h > 25:
                ax.annotate(f"{h:.0f}\\%",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6.3, color="#202124")

    ax.set_ylabel("Completion rate (%)", labelpad=2.0)
    ax.set_xticks(x)
    ax.set_xticklabels(subtasks, fontsize=9.0)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.55, len(subtasks) - 0.45)
    ax.tick_params(axis="x", which="major", pad=2.0)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=5,
        fontsize=7.8,
        columnspacing=1.2,
        handletextpad=0.3,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0.15, 1, 1])
    path = os.path.join(out_dir, "stage3_subtasks.pdf")
    fig.savefig(path, format="pdf")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    eval_data = load_eval_data()
    if eval_data:
        plot_curriculum_curves(eval_data)
        plot_stage3_subtasks(eval_data)