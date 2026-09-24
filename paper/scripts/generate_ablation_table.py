#!/usr/bin/env python3
"""Generate LaTeX micro-ablation table for THIEF."""


import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

VARIANT_META = {
    "none":           ("Full THIEF", "standard framework"),
    "no_incubation":  ("w/o incubation", r"$\tau_{\mathrm{iso}}=0$ (skip sandbox)"),
    "uniform_recomb": ("w/o Fisher", "uniform recombination"),
    "clone_best":     ("clone-best", r"best parent $+$ noise"),
    "no_balance":     (r"w/o balance", r"$\alpha_{\mathrm{bal}}=0$"),
    "no_hysteresis":  ("w/o hysteresis", r"$\epsilon=0$ (no barrier)"),
    "fixed_schedule": ("fixed-cadence", "periodic updates"),
}

VARIANT_ORDER = [
    "none",
    "no_incubation",
    "uniform_recomb",
    "clone_best",
    "no_balance",
    "no_hysteresis",
    "fixed_schedule",
]

STAGE_LABELS = {2: r"St.~2 ($25^2$)", 4: r"St.~4 ($50^2$)"}


def load_eval(variant, episodes):
    path = f"results/eval/ablation_{variant}_eval_n{episodes}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_mechanism_metrics(variant):
    """Aggregate mechanism telemetry from per-run results.json files.

    Routing entropy H(f) is only meaningful when at least two experts were
    spawned (with a single active expert there is no routing distribution and
    H(f) is trivially 0, not evidence of routing collapse). We therefore
    report the mean H(f) over runs with active_experts >= 2, plus the
    multi-expert rate (fraction of runs with active_experts >= 2).
    """
    rows = sorted(glob.glob(f"results/ablation/{variant}/thief/seed_*/stage_*/results.json"))
    spawns, switches, entropies, ginis, actives = [], [], [], [], []
    for path in rows:
        with open(path) as f:
            r = json.load(f)
        spawns.append(r.get("total_spawns", 0))
        actives.append(r.get("active_experts", 1))
        ent = r.get("routing_entropy", r.get("routing_entropy_nats"))
        gini = r.get("gini_expert_imbalance")
        if r.get("active_experts", 1) >= 2:
            if ent is not None and ent == ent:  # skip NaN
                entropies.append(ent)
            if gini is not None and gini == gini:
                ginis.append(gini)
        sw = r.get("expert_switch_rate")
        if sw is not None:
            switches.append(sw * 100.0)
    n = len(actives)
    return {
        "spawns": float(np.mean(spawns)) if spawns else None,
        "switch_rate": float(np.mean(switches)) if switches else None,
        "entropy": float(np.mean(entropies)) if entropies else None,
        "gini": float(np.mean(ginis)) if ginis else None,
        "multi_expert_rate": (100.0 * sum(1 for a in actives if a >= 2) / n) if n else None,
        "n_runs": n,
    }


def wr_cell(mean, std, best_mean):
    if mean is None:
        return "--"
    text = f"{mean:.1f}\\tiny$\\pm${std:.1f}\\%"
    if best_mean is not None and abs(mean - best_mean) < 0.05:
        text = f"\\textbf{{{text}}}"
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1000)
    args = ap.parse_args()

    # Load evaluation results
    raw = {}
    for variant in VARIANT_ORDER:
        data = load_eval(variant, args.episodes)
        if data:
            raw[variant] = data
        else:
            # Fall back to per-run results.json artifacts (final post-train eval).
            fallback = []
            for path in sorted(glob.glob(f"results/ablation/{variant}/thief/seed_*/stage_*/results.json")):
                with open(path) as f:
                    r = json.load(f)
                fallback.append({"stage": r.get("stage"), "win_rate": r.get("win_rate", 0.0), "seed": r.get("seed")})
            if fallback:
                raw[variant] = fallback
                print(f"[info] '{variant}': no eval JSON, using {len(fallback)} per-run results.json artifacts")
            else:
                print(f"[warn] no eval data for '{variant}' -- row will be omitted")
    if not raw:
        print("No ablation evaluation data found. Run paper/scripts/run_ablation_suite.sh first.")
        return

    stages = sorted({row["stage"] for data in raw.values() for row in data})

    # Aggregate win rates over seeds
    agg = defaultdict(dict)
    for variant, data in raw.items():
        by_stage = defaultdict(list)
        for row in data:
            by_stage[row["stage"]].append(row["win_rate"] * 100.0)
        for stage, vals in by_stage.items():
            agg[variant][stage] = (
                float(np.mean(vals)),
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            )

    best = {}
    for stage in stages:
        candidates = [agg[v][stage][0] for v in raw if stage in agg[v]]
        best[stage] = max(candidates) if candidates else None

    # Emit table
    lines = [
        "% Component micro-ablations of THIEF.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Component micro-ablations of THIEF, trained standalone (fresh initialization, no curriculum transfer) on the diagnostic stages with identical per-stage budgets, then evaluated over $N=1,\\!000$ held-out episodes per seed. $H(f)$ is the empirical dispatch routing entropy (nats) over non-dormant experts, aggregated only over runs in which $\\geq 2$ experts were spawned: with a single active expert there is no routing decision and $H(f)$ is trivially $0$, which is not a routing collapse. The fraction of runs reaching $\\geq 2$ experts is reported as K$\\geq$2 (\\%). Mechanism telemetry is averaged across seeds and stages. Bold denotes the best held-out win rate per stage.}",
        "\\label{tab:ablations}",
        "\\vspace{0.04in}",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\renewcommand{\\arraystretch}{1.06}",
        "\\begin{tabular}{l l cc cccc}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{\\textbf{Held-out win rate (\\%)}} & \\multicolumn{4}{c}{\\textbf{Mechanism telemetry}} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-8}",
        "\\textbf{Variant} & \\textbf{Ablated component} & "
        + " & ".join(STAGE_LABELS.get(s, f"Stage {s}") for s in stages)
        + " & \\textbf{Spawns} & \\textbf{Switch} & \\textbf{K$\\geq$2} & \\textbf{$H(f)$} \\\\",
        " & & & & & (\\%) & (\\%) & (nats) \\\\",
        "\\midrule",
    ]

    for variant in VARIANT_ORDER:
        if variant not in raw:
            continue
        name, desc = VARIANT_META[variant]
        cells = [wr_cell(*agg[variant].get(s, (None, None)), best.get(s)) for s in stages]

        mech = load_mechanism_metrics(variant)
        sp = f"{mech['spawns']:.1f}" if mech["spawns"] is not None else "--"
        sw = f"{mech['switch_rate']:.1f}" if mech["switch_rate"] is not None else "--"
        en = f"{mech['entropy']:.2f}" if mech["entropy"] is not None else "--"
        me = f"{mech['multi_expert_rate']:.0f}" if mech.get("multi_expert_rate") is not None else "--"

        lines.append(f"{name} & {desc} & " + " & ".join(cells) + f" & {sp} & {sw} & {me} & {en} \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])

    os.makedirs("paper/tables", exist_ok=True)
    with open("paper/tables/ablation.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Wrote paper/tables/ablation.tex")


if __name__ == "__main__":
    main()