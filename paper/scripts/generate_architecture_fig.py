#!/usr/bin/env python3
"""Generate architecture diagram for THIEF."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

def create_architecture_diagram(out_path="paper/figures/architecture.pdf"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # 5.5 inches wide (NeurIPS textwidth), 2.7 inches tall
    fig, ax = plt.subplots(figsize=(5.5, 2.7), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    # Colors: slate and navy palette
    c_env_bg = "#F4F6F8"
    c_env_border = "#455A64"
    c_env_txt = "#1A252C"
    
    c_exec_bg = "#EBF3FA"
    c_exec_border = "#1565C0"
    c_exec_txt = "#0D47A1"
    
    c_evo_bg = "#FAF0F4"
    c_evo_border = "#880E4F"
    c_evo_txt = "#4A148C"
    
    c_box_bg = "#FFFFFF"
    c_arrow = "#37474F"

    # 1. Left container: HEIST Dec-POMDP and heterogeneous team
    rect_left = FancyBboxPatch((2, 3), 28, 46, boxstyle="round,pad=1.0,rounding_size=2.0",
                               facecolor=c_env_bg, edgecolor=c_env_border, linewidth=1.2, linestyle="-")
    ax.add_patch(rect_left)
    ax.text(16, 46.5, "1. HEIST Environment", ha="center", va="center", fontsize=8.5, weight="bold", color=c_env_txt)
    ax.text(16, 44.0, "Dec-POMDP $(N=4)$", ha="center", va="center", fontsize=7, style="italic", color="#546E7A")

    roles = [
        ("Scout", "Vision r=8; POI Tagging"),
        ("Hacker", "Door Bypass; Terminal Hack"),
        ("Muscle", "Guard Neutralize (20 steps)"),
        ("Extractor", "Loot Securing; Evacuate"),
    ]
    for idx, (role, desc) in enumerate(roles):
        ry = 34.5 - idx * 8.5
        box = FancyBboxPatch((4, ry), 24, 6.8, boxstyle="round,pad=0.5,rounding_size=1.2",
                             facecolor=c_box_bg, edgecolor="#90A4AE", linewidth=0.8)
        ax.add_patch(box)
        ax.text(6, ry + 4.2, role, fontsize=7.5, weight="bold", color="#263238")
        ax.text(6, ry + 1.8, desc, fontsize=6.2, color="#546E7A")

    # 2. Center container: dynamic MoE router and active specialist pool
    rect_center = FancyBboxPatch((34, 3), 32, 46, boxstyle="round,pad=1.0,rounding_size=2.0",
                                 facecolor=c_exec_bg, edgecolor=c_exec_border, linewidth=1.2)
    ax.add_patch(rect_center)
    ax.text(50, 46.5, "2. Decentralized Execution", ha="center", va="center", fontsize=8.5, weight="bold", color=c_exec_txt)
    ax.text(50, 44.0, "Dynamic MoE Specialist Pool", ha="center", va="center", fontsize=7, style="italic", color="#1976D2")

    # Router box
    router_box = FancyBboxPatch((36.5, 30.5), 27, 10.5, boxstyle="round,pad=0.5,rounding_size=1.5",
                                facecolor=c_box_bg, edgecolor=c_exec_border, linewidth=1.0)
    ax.add_patch(router_box)
    ax.text(50, 38.0, "Role-Conditioned Value Bidding", ha="center", va="center", fontsize=7.5, weight="bold", color=c_exec_txt)
    ax.text(50, 35.0, r"$\arg\max_k V_k(s, \mathbf{r}_i, \mathbf{g}_i)$", ha="center", va="center", fontsize=7.5, color="#1565C0")
    ax.text(50, 32.2, r"+ Hysteresis Barrier $\epsilon_{\rm barrier} = 0.05$", ha="center", va="center", fontsize=6.5, color="#37474F")

    # Active Experts Box
    exp_box = FancyBboxPatch((36.5, 6.0), 27, 20.5, boxstyle="round,pad=0.5,rounding_size=1.5",
                             facecolor=c_box_bg, edgecolor="#90CAF9", linewidth=0.9)
    ax.add_patch(exp_box)
    ax.text(50, 24.2, "Active Specialist Pool $\\mathcal{E}$", ha="center", va="center", fontsize=7.5, weight="bold", color="#0D47A1")
    
    # Sub-experts inside
    e_labels = [r"$E_1: \langle \pi_1, V_1 \rangle$", r"$E_2: \langle \pi_2, V_2 \rangle$", r"$E_K: \langle \pi_K, V_K \rangle$"]
    for i, el in enumerate(e_labels):
        ey = 18.0 - i * 5.2
        ebox = FancyBboxPatch((38.5, ey), 23, 4.0, boxstyle="round,pad=0.3,rounding_size=1.0",
                              facecolor="#F5F9FD", edgecolor="#BBDEFB", linewidth=0.7)
        ax.add_patch(ebox)
        ax.text(50, ey + 2.0, el, ha="center", va="center", fontsize=6.8, color="#1565C0")

    # 3. Right container: evolutionary specialist lifecycle (THIEF core)
    rect_right = FancyBboxPatch((70, 3), 28, 46, boxstyle="round,pad=1.0,rounding_size=2.0",
                                facecolor=c_evo_bg, edgecolor=c_evo_border, linewidth=1.2)
    ax.add_patch(rect_right)
    ax.text(84, 46.5, "3. Evolutionary Engine", ha="center", va="center", fontsize=8.5, weight="bold", color=c_evo_txt)
    ax.text(84, 44.0, "Plateau-Triggered Adaptation", ha="center", va="center", fontsize=7, style="italic", color="#AD1457")

    evo_steps = [
        ("Plateau Detector", r"$\Delta W_{100} < 0.03 \rightarrow \text{Trigger}$"),
        ("Deficit Mutation", r"$\mathcal{D}_{\text{fail}} = \{s \mid R - V < 0\}$"),
        ("Fisher Recombination", r"$\theta_{\text{recomb}} \propto \sum w_k F_k \theta_k$"),
        ("Sandbox Incubation", r"$25\%$ Shards reserved ($\tau=10$)"),
    ]
    for idx, (title, math_txt) in enumerate(evo_steps):
        ey = 34.5 - idx * 8.5
        box = FancyBboxPatch((72, ey), 24, 6.8, boxstyle="round,pad=0.5,rounding_size=1.2",
                             facecolor=c_box_bg, edgecolor="#F48FB1", linewidth=0.8)
        ax.add_patch(box)
        ax.text(74, ey + 4.2, title, fontsize=7.2, weight="bold", color="#880E4F")
        ax.text(74, ey + 1.8, math_txt, fontsize=6.0, color="#4A148C")

    # 4. Connecting signal arrows
    # 1. Observation flow from Left to Center Router
    arr1 = FancyArrowPatch((28, 35.5), (36.5, 35.5), arrowstyle="-|>", mutation_scale=10,
                           color=c_arrow, linewidth=1.3)
    ax.add_patch(arr1)
    ax.text(32.25, 37.2, r"$o_i, \mathbf{r}_i, \mathbf{g}_i$", ha="center", va="bottom", fontsize=6.5, color="#37474F")

    # 2. Router dispatch down to Active Experts
    arr2 = FancyArrowPatch((50, 30.5), (50, 26.5), arrowstyle="-|>", mutation_scale=10,
                           color=c_exec_border, linewidth=1.3)
    ax.add_patch(arr2)
    ax.text(53.5, 28.5, r"$k_{\rm chosen}$", ha="left", va="center", fontsize=6.5, color=c_exec_border)

    # 3. Action flow back to Environment
    arr3 = FancyArrowPatch((36.5, 12.0), (28, 12.0), arrowstyle="-|>", mutation_scale=10,
                           color="#2E7D32", linewidth=1.3)
    ax.add_patch(arr3)
    ax.text(32.25, 13.5, r"$a_i \sim \pi_{k^*}$", ha="center", va="bottom", fontsize=6.8, color="#2E7D32")

    # 4. Trajectory return to Evolutionary Plateau Detector (Top loop)
    arr4 = FancyArrowPatch((63.5, 37.5), (72.0, 37.5), arrowstyle="-|>", mutation_scale=10,
                           color="#C2185B", linewidth=1.3, linestyle="-")
    ax.add_patch(arr4)
    ax.text(67.75, 39.0, "Win Rate", ha="center", va="bottom", fontsize=6.5, color="#C2185B")

    # 5. Incubation output returning to Active Pool (Bottom loop)
    arr5 = FancyArrowPatch((72.0, 9.5), (63.5, 9.5), arrowstyle="-|>", mutation_scale=10,
                           color="#0277BD", linewidth=1.3)
    ax.add_patch(arr5)
    ax.text(67.75, 11.0, r"Spawn $E_{\rm child}$", ha="center", va="bottom", fontsize=6.5, color="#0277BD")

    plt.tight_layout(pad=0.2)
    plt.savefig(out_path, format="pdf", dpi=300)
    plt.close()
    print(f"Generated clean architecture diagram: {out_path}")

if __name__ == "__main__":
    create_architecture_diagram()
