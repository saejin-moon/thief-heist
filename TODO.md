# THIEF Compute Additions & Roadmap (`TODO.md`)

This guide outlines the step-by-step procedures for running **Point B** compute jobs on another workstation or GPU cluster:
1. **B.1: Finishing Seeds 3–9 for Table 5 Micro-Ablations** (unifying all 7 variants to $N=10$ seeds).
2. **B.2: Implementing the 100% Decentralized Local-Observation Router Ablation** (eliminating global state $s$ during test-time routing).

---

## 1. Machine Setup & Environment Preparation

Ensure the target machine has an NVIDIA GPU with CUDA drivers, Rust, and `uv` installed.

```bash
# 1. Clone repository and navigate to root
git clone <repo_url> thief-heist
cd thief-heist

# 2. Install uv (if not present) and Rust toolchain
curl -LsSf https://astral.sh/uv/install.sh | sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# 3. Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv sync

# 4. Compile the high-throughput native Rust simulation engine
uv run maturin develop --release
```

---

## 2. Point B.1: Complete Remaining 7 Seeds for Table 5 Micro-Ablations

### Objective
Currently, the top 4 rows of Table 5 (`none`, `no_incubation`, `uniform_recomb`, `clone_best`) have 10 seeds ($N=10$, seeds 0–9).  
The bottom 3 rows (`no_balance`, `no_hysteresis`, `fixed_schedule`) only have 3 seeds ($N=3$, seeds 0–2).  
Running seeds 3–9 across diagnostic stages 2 and 4 (21 short runs total) shrinks standard errors and makes the micro-ablation table 100% uniform.

### Execution Command

Run the ablation runner with `--seeds 3-9` over the three target variants and diagnostic stages:

```bash
# Execute standalone training and evaluation (1000 held-out episodes per seed/stage)
PYTHONPATH=src/py uv run python src/py/ablation.py \
    --variants no_balance,no_hysteresis,fixed_schedule \
    --stages 2,4 \
    --seeds 3-9 \
    --rust \
    --episodes 1000 \
    --num-envs 64
```

*Note:* Using `--num-envs 64` with the Rust core sustains $\sim 12,000$ steps/sec, allowing each seed/stage to finish in $\sim 2$--$3$ minutes on an RTX 3080/4080 GPU.

### Table & Paper Re-generation

Once all seed logs and checkpoints populate into `results/ablation/` and `results/eval/`:

```bash
# 1. Regenerate paper/tables/ablation.tex
uv run python paper/scripts/generate_ablation_table.py --episodes 1000

# 2. Verify Table 5 in LaTeX
cat paper/tables/ablation.tex

# 3. Recompile the paper to verify
cd paper && tectonic --keep-intermediates main.tex
```

Verify that all rows in `tab:ablations` now aggregate $N=10$ seeds, and update the note in `paper/sections/10_appendix.tex` to remove the 3-seed disclaimer.

---

## 3. Point B.2: Implement 100% Decentralized Local-Observation Routing Ablation

### Objective
In standard THIEF, the critic bids $V_j(s, \mathbf{r}_i, \mathbf{g}_i)$ evaluate the 2,550-dimensional global facility state $s$ (orchestrated team dispatch).  
To bullet-proof the paper for main-track ICML/NeurIPS against strict Dec-POMDP purists, we evaluate a **decentralized student gating network** $g_\psi(o_i, \mathbf{r}_i, \mathbf{g}_i)$ distilled or trained strictly on each agent's egocentric $7 \times 7$ observation $o_i$. At test time, routing requires **zero** access to global state $s$.

---

### Step-by-Step Implementation

#### Step 1: Add `LocalGatingNetwork` in `src/py/train_thief.py`
Define a lightweight student router that maps local agent observations to expert logits:

```python
class LocalGatingNetwork(nn.Module):
    """
    Decentralized Gating Network.
    Evaluates only local observation o_i, role r_i, and mission compass g_i.
    Input dim: (7 * 7) + 4 (role) + 2 (compass) = 55 dimensions.
    Output: K-dimensional logits for expert dispatch.
    """
    def __init__(self, num_experts=THIEF_INITIAL_EXPERTS):
        super().__init__()
        in_dim = (OBSERVATION_SIZE[0] * OBSERVATION_SIZE[1]) + N_AGENTS + GOAL_VECTOR_DIM
        self.network = nn.Sequential(
            layer_init(nn.Linear(in_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_experts), std=0.01),
        )

    def forward(self, x_local):
        return self.network(x_local)
```

#### Step 2: Add Distillation Loss to the Optimization Step in `train_thief.py`
During centralized training rollouts, optimize $g_\psi$ to mimic the teacher's critic value bidding:

```python
# Teacher soft probabilities from critic value bids
teacher_probs = torch.softmax(bids / tau_gate, dim=-1)

# Student local logits from egocentric observation
x_local = torch.cat([obs_flat, role_flat, goal_flat], dim=1)
student_logits = local_gate(x_local)

# Cross-entropy / KL distillation loss
loss_gate_distill = F.cross_entropy(student_logits, teacher_probs)
total_loss = loss_ppo + (alpha_bal * loss_balance) + (0.05 * loss_gate_distill)
```

#### Step 3: Support Decentralized Dispatch in `get_action_and_value`
When `decentralized_eval=True`, route via the student gate instead of centralized critic bids:

```python
if decentralized_eval:
    x_local = torch.cat([obs, role, goal], dim=1)
    gate_logits = self.local_gate(x_local)[:, candidate_experts]
    best_candidate_idx = gate_logits.argmax(dim=-1)
else:
    # Standard centralized critic bidding
    bids = torch.stack([self.experts[k].critic(x_critic) for k in candidate_experts], dim=-1).squeeze(1)
    best_candidate_idx = bids.argmax(dim=-1)
```

#### Step 4: Register Variant in `src/py/ablation.py`
Add `"local_router"` to `ABLATION_VARIANTS` and `VARIANT_META` in `src/py/ablation.py`:

```python
ABLATION_VARIANTS = [
    "none",
    "no_incubation",
    "uniform_recomb",
    "clone_best",
    "no_balance",
    "no_hysteresis",
    "fixed_schedule",
    "local_router",
]

VARIANT_META["local_router"] = (
    "local-router",
    r"decentralized $g_\psi(o_i)$ (zero global state at eval)"
)
```

#### Step 5: Train and Evaluate the Variant
Execute training across diagnostic stages 2 and 4 with 10 random seeds:

```bash
PYTHONPATH=src/py uv run python src/py/ablation.py \
    --variants local_router \
    --stages 2,4 \
    --seeds 0-9 \
    --rust \
    --episodes 1000 \
    --num-envs 64
```

---

## 4. Updating Paper Artifacts

After running Point B.1 and/or B.2:

```bash
# 1. Regenerate ablation tables
uv run python paper/scripts/generate_ablation_table.py --episodes 1000

# 2. Regenerate performance plots
uv run python paper/scripts/generate_plots.py

# 3. Build the full NeurIPS paper with Tectonic
cd paper
tectonic --keep-intermediates main.tex
```

### Documenting Results in `paper/sections/08_discussion.tex`
Add a dedicated discussion paragraph under component micro-ablations:
```latex
\paragraph{Decentralized Edge Routing via Local Distillation.}
When routing is restricted to purely decentralized egocentric observations 
$g_\psi(o_i, \mathbf{r}_i, \mathbf{g}_i)$ via online critic distillation, 
THIEF achieves XX.X\% win rate on Stage 4 (compared to 13.7\% under centralized 
dispatch), confirming that specialist behavioral commitments can be executed 
fully decentralized without access to global simulator telemetry.
```

---
*Created for multi-machine compute dispatch and publication refinement.*
