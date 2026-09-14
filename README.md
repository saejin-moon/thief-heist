# HEIST: Heterogeneous Evolutionary & Multi-Agent Reinforcement Learning Benchmark

[![Rust Engine](https://img.shields.io/badge/Engine-Native%20Rust%20Rayon-orange.svg)](src/rs)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](src/py)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-red.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**HEIST** is a high-throughput multi-agent cooperative environment and algorithmic benchmark designed to study heterogeneous coordination, spatial credit assignment, and evolutionary mixture-of-experts (MoE) architectures in complex, partially observable gridworlds.

---

## 🎮 The HEIST Environment

Four specialized agents with distinct capabilities coordinate to infiltrate procedurally generated facilities, evade/neutralize dynamic security guards, bypass locked security doors, hack security terminals, secure vault loot, and execute a synchronized extraction.

```
╔════════════════════════════════════════════════════════════════════╗
║ HEIST DEBUGGER │ THIEF  │ Stg 3 (35x35) │ Step: 120/2000 │ RUST+CUDA     ║
║ Alarm: [█████░░░░░░░░░░░░░░░░░░░]   32.4 / 150.0    Seed: 42       ║
║ Terminal: [✓] Hacked │ Loot: [✓] Secured │ Escape: [✓] Triggered (780s)  ║
╠════════════════════════════════════════════════════════════════════╣
  ███████████████████████████████████
  ███···············█████████████████
  ███···S·X·········███···········███
  ███···············███·C·········███
  ███···············███·····D·······█
  ███···$···········███·······g···███
  ███·········H·····███████·······███
  ███████D█████████████···········███
  ███·················█·······G···███
  ███·····T···········D···········███
  ███·················█·····M·····███
  ███████████████████████████████████
╠════════════════════════════════════════════════════════════════════╣
  Scout     pos: (2, 6)   action: WAIT      reward: +0.200
  Hacker    pos: (6, 12)  action: UP        reward: +0.200
  Muscle    pos: (10, 26) action: LEFT      reward: +0.200
  Extractor pos: (2, 8)   action: WAIT      reward: +0.200
╚════════════════════════════════════════════════════════════════════╝
  Total Episode Return (Per-Agent Avg): +17.480
```

### Roles & Responsibilities

| Role | Symbol | Mechanics & Tactical Capabilities |
| :--- | :---: | :--- |
| **Scout** | `S` | Extended field of view (radius 8), stand-off line-of-sight tagging of cameras, terminals, and loot caches. |
| **Hacker** | `H` | Disables cameras, hacks terminal vault locks, opens locked security doors. |
| **Muscle** | `M` | Engages and temporarily neutralizes patrolling security guards within standoff range. |
| **Extractor** | `E` | Infiltrates the unlocked vault, secures the loot payload, and triggers extraction countdown. |

---

## 🧠 Algorithms Implemented

| Algorithm | Paradigm | Description |
| :--- | :--- | :--- |
| **THIEF** | Evolutionary MoE | **Targeted Hysteresis-routed Incubated Evolution via Fisher-geometry.** Spawns task-specialized expert policies on empirical failure sub-manifolds, incubates them in dedicated sandbox environments, applies load-balancing regularization, and maintains an open-ended specialist pool with dormancy freezing. |
| **E-COOP** | Evolutionary MoE | **Evolutionary Cooperative Multi-Agent PPO.** Uses all-pool Fisher Information Matrix parameter recombination to mutate champion policies into an elite consolidated policy. |
| **MAPPO** | CTDE Baseline | Multi-Agent PPO with centralized state value function and decentralized actor policies. |
| **H-MAPPO** | Hierarchical MARL | Two-level hierarchy with a global Manager setting 2D directional sub-goals and local Workers executing spatial actions. |
| **COOP** | Fixed-Pool MoE | Competitive value-bidding Mixture-of-Experts with a fixed number of specialists. |
| **MARC** | Credit Assignment | Affordance-aware macro causal credit assignment penalizing actions that induce catastrophic alarm bleed. |
| **COMA** | Counterfactual PG | Counterfactual Multi-Agent Policy Gradients with a centralized critic that marginalizes individual agent actions. |

---

## ⚡ Architecture: Hybrid Rust + CUDA Engine

* **Native Rust Simulation Core (`src/rs`):** Parallel procedural map generation, line-of-sight raycasting, BFS pathfinding, guard state machines, and reward calculations implemented in Rust with Rayon multi-threading (>10,000 FPS).
* **PyTorch GPU Pipeline (`src/py`):** Zero-copy contiguous memory transfers between the compiled PyO3 native shared library (`heist_core_rs`) and CUDA tensors.
* **Polars + Parquet Analytics (`src/py/dataset.py`):** High-throughput lock-protected telemetry engine recording sub-step training curves, per-episode evaluation milestones, and MoE spawn histories into columnar Apache Parquet files.

---

## 🗺️ Curriculum Stages

| Stage | Map Dimensions | Guards | Cameras | Doors | Max Steps | Alarm Max | Tactical Challenge |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | 11 × 11 | 0 | 0 | 0 | 300 | 100.0 | Basic movement, terminal hacking, loot extraction |
| **1** | 17 × 17 | 1 | 0 | 1 | 400 | 100.0 | Guard avoidance, locked door bypassing |
| **2** | 25 × 25 | 2 | 1 | 2 | 900 | 125.0 | Camera cones, multi-guard patrol zones, room coordination |
| **3** | 35 × 35 | 3 | 2 | 3 | 2,000 | 150.0 | Labyrinth navigation, long-range credit assignment |
| **4** | 50 × 50 | 4 | 3 | 4 | 5,000 | 175.0 | Full Master Challenge (large-scale spatial coordination) |

---

## 🚀 Quickstart & Installation

### Option 1: Standard Python / Pip Installation

Clone the repository and install with `pip` or [uv](https://github.com/astral-sh/uv). The Rust extension is compiled automatically via [maturin](https://github.com/PyO3/maturin):

```bash
git clone https://github.com/saejin-moon/thief-heist.git
cd thief-heist

# Using uv (fastest)
uv pip install -e .

# Or using standard pip
pip install -e .
```

### Option 2: Pure Rust / Cargo Build

If working directly on the simulation core:

```bash
cargo build --release
```

---

## 🐳 Docker Containerization & Execution

The repository includes a production multi-stage `Dockerfile` with full NVIDIA CUDA GPU acceleration and CPU fallback.

### 1. Build Container Image Locally
```bash
docker build -t thief-heist:latest .
```

### 2. Run Benchmark with Docker
```bash
# Run full benchmark with GPU passthrough and persist results
docker run --gpus all -v $(pwd)/results:/app/results -v $(pwd)/paper/tables:/app/paper/tables thief-heist:latest

# Or using docker-compose
docker compose up
```

### 3. Publishing Container Images (For Researchers & Collaborators)

Publishing a pre-built container image allows external reviewers, employers, and collaborators to reproduce paper benchmarks with a single command without installing Rust, CUDA, or Python environments.

#### Publishing to Docker Hub:
```bash
# Log in to your Docker Hub account
docker login

# Tag the image with your Docker Hub username
docker tag thief-heist:latest <your-username>/thief-heist:latest

# Push the image to Docker Hub
docker push <your-username>/thief-heist:latest
```

#### Publishing to GitHub Container Registry (GHCR):
```bash
# Authenticate with your GitHub Personal Access Token (PAT)
echo $GITHUB_TOKEN | docker login ghcr.io -u <your-username> --password-stdin

# Tag and push to GHCR
docker tag thief-heist:latest ghcr.io/<your-username>/thief-heist:latest
docker push ghcr.io/<your-username>/thief-heist:latest
```

---

## 🔬 Benchmark Reproduction Pipeline

### 1-Command Master Benchmark
Run all 7 algorithms in priority order (`thief` → `ecoop` → `mappo` → `hmappo` → `coop` → `marc` → `coma`) with dynamic 2-algorithm concurrency, 10 seeds (0-9 inclusive), 1,000 evaluation episodes per seed, and THIEF micro-ablations:

```bash
bash script.sh
```

Or invoke directly via Python:

```bash
PYTHONPATH=src/py uv run python src/py/curriculum.py \
    --algos all \
    --concurrent-algos 2 \
    --seeds 0-9 \
    --rust \
    --eval \
    --eval-episodes 1000 \
    --ablations
```

### Standalone Algorithm Training
```bash
# Train THIEF across all stages with native Rust engine on GPU
PYTHONPATH=src/py uv run python src/py/curriculum.py --algo thief --stages all --seeds 0-2 --rust

# Train specific subset with custom concurrency
PYTHONPATH=src/py uv run python src/py/curriculum.py --algos thief,ecoop --concurrent-algos 2 --seeds 0-4 --rust
```

### Parallel Evaluation Suite
Evaluate trained checkpoints with high-throughput parallel rollouts across 1,000 episodes per seed (rigorously proven for standard error $\le \pm 0.5\%$):

```bash
PYTHONPATH=src/py uv run python src/py/eval.py --algos all --stages all --train-seeds 0-9 --episodes 1000 --num-envs 16 --rust
```

### THIEF Component Micro-Ablation Suite
Train and evaluate all 7 ablation variants standalone on diagnostic stages (Stages 2 and 4), generating `paper/tables/ablation.tex`:

```bash
PYTHONPATH=src/py uv run python src/py/ablation.py --seeds 0-2 --episodes 1000 --rust
```

### Interactive Terminal Visualizer
Step through live episodes or trained checkpoints in rich ANSI color:

```bash
# Step through a trained Stage 3 THIEF checkpoint (GPU + Rust accelerated)
PYTHONPATH=src/py uv run python src/py/ascii.py --algo thief --stage 3 --checkpoint results/thief/seed_0/stage_3/model.pt --delay 0.05 --seed 42
```

---

## 📁 Repository Structure

```
thief-heist/
├── Cargo.toml                 # Root Rust workspace definition
├── Dockerfile                 # Multi-stage GPU-accelerated Docker build
├── docker-compose.yml         # 1-click Docker benchmark service
├── pyproject.toml             # Python package configuration (maturin backend)
├── script.sh                  # 1-command master reproduction pipeline
├── src/
│   ├── py/                    # Python RL trainers, curriculum, evaluation, and analytics
│   │   ├── ablation.py        # THIEF micro-ablation suite (7 variants, standalone training)
│   │   ├── ascii.py           # Terminal ASCII visualizer & live debugger
│   │   ├── constants.py       # Global environment constants & hyperparameters
│   │   ├── curriculum.py      # Multi-algorithm concurrent curriculum runner
│   │   ├── dataset.py         # Polars + Parquet logging pipeline & LLM report generator
│   │   ├── eval.py            # High-throughput parallel vectorized evaluation suite
│   │   ├── thermal_guard.py   # Hardware thermal protection monitor
│   │   ├── train_thief.py     # THIEF (Targeted Hysteresis-routed Evolution)
│   │   ├── train_ecoop.py     # E-COOP (Evolutionary MoE Cannibalization)
│   │   ├── train_mappo.py     # MAPPO baseline trainer
│   │   ├── train_hmappo.py    # Hierarchical H-MAPPO baseline
│   │   ├── train_coop.py      # COOP fixed-pool MoE baseline
│   │   ├── train_marc.py      # MARC credit assignment baseline
│   │   ├── train_coma.py      # Counterfactual COMA baseline
│   │   └── vec_env.py         # Zero-copy RustVectorEnv wrapper
│   └── rs/                    # Native Rust high-throughput simulation engine
│       ├── Cargo.toml         # Rust crate configuration (pyo3 + rayon + ndarray)
│       └── src/
│           ├── batch.rs       # Multi-threaded parallel batch environment
│           ├── env.rs         # Single HEIST environment logic & reward shaping
│           ├── grid.rs        # Procedural dungeon room & corridor generator
│           ├── pathfinding.rs # Fast O(1) BFS pathfinder
│           └── vision.rs      # Raycasting, FOV calculations & line-of-sight
├── paper/
│   ├── tables/                # LaTeX tables (ablation.tex, results tables)
│   └── results_master_table.md # Markdown master benchmark table for paper build
├── tests/
│   ├── test_dataset_pipeline.py # Unit tests for Polars/Parquet dataset pipeline
│   └── test_env.py            # Unit tests for RustVectorEnv & models
├── results/                   # Evaluation results, Parquet telemetry, and JSON summaries
└── README.md
```

---

## 🧪 Testing

Run the automated pytest test suite:

```bash
uv run pytest
```

---

## 📜 Citation & License

This project is licensed under the MIT License.
If you use this benchmark or algorithms in your research, please cite:

```bibtex
@article{moon2025thief,
  title={THIEF: Targeted Hysteresis-routed Incubated Evolution via Fisher-geometry for Heterogeneous Multi-Agent Coordination},
  author={Sae-Jin Moon},
  year={2025}
}
```
