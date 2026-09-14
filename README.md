# HEIST & THIEF: Multi-Agent RL Benchmark and Dynamic Mixture-of-Experts

[![Rust Engine](https://img.shields.io/badge/Engine-Native%20Rust%20Rayon-orange.svg)](src/rs)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](src/py)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-red.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**HEIST** (Hierarchical Environment for Interdependent Sequential Tasks) is a multi-agent reinforcement learning benchmark featuring asymmetric agent capabilities, line-of-sight perception, patrolling adversaries, locked security doors, and multi-stage extraction objectives.

**THIEF** is a dynamic Mixture-of-Experts (MoE) algorithm designed for heterogeneous cooperation. It pairs decentralized value-bidding routing with win-rate plateau-triggered expert spawning, Fisher-weighted parameter recombination, isolated warmup on dedicated environments, and switching hysteresis.

The repository includes a multi-threaded Rust simulation engine (~12,000 steps/sec), Python/PyTorch bindings, standard baseline algorithms (MAPPO, H-MAPPO, COOP, E-COOP, MARC, COMA), a 5-stage curriculum training pipeline, and automated evaluation tools.

---

## Environment Mechanics

### Agent Roles

Four asymmetric agents must coordinate to complete mission objectives:

| Role | Symbol | Field of View | Primary Capabilities |
| :--- | :---: | :---: | :--- |
| **Scout** | `S` | Radius 8 | Extended perception. Tags points of interest (cameras, terminal, vault loot) from up to 8 tiles away, projecting directional compass guidance to teammates. |
| **Hacker** | `H` | Radius 3 | Bypasses locked security doors and hacks the security terminal (3 consecutive interaction turns) to disable cameras and unlock the vault. |
| **Muscle** | `M` | Radius 3 | Engages and neutralizes patrolling security guards in close quarters, clearing safe paths for the squad. |
| **Extractor** | `E` | Radius 3 | Infiltrates the unlocked vault, collects the loot payload, and initiates the exfiltration countdown. All agents must then reach extraction. |

### Mission Workflow

1. **Reconnaissance**: Scout spots and tags security terminals, locked doors, and guards through fog of war.
2. **Infiltration**: Muscle neutralizes patrolling guards; Hacker bypasses locked corridor doors.
3. **Terminal Override**: Hacker interacts with the terminal for 3 turns to disable cameras and unlock the vault.
4. **Loot Retrieval**: Extractor navigates to the vault and secures the loot.
5. **Extraction**: All four agents converge on the extraction zone before the countdown expires or alarm maxes out.

### Alarm Dynamics

The squad shares an alarm meter (0 to `ALARM_MAX`):
- **Camera Exposure**: +0.10 / step while in active camera field of view.
- **Active Hacking**: +1.0 / step during terminal interaction.
- **Door Bypass**: +3.0 upon bypassing a locked door.
- **Guard Detection**: +10.0 initial line-of-sight spotting, +1.5 / step continuous tracking.
- **Guard Neutralize**: +5.0 spike upon neutralizing a guard.
- **Extraction Timeout**: +15.0 penalty if the extraction countdown expires.

Episodes terminate in defeat if the alarm reaches capacity or if any agent is caught by a guard.

---

## Algorithms Implemented

| Algorithm | Type | Description |
| :--- | :--- | :--- |
| **THIEF** | Dynamic MoE | Value-bidding Mixture-of-Experts with plateau-triggered spawning, Fisher-weighted parameter recombination, isolated warmup environments, and routing hysteresis. |
| **E-COOP** | Evolutionary MoE | Value-bidding Mixture-of-Experts with periodic parameter recombination into a consolidated policy. |
| **MAPPO** | CTDE Baseline | Multi-Agent PPO with centralized value function and decentralized actor policies. |
| **H-MAPPO** | Hierarchical MARL | Two-level hierarchy with a global Manager setting 2D directional sub-goals and local Workers executing actions. |
| **COOP** | Fixed-Pool MoE | Value-bidding Mixture-of-Experts with a fixed number of candidate experts. |
| **MARC** | Credit Assignment | Monolithic actor-critic with affordance-aware potential-based reward shaping. |
| **COMA** | Counterfactual PG | Counterfactual Multi-Agent Policy Gradients with a centralized critic that marginalizes individual agent actions. |

---

## Curriculum Stages

| Stage | Map Dimensions | Guards | Cameras | Doors | Max Steps | Alarm Max | Key Challenge |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | 11 x 11 | 0 | 0 | 0 | 300 | 100.0 | Basic movement, terminal hacking, loot extraction |
| **1** | 17 x 17 | 1 | 0 | 1 | 400 | 100.0 | Single guard avoidance, locked door bypassing |
| **2** | 25 x 25 | 2 | 1 | 2 | 900 | 125.0 | Camera scan cones, multi-guard patrols, room coordination |
| **3** | 35 x 35 | 3 | 2 | 3 | 2,000 | 150.0 | Labyrinth navigation, long-range credit assignment |
| **4** | 50 x 50 | 4 | 3 | 4 | 5,000 | 175.0 | Full floorplan with active security grid |

---

## Installation & Setup

### Prerequisites

- Python >= 3.10 (Python 3.14 supported)
- Rust toolchain (`cargo`, `rustc` >= 1.75) for native engine compilation
- [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`

### 1. Install Dependencies

Using `uv`:
```bash
uv pip install -e .
```

Or standard pip:
```bash
pip install -e .
```

### 2. Build the Rust Engine (Optional)

The Rust extension compiles automatically on package install via maturin. To build manually:
```bash
cargo build --release
```

---

## Quickstart

### 1. Run Benchmark Pipeline

Train all 7 algorithms across curriculum stages with 10 seeds and parallel evaluation:
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

### 2. Standalone Training

Train a specific algorithm (e.g. THIEF) across curriculum stages:
```bash
PYTHONPATH=src/py uv run python src/py/curriculum.py --algo thief --stages all --seeds 0-2 --rust
```

Or train directly on a single stage:
```bash
PYTHONPATH=src/py uv run python src/py/train_thief.py --stage 2 --timesteps 500000 --rust
```

### 3. Evaluation Suite

Evaluate model checkpoints across curriculum stages in parallel:
```bash
PYTHONPATH=src/py uv run python src/py/eval.py --algos all --stages all --train-seeds 0-9 --episodes 1000 --num-envs 16 --rust
```

### 4. THIEF Micro-Ablation Suite

Train and evaluate the 7 component ablation variants standalone on diagnostic stages (Stages 2 and 4):
```bash
PYTHONPATH=src/py uv run python src/py/ablation.py --seeds 0-2 --episodes 1000 --rust
```

---

## Docker Containerization

A multi-stage `Dockerfile` with NVIDIA CUDA GPU acceleration and CPU fallback is included.

### Build Image
```bash
docker build -t thief-heist:latest .
```

### Run Container
```bash
# Run benchmark with GPU passthrough and volume mounts
docker run --gpus all \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/paper/tables:/app/paper/tables \
    thief-heist:latest

# Or using docker-compose
docker compose up
```

### Publishing Container Images

#### Docker Hub
```bash
docker login
docker tag thief-heist:latest <your-username>/thief-heist:latest
docker push <your-username>/thief-heist:latest
```

#### GitHub Container Registry (GHCR)
```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u <your-username> --password-stdin
docker tag thief-heist:latest ghcr.io/<your-username>/thief-heist:latest
docker push ghcr.io/<your-username>/thief-heist:latest
```

---

## Repository Structure

```text
thief-heist/
├── Cargo.toml                 # Root Rust workspace configuration
├── Dockerfile                 # Multi-stage GPU-accelerated Docker build
├── docker-compose.yml         # Docker benchmark service
├── pyproject.toml             # Python package configuration (maturin backend)
├── script.sh                  # Master reproduction script
├── src/
│   ├── py/                    # Python RL trainers, curriculum, evaluation, and analytics
│   │   ├── ablation.py        # THIEF micro-ablation suite (7 variants)
│   │   ├── ascii.py           # Terminal ASCII visualizer & debugger
│   │   ├── constants.py       # Global environment constants & hyperparameters
│   │   ├── curriculum.py      # Multi-algorithm concurrent curriculum runner
│   │   ├── dataset.py         # Polars + Parquet logging pipeline & report generator
│   │   ├── eval.py            # High-throughput parallel vectorized evaluation suite
│   │   ├── thermal_guard.py   # Hardware thermal protection monitor
│   │   ├── train_thief.py     # THIEF (Mixture-of-Experts with dynamic spawning)
│   │   ├── train_ecoop.py     # E-COOP (Evolutionary MoE baseline)
│   │   ├── train_mappo.py     # MAPPO baseline trainer
│   │   ├── train_hmappo.py    # Hierarchical H-MAPPO baseline
│   │   ├── train_coop.py      # COOP fixed-pool MoE baseline
│   │   ├── train_marc.py      # MARC credit assignment baseline
│   │   ├── train_coma.py      # Counterfactual COMA baseline
│   │   └── vec_env.py         # Zero-copy RustVectorEnv wrapper
│   └── rs/                    # Native Rust high-throughput simulation engine
│       ├── Cargo.toml         # Rust crate configuration (pyo3, rayon, ndarray)
│       └── src/
│           ├── batch.rs       # Multi-threaded parallel batch environment
│           ├── env.rs         # HEIST environment logic & reward shaping
│           ├── grid.rs        # Procedural dungeon room & corridor generator
│           ├── pathfinding.rs # Fast O(1) BFS pathfinder
│           └── vision.rs      # Raycasting, FOV calculations & line-of-sight
├── paper/
│   ├── tables/                # LaTeX tables (ablation.tex, results tables)
│   └── results_master_table.md # Master benchmark table for paper build
├── tests/
│   ├── test_dataset_pipeline.py # Unit tests for Polars/Parquet dataset pipeline
│   └── test_env.py            # Unit tests for RustVectorEnv & models
├── results/                   # Evaluation results and Parquet telemetry
└── README.md
```

---

## Testing

Run unit tests:
```bash
uv run python -m unittest discover -s tests
```

---

## Citation & License

This project is licensed under the MIT License.

```bibtex
@article{moon2026thief,
  title={THIEF: Targeted Hysteresis-routed Incubated Evolution via Fisher-geometry for Heterogeneous Multi-Agent Coordination},
  author={Sae-Jin Moon},
  year={2026}
}
```
