# Level-Based Foraging (LBF) Heterogeneous MARL Benchmark

Trained-from-scratch evaluation on the canonical Level-Based Foraging benchmark (Papoudakis et al., NeurIPS 2021 Datasets & Benchmarks). All methods use identical PPO budgets; final policies are evaluated greedily.

## LBF 8x8 (2p, 3f, Asymmetric Levels) (`Foraging-8x8-2p-3f-v3`)
*2 asymmetric players (levels 1-2), 3 multi-level foods; mixed solo/cooperative loading.*

| Algorithm | Return (±Std) | Food Collected (%) | Steps |
| :--- | :--- | :--- | :--- |
| **THIEF** | 0.211 ± 0.036 | 26.3% ± 4.7% | 49.5 |
| **ROMA** | 0.155 ± 0.023 | 19.3% ± 2.7% | 49.5 |
| **RODE** | 0.036 ± 0.009 | 4.7% ± 1.2% | 50.0 |
| **MAPPO** | 0.144 ± 0.072 | 17.8% ± 8.7% | 49.6 |
| **HMAPPO** | 0.202 ± 0.057 | 24.5% ± 6.2% | 49.7 |
| **COMA** | 0.042 ± 0.033 | 5.4% ± 4.3% | 50.0 |

## LBF 8x8 (3p, 3f, Asymmetric Levels) (`Foraging-8x8-3p-3f-v3`)
*3 asymmetric players (levels [1,2,2] across seeds), 3 multi-level foods up to level 5 requiring whole-team rendezvous.*

| Algorithm | Return (±Std) | Food Collected (%) | Steps |
| :--- | :--- | :--- | :--- |
| **THIEF** | 0.145 ± 0.035 | 21.3% ± 4.0% | 49.8 |
| **ROMA** | 0.276 ± 0.282 | 32.1% ± 27.6% | 46.0 |
| **RODE** | 0.033 ± 0.004 | 4.9% ± 1.1% | 50.0 |
| **MAPPO** | 0.230 ± 0.184 | 29.6% ± 19.2% | 48.3 |
| **HMAPPO** | 0.329 ± 0.308 | 37.9% ± 30.2% | 45.1 |
| **COMA** | 0.007 ± 0.008 | 1.1% ± 1.3% | 50.0 |

## THIEF-on-LBF Component Ablations

### LBF 8x8 (2p, 3f, Asymmetric Levels)

| Variant | Return (±Std) | Food Collected (%) | Spawns | Final Experts |
| :--- | :--- | :--- | :--- | :--- |
| thief | 0.211 ± 0.036 | 26.3% ± 4.7% | 2.0 | 4.0 |

### LBF 8x8 (3p, 3f, Asymmetric Levels)

| Variant | Return (±Std) | Food Collected (%) | Spawns | Final Experts |
| :--- | :--- | :--- | :--- | :--- |
| thief | 0.145 ± 0.035 | 21.3% ± 4.0% | 2.0 | 4.0 |

