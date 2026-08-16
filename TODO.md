- migrate hard-coded numbers to src/constants.py
- read through the code and understand every line
- get a curriculum.py for moving from stage 0 to stage 4 and determine which stages are necessary
- save checkpoints and results and logs
- have models be able to load the previous checkpoint weights and move to the next stage


read the following
Flat MAPPO & Independent Paradigms

  • The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (Yu et al., 2022) — Essential for defending MAPPO as your baseline.
  • Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge? (Schroeder de Witt et al., 2020) — The basis of the IPPO baseline.

  Counterfactual Credit Assignment (Addressing Causal Dilution)

  • Counterfactual Multi-Agent Policy Gradients (Foerster et al., 2018) — The foundational COMA paper. CO-OP replaces this expectation-based marginalization with dynamic bottom-up skill routing.
  • Multi-Agent Credit Assignment via Causal Graphs (Wang et al., 2024) — The MACCA baseline.

  Hierarchical & Role-Based MARL (Addressing the Sparsity Wall)

  • Data-Efficient Hierarchical Reinforcement Learning (Nachum et al., 2018) — This is the HIRO paper, which serves as the theoretical backbone for your HMAPPO (MAHIRO) script.
  • ROMA: Multi-Agent Reinforcement Learning with Emergent Roles (Wang et al., 2020) — Highly relevant to HEIST's strict role specialization and why continuous routing (E-COOP) outperforms discrete
  role generation.