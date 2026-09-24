# Paper Revision Notes — THIEF in HEIST

## 1. Verdict: contribution type

**THIEF is a composition/systems contribution, not a methods contribution.** Every individual mechanism is traceable to prior work. The defensible novelty is the *targeted spawning loop* (detect interference → isolate failure transitions → spawn specialist → sandbox incubate → hysteresis re-entry) applied to heterogeneous-role MARL. Position the paper that way. If you sell Fisher recombination or MoE routing as novel in themselves, a knowledgeable reviewer will torch the paper in 10 minutes.

---

## 2. Missing citations (critical — must add)

These are the nearest neighbors. A reviewer who knows any of them will flag the paper as under-cited immediately. Every one needs a paragraph in Related Work and a bib entry.

| Key | Paper | Why it matters |
|-----|-------|---------------|
| `matena2021merging` | Matena & Raffel, *Merging Models with Fisher-Weighted Averaging*, NeurIPS 2021 | **Your Prop 1 is literally their result.** You must cite this and frame THIEF as *applying* Fisher merging to inter-expert recombination in an evolving MARL pool, not as inventing it. |
| `yu2020gradient` | Yu et al., *Gradient Surgery for Multi-Task Learning* (PCGrad), NeurIPS 2020 | **Your gradient-conflict trigger uses their exact cosine-similarity diagnostic.** The twist — success/failure partitions of a *single* task as a spawn signal — is genuinely new, but only if you acknowledge the debt. |
| `andrychowicz2017hindsight` | Andrychowicz et al., *Hindsight Experience Replay*, NeurIPS 2017 | HER is central to your macro-horizon auxiliary loss. Cannot omit. |
| `rusu2016progressive` | Rusu et al., *Progressive Neural Networks*, arXiv 2016 | Dynamic capacity growth via lateral columns — the intellectual grandparent of your pool-doubling. |
| `yoon2018lifelong` | Yoon et al., *Lifelong Learning with Dynamically Expandable Networks*, ICLR 2018 | Expandable networks that grow when loss stagnates — directly adjacent to your spawn trigger. |
| `yan2021dynamically` | Yan et al., *Dynamically Expandable Mixture of Experts for Continual Learning* (DEMO), ICML 2021 (verify venue) | The closest prior work to your expert-pool growth idea. Must contrast: DEMO expands on *task* boundaries; THIEF expands on *gradient conflict within a single task*. |
| `khadka2018evolution` | Khadka & Tumer, *Evolution-Guided Policy Gradient in Reinforcement Learning*, NeurIPS 2018 | ERL — evolutionary RL baseline. Your E-COOP comparison is fine, but ERL is the broader literature you need to cite. |
| `jaderberg2017population` | Jaderberg et al., *Population Based Training of Neural Networks*, arXiv 2017 | PBT — your sandbox/cooldown/FIFO queue is a PBT-flavored exploit/explore cycle. Cite it. |
| `bacon2017option` | Bacon et al., *The Option-Critic Architecture*, AAAI 2017 | Options / temporally-extended actions. Your HER macro-horizon segments are option-like; cite for intellectual honesty even if you differ. |
| `kuba2022trust` | Kuba et al., *Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning* (HATRPO/HAPPO), ICLR 2022 | Heterogeneous-agent policy optimization — the MARL baseline family you compete against but do not cite. |
| `wang2020roma` | Wang et al., *ROMA: Multi-Agent Reinforcement Learning with Emergent Roles*, ICLR 2020 | Role-conditioned MARL with emergent specialization. Very close to your value-bidding expert selection story. |
| `wang2021rode` | Wang et al., *RODE: Learning Roles to Decompose Knowledge Sharing in Multi-Agent Reinforcement Learning*, ICLR 2021 | Role decomposition via action-effect prediction — another role-based MARL neighbor. |
| `carroll2019utility` | Carroll et al., *On the Utility of Learning about Humans for Human-AI Coordination*, NeurIPS 2019 | Overcooked — the canonical heterogeneous-cooperation gridworld benchmark you need to acknowledge. |
| `resnick2018pommerman` | Resnick et al., *Pommerman: A Multi-Agent Playground*, arXiv 2018 | Multi-agent competitive/cooperative gridworld with partial observability. |
| `leibo2021scalable` | Leibo et al., *Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot*, ICML 2021 | Multi-agent evaluation benchmark with social dilemmas and cooperation. |
| `matignon2007hysteretic` | Matignon et al., *Hysteretic Q-Learning*, IROS 2007 | You use the word "hysteresis" and have a hysteresis epsilon parameter. Cite the origin, even though your usage is about switching barriers rather than Q-value optimism. |
| `papoudakis2021benchmarking` | Papoudakis et al., *Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks*, NeurIPS Datasets & Benchmarks 2021 | E-PyMARL / Level-Based Foraging — the standard MARL cooperative benchmark suite. |

### Venue uncertainty notes
- `yan2021dynamically` (DEMO): I believe ICML 2021 but **verify before submission**. If uncertain, use arXiv.
- `leibo2021scalable` (Melting Pot): ICML 2021 Datasets & Benchmarks track — verify.

---

## 3. Related Work rewrite plan

Current Section 2 is ~3 paragraphs (MARL, MoE, Evolutionary RL). Expand to **5 paragraphs**:

1. **Cooperative MARL & CTDE** (keep, add HAPPO/HATRPO `kuba2022trust`, ROMA `wang2020roma`, RODE `wang2021rode`). Frame: fixed-capacity CTDE methods fail when role-specific reward densities are temporally mismatched.
2. **Mixture of Experts & Capacity Growth** (keep Shazeer/Switch, add DEMO `yan2021dynamically`, Progressive Nets `rusu2016progressive`, DEN `yoon2018lifelong`). Frame: existing MoE growth is task-driven or architecture-search-driven; THIEF is *deficit-driven* within a single task.
3. **Gradient Conflict & Multi-Task Optimization** (new paragraph, PCGrad `yu2020gradient`). Frame: task-gradient conflict is a known MTL pathology; THIEF repurposes the diagnostic as a *spawn trigger* for policy specialization.
4. **Model Merging & Fisher Geometry** (new paragraph, Fisher Merging `matena2021merging`, EWC `kirkpatrick2017overcoming`). Frame: Fisher-weighted averaging merges models post-hoc; THIEF uses it as a *genetic operator* during online learning.
5. **Population-Based Training & Evolutionary RL** (keep ES/Salimans, add PBT `jaderberg2017population`, ERL `khadka2018evolution`). Frame: THIEF's sandbox/cooldown/FIFO is a PBT-like exploit/explore cycle with mechanistic targeting instead of random mutation.

Also add HER citation `andrychowicz2017hindsight` in the Methods section where macro-horizon hindsight relabeling is described, not in Related Work.

---

## 4. Theory fixes

### Proposition 1 (Fisher recombination)
- **Problem:** The closed-form curvature-optimal recombination is the standard Fisher-merging result from Matena & Raffel 2021.
- **Fix:** Cite `matena2021merging` explicitly. Reframe the proposition as "Adapting Fisher-weighted model merging to online MARL expert recombination" rather than a novel derivation. The contribution is the *application*, not the proof.

### Proposition 2 (Load-balance entropy bound)
- **Problem:** The bound assumes `P_k = f_k` (calibration), which does not hold during actual training — gating probabilities and empirical frequencies diverge because the critic bids are trained while the routing is discrete.
- **Fix:** Add a one-sentence caveat: "Under the idealized calibration assumption `P_k = f_k`, which holds approximately when critic bids are well-calibrated…" Alternatively, reformulate the bound in terms of empirical frequencies only (remove `P_k`) using Jensen-Shannon or simply the Switch-Transformer loss bound, which is what you actually optimize. The current proof is fine as a pedagogical tool if you label the assumption.

---

## 5. Contribution repositioning (title & abstract)

**Current framing problem:** The title says "Targeted Hysteresis-routed Incubated Evolution via Fisher-geometry" — four jargoned mechanisms, all of which are mostly prior work.

**Better framing:**
- **Title option A:** *Deficit-Targeted Expert Spawning for Heterogeneous Multi-Agent Cooperation*
- **Title option B:** *Dynamic Mixture-of-Experts for Heterogeneous MARL via Gradient-Conflict Spawning*
- Either keeps THIEF as the acronym but de-emphasizes the Fisher/hysteresis/incubation jargon in the title.

**Abstract:** Lead with the *problem* (temporally mismatched reward densities across heterogeneous roles cause representation collapse in fixed-capacity CTDE), then the *solution* (spawn specialists targeted at failure transitions via gradient-conflict detection, sandbox them, recombine with Fisher geometry), then the *result* (win rate X vs Y on HEIST). Do not claim to invent MoE routing, Fisher merging, or HER.

---

## 6. Empirical checklist (must run before submission)

The current ablation table is all zeros (smoke-scale runs). This is fatal for a composition paper — the ablations *are* the contribution.

### Minimum viable experiments
| Experiment | Purpose | Budget |
|---|---|---|
| `none` (full THIEF) vs `clone_best` | Does pool recombination beat single-parent cloning? | Full per-stage |
| `none` vs `uniform_recomb` | Does Fisher weighting matter vs uniform average? | Full per-stage |
| `none` vs `no_incubation` | Does sandbox incubation matter? | Full per-stage |
| `none` vs `fixed_schedule` | Does gradient-conflict trigger beat fixed cadence? | Full per-stage |
| `none` vs `no_balance` | Does load-balancing loss prevent monopoly? | Full per-stage |
| `none` vs `no_hysteresis` | Does switching hysteresis matter? | Full per-stage |

### Critical diagnostics to log and report
- Spawn history per seed (already logged in `spawn_history`) — report: number of spawns, parent diversity, cos_sim distribution, deficit sample count.
- Expert usage histogram at stage end (already in results) — report: effective number of experts (exp(entropy)), dormancy rate.
- HER reach rate and relabel count per stage (now logged) — report: average goal-reach %, relabeled transitions per update.
- Stage-wise win-rate curves with std error across seeds (not just final summary).

### Comparison baselines
You have 7 algorithms benchmarked. Make sure the comparison table includes:
- MAPPO, H-MAPPO, COMA, QMIX (or VDN), E-COOP, MARC, and THIEF.
- Report **sample efficiency** (timesteps to 50% WR, 80% WR) not just final WR — your spawning advantage is early specialization.

---

## 7. HEIST environment positioning

HEIST is a new instance of a well-studied class. To claim novelty, emphasize the *specific structural property* it exposes:

- **Temporal reward-density mismatch:** Scout gets dense early rewards (tagging POIs); Extractor gets sparse late rewards (loot + extract). This breaks fixed-capacity CTDE because value functions for early-reward roles dominate gradient updates, starving late-reward roles.
- **Role-conditioned partial observability:** Each role sees different local state (scout sees alarms, hacker sees terminals, etc.), forcing heterogeneous representations.

Cite Overcooked `carroll2019utility`, Pommerman `resnick2018pommerman`, LBF/E-PyMARL `papoudakis2021benchmarking`, Melting Pot `leibo2021scalable` as the benchmark class, then state what HEIST adds.

---

## 8. Nomenclature consistency

HEAD has de-jargoned names ("warmup", "expert", "pruning"). This is good for readability. Keep it, but:
- Add a footnote or glossary table mapping old→new terms for reviewers reading the code.
- In the paper, briefly explain that "warmup" = sandbox incubation, "pruning" = dormancy culling, etc.

---

## 9. Quick fixes in existing text

### `sections/04_method_thief.tex`
- Where Fisher recombination is introduced: add `\cite{matena2021merging}` and clarify that the closed-form is their result.
- Where gradient-conflict trigger is introduced: add `\cite{yu2020gradient}` and frame as "adapting PCGrad's task-conflict diagnostic to success/failure partitions within a single MARL task."
- Where HER macro-horizon is introduced: add `\cite{andrychowicz2017hindsight}`.
- Where hysteresis is introduced: add `\cite{matignon2007hysteretic}` as the origin of the term, then explain your different usage.

### `sections/02_related_work.tex`
- Rewrite per Section 3 above.

### `sections/05_theoretical_analysis.tex`
- Add calibration caveat to Prop 2.
- Re-attribute Prop 1.

---

## 10. BibTeX entries to append to `references.bib`

(These are provided in `references.bib` — append them verbatim.)
