import os
import re


def process_file(filepath, replacements, add_logging=False):
    if not os.path.exists(filepath):
        return
    with open(filepath, "r") as f:
        content = f.read()

    for pattern, rep in replacements:
        content = re.sub(pattern, rep, content, flags=re.DOTALL)

    if add_logging and "import logging" not in content:
        content = "import logging\n" + content

    with open(filepath, "w") as f:
        f.write(content)


# 1. MARC
marc_reps = [
    (
        r'"""\s*MARC.*?"""',
        '"""\nMarginal Action Retroactive Credit (MARC) trainer.\n\nThis script executes the MARC algorithm. The algorithm applies localized credit for affordance changes. It also applies a multiplicative global alarm penalty. Success masking isolates enablers from downstream failures. Retroactive advantages propagate backward through the trajectory.\n"""',
    ),
    (
        r"# Structural Affordance Detection for MARC Micro Credit.*?# If an agent interacted and the total mask volume increased, they unlocked an affordance!",
        "# The team unlocked an affordance if an agent interacted and the mask volume increased.",
    ),
    (
        r"# Determine trajectory win states.*?dim=0\)",
        "# Trajectory win states derive directly from the environment signals.",
    ),
    (
        r"# Macro Weighting \(Omega_t\): Alarm scaling and Outcome factor",
        "# Apply alarm scaling and outcome factors.",
    ),
    (
        r"# Binary Success Masking \(Shielding\):\s*# Protect upstream enablers \(affordance > 0\) from downstream incompetence \(loss\)",
        "# Shielding protects upstream enablers from downstream failures.",
    ),
    (
        r"# Retroactive Causal Trace Propagation",
        "# Retroactive causal trace propagation.",
    ),
]
process_file("src/train_marc.py", marc_reps)

# 2. HMAPPO
hmappo_reps = [
    (
        r'"""\s*MAHIRO \(Multi-Agent HIRO\) Architecture.*?"""',
        '"""\nMAHIRO Architecture.\n\nThe manager outputs a spatial goal. The worker outputs a discrete action based on the observation and the manager goal.\n"""',
    ),
    (
        r"# --- MANAGER LOGIC \(Every MACRO_STEP\) ---",
        "# Manager steps evaluate global goals.",
    ),
    (
        r"# INTRINSIC REWARD CALCULATION for Worker \(L2 distance to goal\)\s*# Goal is transformed into spatial coords to check distance",
        "# Calculate the intrinsic reward using the L2 distance between the agent position and the spatial goal.",
    ),
]
process_file("src/train_hmappo.py", hmappo_reps)

# 3. ECOOP
ecoop_reps = [
    (
        r'"""\s*E-COOP: Evolutionary Confidence-Oriented Option Pool.*?"""',
        '"""\nEvolutionary Confidence-Oriented Option Pool (E-COOP).\n\nThe network maintains a pool of base networks. Agents route to the expert with the highest predicted value.\n"""',
    ),
    (r"# 1\. Routing Logic", "# Route agent execution."),
    (
        r"# Hybrid Routing Hysteresis: prevent routing chatter",
        "# Hysteresis prevents rapid switching between experts.",
    ),
    (
        r"# Evolutionary Crossover Event \(E-COOP specific mechanic\)\s*# At rigid generational intervals, E-COOP duplicates the best expert\.",
        "# E-COOP duplicates the best expert at defined generation intervals.",
    ),
    (
        r"# FIM-Scaled Asexual Mutation\s*# E-COOP uses Fisher Information Matrix scaling to inject noise,\s*# protecting critical neural pathways while mutating flat ones\.",
        "# FIM-scaled mutation applies noise to parameter manifolds.",
    ),
]
process_file("src/train_ecoop.py", ecoop_reps)

# 4. MAPPO
mappo_reps = [
    (
        r'"""\s*MAPPO.*?"""',
        '"""\nMulti-Agent Proximal Policy Optimization (MAPPO).\n\nThis script trains a decentralized policy with centralized value functions.\n"""',
    )
]
process_file("src/train_mappo.py", mappo_reps)

# 5. COOP
coop_reps = [
    (
        r'"""\s*CO-OP: Confidence-Oriented Option Pool.*?"""',
        '"""\nConfidence-Oriented Option Pool (CO-OP).\n\nThis implements base confidence routing without evolutionary crossover events.\n"""',
    )
]
process_file("src/train_coop.py", coop_reps)

# 6. env.py (Add logging)
env_reps = [
    (
        r"def _scout_vision\(self\):",
        'def _scout_vision(self):\n        logging.debug(f"Computing scout vision at step {self.current_step}")',
    ),
    (
        r"def _hacker_hack\(self, action\):",
        'def _hacker_hack(self, action):\n        logging.debug(f"Hacker executes hack action {action}")',
    ),
]
process_file("src/env.py", env_reps, add_logging=True)

# 7. vec_env.py (Add logging)
vec_env_reps = [
    (
        r"def step_async\(self, actions\):",
        'def step_async(self, actions):\n        logging.debug("VectorEnv dispatching actions to workers.")',
    )
]
process_file("src/vec_env.py", vec_env_reps, add_logging=True)

print("Rewrote comments and added logs.")
