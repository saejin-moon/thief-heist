import re

# 1. Update src/constants.py
with open("src/constants.py", "r") as f:
    constants = f.read()

append_str = """
# Further RL & Training Constants
ENT_COEF = 0.01
VF_COEF = 0.5
WIN_REWARD_THRESHOLD = 5.0

# E-COOP Specific
ECOOP_EVOLUTION_START = 500
ECOOP_EVOLUTION_INTERVAL = 200
ECOOP_MUTATION_NOISE = 0.05

# MAHIRO / H-MAPPO Specific
MAHIRO_INTRINSIC_REWARD_COEF = 0.05
"""

if "ENT_COEF" not in constants:
    with open("src/constants.py", "a") as f:
        f.write(append_str)

import_additions = ", ENT_COEF, VF_COEF, WIN_REWARD_THRESHOLD, ECOOP_EVOLUTION_START, ECOOP_EVOLUTION_INTERVAL, ECOOP_MUTATION_NOISE, MAHIRO_INTRINSIC_REWARD_COEF"


# Helper function to modify files
def replace_in_file(filepath, pattern, replacement):
    with open(filepath, "r") as f:
        content = f.read()
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    with open(filepath, "w") as f:
        f.write(content)


def add_imports(filepath):
    replace_in_file(
        filepath, r"(from constants import .*?)\n", r"\1" + import_additions + r"\n"
    )


# 2. Modify train_ecoop.py
filepath = "src/train_ecoop.py"
add_imports(filepath)
replace_in_file(
    filepath,
    r"if update == 500 or \(update > 500 and \(update - 500\) % 200 == 0\):",
    "if update == ECOOP_EVOLUTION_START or (update > ECOOP_EVOLUTION_START and (update - ECOOP_EVOLUTION_START) % ECOOP_EVOLUTION_INTERVAL == 0):",
)
replace_in_file(
    filepath,
    r"torch\.randn_like\(param\) \* 0\.05",
    "torch.randn_like(param) * ECOOP_MUTATION_NOISE",
)
replace_in_file(
    filepath,
    r"torch\.randn_like\(param\) \* scale \* 0\.005",
    "torch.randn_like(param) * scale * (ECOOP_MUTATION_NOISE / 10.0)",
)

# 3. Modify train_hmappo.py
filepath = "src/train_hmappo.py"
add_imports(filepath)
replace_in_file(
    filepath,
    r"0\.05 \* intrinsic_reward",
    "MAHIRO_INTRINSIC_REWARD_COEF * intrinsic_reward",
)

# 4. Modify all files for ENT_COEF, VF_COEF, WIN_REWARD_THRESHOLD
for file in [
    "src/train_mappo.py",
    "src/train_coop.py",
    "src/train_marc.py",
    "src/train_hmappo.py",
    "src/train_ecoop.py",
]:
    add_imports(file)
    replace_in_file(file, r"0\.01 \* entropy\.mean\(\)", "ENT_COEF * entropy.mean()")
    replace_in_file(file, r"0\.5 \* v_loss", "VF_COEF * v_loss")
    replace_in_file(file, r"> 5\.0\)", "> WIN_REWARD_THRESHOLD)")

print("Hardcoded constants migrated successfully!")
