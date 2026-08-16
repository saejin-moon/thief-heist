import os
import re


def fix_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    # 1. Fix the locals() hack for terms/truncs
    bad_tracker = r"""                is_done = False
                if 'terms' in locals\(\) and \(terms\["scout"\]\[e\] or truncs\["scout"\]\[e\]\):
                    is_done = True
                elif 'terminations' in locals\(\) and \(terminations\["scout"\]\[e\] or truncations\["scout"\]\[e\]\):
                    is_done = True
                    
                if is_done:"""

    good_tracker = r"""                if terms["scout"][e] or truncs["scout"][e]:"""

    content = re.sub(bad_tracker, good_tracker, content)

    # 2. Fix win_mask in train_marc.py
    if "train_marc.py" in filepath:
        # Insert win_mask = b_wins.any(dim=0)
        content = content.replace(
            "# Apply alarm scaling and outcome factors.",
            "win_mask = b_wins.any(dim=0)\n            # Apply alarm scaling and outcome factors.",
        )

    # 3. Remove redefined constants
    # In train_marc.py, GAMMA_CAUSAL and AFFORDANCE_COEF were defined twice.
    if "train_marc.py" in filepath:
        content = re.sub(
            r"GAMMA_CAUSAL = 0\.95\s*# Retroactive discount factor for causal trace\n",
            "",
            content,
        )
        content = re.sub(
            r"AFFORDANCE_COEF = 0\.5\s*# Reward weight for unlocking affordances for the team\n",
            "",
            content,
        )

    with open(filepath, "w") as f:
        f.write(content)


for file in [
    "src/train_mappo.py",
    "src/train_coop.py",
    "src/train_marc.py",
    "src/train_hmappo.py",
    "src/train_ecoop.py",
]:
    if os.path.exists(file):
        fix_file(file)

print("Ruff fixes applied.")
