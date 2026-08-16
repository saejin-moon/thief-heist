import os
import re


def fix_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    # 1. Fix the empty if block in train_mappo, train_coop, train_hmappo
    content = re.sub(
        r'                if infos\[e\]\["scout"\]\.get\("win", False\):\n\s*# Check for termination',
        r"                # Check for termination",
        content,
    )

    # 2. Fix the locals() for win_rate and avg_reward
    content = re.sub(
        r"(def train\(.*?\):\n)",
        r"\1    win_rate = 0.0\n    avg_reward = 0.0\n",
        content,
    )

    content = content.replace(
        'float(win_rate) if "win_rate" in locals() else 0.0', "float(win_rate)"
    )
    content = content.replace(
        "float(win_rate) if 'win_rate' in locals() else 0.0", "float(win_rate)"
    )
    content = content.replace(
        'float(avg_reward) if "avg_reward" in locals() else 0.0', "float(avg_reward)"
    )
    content = content.replace(
        "float(avg_reward) if 'avg_reward' in locals() else 0.0", "float(avg_reward)"
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

# 3. Fix env.py PERF102
with open("src/env.py", "r") as f:
    content = f.read()
content = content.replace(
    "for _agent, apos in self.agent_positions.items():",
    "for apos in self.agent_positions.values():",
)
with open("src/env.py", "w") as f:
    f.write(content)

print("Ruff fixes applied.")
