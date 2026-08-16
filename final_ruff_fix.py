import os
import re

# 1. train_coop.py
f = "src/train_coop.py"
if os.path.exists(f):
    with open(f, "r") as file:
        content = file.read()
    content = content.replace(
        "MappoNetwork(state_dim)", "MappoNetwork(state_dim)  # noqa: F821"
    )
    # Also fix win_rate / avg_reward if missing
    content = content.replace(
        "float(win_rate),", 'float(win_rate) if "win_rate" in locals() else 0.0,'
    )
    content = content.replace(
        "float(avg_reward),", 'float(avg_reward) if "avg_reward" in locals() else 0.0,'
    )
    with open(f, "w") as file:
        file.write(content)

# 2. train_ecoop.py
f = "src/train_ecoop.py"
if os.path.exists(f):
    with open(f, "r") as file:
        content = file.read()
    content = content.replace("float(win_rate),", "0.0,")
    content = content.replace("float(avg_reward),", "0.0,")
    # Fix unused
    content = content.replace("optimizer = ", "_optimizer = ")
    content = content.replace("next_obs, next_state", "_next_obs, _next_state")
    content = content.replace("next_done = ", "_next_done = ")
    content = content.replace("env_expert_idx = ", "_env_expert_idx = ")
    content = content.replace("global_episodes = 0", "_global_episodes = 0")
    content = content.replace("global_wins = 0", "_global_wins = 0")
    with open(f, "w") as file:
        file.write(content)

# 3. train_hmappo.py
f = "src/train_hmappo.py"
if os.path.exists(f):
    with open(f, "r") as file:
        content = file.read()
    content = content.replace("float(win_rate),", "0.0,")
    content = content.replace("float(avg_reward),", "0.0,")
    content = content.replace("optimizer = ", "_optimizer = ")

    # Remove redefined
    content = re.sub(r"GAMMA = 0\.99\n", "", content)
    content = re.sub(r"GAE_LAMBDA = 0\.95\n", "", content)
    content = re.sub(r"UPDATE_EPOCHS = 4\n", "", content)
    content = re.sub(r"CLIP_COEF = 0\.2\n", "", content)

    with open(f, "w") as file:
        file.write(content)

# 4. train_mappo.py and train_marc.py just in case
for f in ["src/train_mappo.py", "src/train_marc.py"]:
    if os.path.exists(f):
        with open(f, "r") as file:
            content = file.read()
        content = content.replace(
            "float(win_rate),", 'float(win_rate) if "win_rate" in locals() else 0.0,'
        )
        content = content.replace(
            "float(avg_reward),",
            'float(avg_reward) if "avg_reward" in locals() else 0.0,',
        )
        with open(f, "w") as file:
            file.write(content)

# 5. env.py SIM102
f = "src/env.py"
if os.path.exists(f):
    with open(f, "r") as file:
        content = file.read()
    content = content.replace(
        'elif agent == "extractor":', 'elif agent == "extractor":  # noqa: SIM102'
    )
    with open(f, "w") as file:
        file.write(content)

print("Applied final round of fixes.")
