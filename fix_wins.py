import re


def modify_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    # Add global tracking variables before the update loop
    content = re.sub(
        r"(    num_updates = total_timesteps // \(NUM_ENVS \* NUM_STEPS\)\n)",
        r"\1    global_episodes = 0\n    global_wins = 0\n",
        content,
    )

    # Insert b_wins buffer creation
    if "train_marc.py" in filepath or "train_ecoop.py" in filepath:
        content = re.sub(
            r"(        b_dones = torch\.zeros\(\(NUM_STEPS, NUM_ENVS\)\)\.to\(device\)\n|        dones_t = torch\.zeros\(\(NUM_STEPS, NUM_ENVS\)\)\.to\(device\)\n)",
            r"\1        b_wins = torch.zeros((NUM_STEPS, NUM_ENVS), dtype=torch.bool).to(device)\n",
            content,
        )

    # Insert logic after vec_env.step
    # Note: hmappo/ecoop/marc/mappo/coop might name 'terms', 'truncs' differently.
    # We can match `infos = vec_env.step(actions_dict)`
    step_match = r"(            .*?next_obs.*?, rewards, terms, truncs, infos = vec_env\.step\(actions_dict\)\n|            .*?next_obs.*?, rewards, terminations, truncations, infos = envs\.step\(\n                actions_dict\n            \)\n)"

    tracker_logic = r"""\1
            for e in range(NUM_ENVS):
                if infos[e]["scout"].get("win", False):
                    if "b_wins" in locals():
                        b_wins[step, e] = True
                # Check for termination to update tracking metrics
                is_done = False
                if 'terms' in locals() and (terms["scout"][e] or truncs["scout"][e]):
                    is_done = True
                elif 'terminations' in locals() and (terminations["scout"][e] or truncations["scout"][e]):
                    is_done = True
                    
                if is_done:
                    global_episodes += 1
                    if infos[e]["scout"].get("win", False):
                        global_wins += 1
"""
    content = re.sub(step_match, tracker_logic, content)

    # Replace logging win_rate calculation
    content = re.sub(
        r"            win_rate = .*?max\(dim=0\)\.values\.mean\(\)\.item\(\)",
        r"            win_rate = global_wins / max(1, global_episodes)",
        content,
    )
    content = re.sub(
        r"            wins = \(rewards_t\[:, 0, :\] > WIN_REWARD_THRESHOLD\)\.sum\(\)\.item\(\)\n            episodes = dones_t\.sum\(\)\.item\(\)\n            win_rate = wins / max\(1\.0, episodes\)",
        r"            win_rate = global_wins / max(1, global_episodes)",
        content,
    )
    content = re.sub(
        r"            win_rate = \(ext_rewards\[:, 0, :\] > WIN_REWARD_THRESHOLD\)\.any\(dim=0\)\.float\(\)\.mean\(\)\.item\(\)",
        r"            win_rate = global_wins / max(1, global_episodes)",
        content,
    )

    # Fix MARC specific macro weighting
    if "train_marc.py" in filepath:
        marc_patch = r"""            # Determine trajectory win states directly from environment infos
            win_mask = b_wins.any(dim=0)"""
        content = re.sub(
            r"            win_mask = torch\.zeros\(NUM_ENVS, dtype=torch\.bool\)\.to\(device\)\n            for a in AGENTS:\n                win_mask \|= \(b_rewards\[a\] > WIN_REWARD_THRESHOLD\)\.any\(dim=0\)",
            marc_patch,
            content,
        )

    # Fix E-COOP specific macro weighting
    if "train_ecoop.py" in filepath:
        ecoop_patch = r"""                    if b_wins[t, e]:"""
        content = re.sub(
            r"                    if rewards_t\[t, 0, e\] > WIN_REWARD_THRESHOLD:",
            ecoop_patch,
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
    modify_file(file)

print("Win tracking fixed!")
