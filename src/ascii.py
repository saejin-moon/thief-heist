"""
HEIST Terminal ASCII Visualizer & Debugger.

Allows stepping through environment episodes with rich ANSI colors,
live status HUD, and model checkpoint inference for all supported algorithms.
"""

import argparse
import os
import sys
import time

import torch

from constants import (
    AGENTS,
    ALARM_MAX,
    CAMERA,
    CURRICULUM_STAGES,
    DOOR,
    EXTRACT,
    LOOT,
    MACRO_STEP,
    MAP_SIZE,
    N_AGENTS,
    TERMINAL,
    WAIT,
    WALL,
)
from env import HeistEnv
from train_coma import ComaNetwork
from train_coop import CoopNetwork
from train_ecoop import EcoopNetwork
from train_hmappo import HierarchicalNetwork
from train_mappo import MappoNetwork
from train_marc import MarcNetwork

ACTION_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    4: "WAIT",
    5: "INTERACT",
}

# ANSI Colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
INVERT = "\033[7m"

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

BRIGHT_RED = "\033[91m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_WHITE = "\033[97m"
GRAY = "\033[90m"

AGENT_COLORS = {
    "scout": BRIGHT_CYAN,
    "hacker": BRIGHT_MAGENTA,
    "muscle": BRIGHT_RED,
    "extractor": BRIGHT_YELLOW,
}

AGENT_GLYPHS = {
    "scout": "S",
    "hacker": "H",
    "muscle": "M",
    "extractor": "E",
}


def load_model(algo: str, state_dim: int, checkpoint_path: str, device: str = "cpu"):
    """Load neural network architecture and optional weights for inference."""
    agent = None
    if algo == "mappo":
        agent = MappoNetwork(state_dim).to(device)
    elif algo == "coop":
        agent = CoopNetwork(state_dim, num_experts=2).to(device)
    elif algo == "ecoop":
        agent = EcoopNetwork(state_dim, num_initial_experts=1).to(device)
    elif algo == "hmappo":
        agent = HierarchicalNetwork(state_dim).to(device)
    elif algo == "marc":
        agent = MarcNetwork(state_dim).to(device)
    elif algo == "coma":
        agent = ComaNetwork(state_dim).to(device)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state_dict = (
                ckpt["model_state"]
                if (isinstance(ckpt, dict) and "model_state" in ckpt)
                else ckpt
            )
            if algo == "ecoop":
                expert_indices = {
                    int(k.split(".")[1]) for k in state_dict if k.startswith("experts.")
                }
                needed = max(expert_indices) + 1 if expert_indices else 1
                while len(agent.experts) < needed:
                    agent.add_expert()
                if len(agent.experts) > needed:
                    agent.prune_experts(list(range(needed)))
            agent.load_state_dict(state_dict)
            print(f"{GREEN}[✓] Loaded model weights from: {checkpoint_path}{RESET}")
        except Exception as e:  # noqa: BLE001
            print(
                f"{YELLOW}[!] Failed to load checkpoint ({e}). Using initialized policy.{RESET}"
            )
    else:
        print(
            f"{YELLOW}[!] No checkpoint found at '{checkpoint_path}'. Using uninitialized policy.{RESET}"
        )

    agent.eval()
    return agent


def render_alarm_bar(alarm: float, width: int = 24) -> str:
    """Return colored ANSI progress bar for alarm."""
    ratio = min(max(alarm / ALARM_MAX, 0.0), 1.0)
    filled = round(ratio * width)
    empty = width - filled

    if alarm < 40:
        bar_color = BRIGHT_GREEN
    elif alarm < 75:
        bar_color = BRIGHT_YELLOW
    else:
        bar_color = BRIGHT_RED

    bar = f"{bar_color}{'█' * filled}{GRAY}{'░' * empty}{RESET}"
    return f"[{bar}] {bar_color}{alarm:5.1f} / {ALARM_MAX:.1f}{RESET}"


def render_ascii_frame(
    env: HeistEnv,
    step_num: int,
    actions: dict,
    rewards: dict,
    total_reward: float,
    algo: str,
    stage: int,
    seed: int | None,
    god_mode: bool = True,
):
    """Clear terminal and render full ASCII view with colored glyphs and HUD."""
    h, w = env.map_h, env.map_w
    grid = env.grid

    # Clear screen
    sys.stdout.write("\033[H\033[J")

    # Header
    print(
        f"{BOLD}{BRIGHT_BLUE}╔════════════════════════════════════════════════════════════════════╗{RESET}"
    )
    print(
        f"{BOLD}{BRIGHT_BLUE}║ {BRIGHT_WHITE}HEIST DEBUGGER{BRIGHT_BLUE} │ Algo: {BRIGHT_CYAN}{algo.upper():<7}{BRIGHT_BLUE} │ Stage: {BRIGHT_YELLOW}{stage} ({h}x{w}){BRIGHT_BLUE} │ Step: {BRIGHT_WHITE}{step_num:03d}/{env.config.get('max_steps', 100):<3d}{BRIGHT_BLUE} ║{RESET}"
    )
    print(
        f"{BOLD}{BRIGHT_BLUE}║ {WHITE}Alarm: {render_alarm_bar(env.alarm)}    Seed: {seed!s:<6}   {BRIGHT_BLUE}║{RESET}"
    )

    # Objectives Status
    term_status = (
        f"{BRIGHT_GREEN}[✓] Hacked{RESET}"
        if env.terminal_disabled
        else f"{GRAY}[ ] Locked (Progress: {env.hack_progress}/3){RESET}"
    )
    loot_status = (
        f"{BRIGHT_GREEN}[✓] Secured{RESET}"
        if env.loot_acquired
        else f"{GRAY}[ ] In Vault{RESET}"
    )
    ext_status = (
        f"{BRIGHT_GREEN}[✓] Triggered ({env.extraction_countdown}s){RESET}"
        if env.extraction_triggered
        else f"{GRAY}[ ] Inactive{RESET}"
    )

    print(
        f"{BOLD}{BRIGHT_BLUE}║ {WHITE}Terminal: {term_status}  │ Loot: {loot_status}  │ Escape: {ext_status} {BRIGHT_BLUE}║{RESET}"
    )
    print(
        f"{BOLD}{BRIGHT_BLUE}╠════════════════════════════════════════════════════════════════════╣{RESET}"
    )

    # Build character grid
    char_grid = []
    for r in range(h):
        row = []
        for c in range(w):
            tile = grid[r, c]
            is_explored = env.explored_map[r, c]

            if not god_mode and not is_explored:
                row.append(f"{GRAY}·{RESET}")
                continue

            if tile == WALL:
                row.append(f"{BLUE}█{RESET}")
            elif tile == TERMINAL:
                glyph = (
                    f"{DIM}T{RESET}"
                    if env.terminal_disabled
                    else f"{BRIGHT_BLUE}{BOLD}T{RESET}"
                )
                row.append(glyph)
            elif tile == LOOT:
                glyph = (
                    f"{DIM}${RESET}"
                    if env.loot_acquired
                    else f"{BRIGHT_YELLOW}{BOLD}${RESET}"
                )
                row.append(glyph)
            elif tile == EXTRACT:
                row.append(f"{BRIGHT_GREEN}{BOLD}X{RESET}")
            elif tile == DOOR:
                row.append(f"{YELLOW}D{RESET}")
            elif tile == CAMERA:
                row.append(f"{MAGENTA}C{RESET}")
            else:
                row.append(f"{GRAY}·{RESET}")
        char_grid.append(row)

    # Overlay Guards
    for gi, (gr, gc) in enumerate(env.guard_positions):
        if (0 <= gr < h and 0 <= gc < w) and (god_mode or env.explored_map[gr, gc]):
            if env.neutralized[gi] > 0:
                char_grid[gr][gc] = f"{GRAY}{BOLD}g{RESET}"
            else:
                char_grid[gr][gc] = f"{BRIGHT_RED}{BOLD}{INVERT}G{RESET}"

    # Overlay Agents
    for agent, (ar, ac) in env.agent_positions.items():
        if 0 <= ar < h and 0 <= ac < w:
            color = AGENT_COLORS[agent]
            glyph = AGENT_GLYPHS[agent]
            char_grid[ar][ac] = f"{color}{BOLD}{INVERT}{glyph}{RESET}"

    # Print Grid
    for row in char_grid:
        print("  " + "".join(row))

    print(
        f"{BOLD}{BRIGHT_BLUE}╠════════════════════════════════════════════════════════════════════╣{RESET}"
    )

    # Agent Action Breakdown
    for agent in AGENTS:
        pos = env.agent_positions.get(agent, (0, 0))
        act = actions.get(agent, WAIT)
        act_name = ACTION_NAMES.get(act, str(act))
        rew = rewards.get(agent, 0.0)
        color = AGENT_COLORS[agent]
        print(
            f"  {color}{BOLD}{agent.capitalize():<9}{RESET} pos: {pos!s:<8} action: {BRIGHT_WHITE}{act_name:<9}{RESET} reward: {rew:+6.3f}"
        )

    print(
        f"{BOLD}{BRIGHT_BLUE}╚════════════════════════════════════════════════════════════════════╝{RESET}"
    )
    print(
        f"  {WHITE}Total Episode Return (Per-Agent Avg): {BOLD}{BRIGHT_GREEN if total_reward >= 0 else BRIGHT_RED}{total_reward:+.3f}{RESET}"
    )
    sys.stdout.flush()


def run_playback(
    algo: str = "mappo",
    stage: int = 0,
    checkpoint: str | None = None,
    delay: float = 0.15,
    seed: int | None = None,
    god_mode: bool = True,
    max_steps: int | None = None,
):
    """Run interactive playback loop for a single episode."""
    if stage < 0 or stage >= len(CURRICULUM_STAGES):
        raise ValueError(f"Stage must be between 0 and {len(CURRICULUM_STAGES) - 1}")

    config = dict(CURRICULUM_STAGES[stage])
    if max_steps is not None:
        config["max_steps"] = max_steps

    env = HeistEnv(config)
    state_dim = 6 + (MAP_SIZE[0] * MAP_SIZE[1]) + (N_AGENTS * 2) + 24 + 12

    if checkpoint is None:
        checkpoint = os.path.join("results", algo, f"stage_{stage}", "model.pt")

    agent_model = load_model(algo, state_dim, checkpoint, device="cpu")

    obs, _ = env.reset(seed=seed)
    step_num = 0
    total_reward = 0.0
    actions = {a: WAIT for a in AGENTS}
    rewards = {a: 0.0 for a in AGENTS}

    current_goals = {a: torch.zeros(2) for a in AGENTS}

    while True:
        step_num += 1
        state_t = torch.tensor(env.state(), dtype=torch.float32).unsqueeze(0)

        # Decide actions
        with torch.no_grad():
            if algo == "mappo":
                for a in AGENTS:
                    o = torch.tensor(
                        obs[a]["observation"], dtype=torch.float32
                    ).unsqueeze(0)
                    r = torch.tensor(obs[a]["role_id"], dtype=torch.float32).unsqueeze(
                        0
                    )
                    m = torch.tensor(
                        obs[a]["action_mask"], dtype=torch.float32
                    ).unsqueeze(0)
                    act, _, _, _ = agent_model.get_action_and_value(o, r, m, state_t)
                    actions[a] = int(act.item())
            elif algo == "coop":
                for a in AGENTS:
                    o = torch.tensor(
                        obs[a]["observation"], dtype=torch.float32
                    ).unsqueeze(0)
                    r = torch.tensor(obs[a]["role_id"], dtype=torch.float32).unsqueeze(
                        0
                    )
                    m = torch.tensor(
                        obs[a]["action_mask"], dtype=torch.float32
                    ).unsqueeze(0)
                    act, _, _, _, _ = agent_model.get_action_and_value(o, r, m, state_t)
                    actions[a] = int(act.item())
            elif algo == "ecoop":
                for a in AGENTS:
                    o = torch.tensor(
                        obs[a]["observation"], dtype=torch.float32
                    ).unsqueeze(0)
                    r = torch.tensor(obs[a]["role_id"], dtype=torch.float32).unsqueeze(
                        0
                    )
                    m = torch.tensor(
                        obs[a]["action_mask"], dtype=torch.float32
                    ).unsqueeze(0)
                    act, _, _, _, _ = agent_model.get_action_and_value(
                        o,
                        r,
                        m,
                        state_t,
                        active_experts=len(agent_model.experts),
                        deterministic=True,
                    )
                    actions[a] = int(act.item())
            elif algo == "hmappo":
                if (step_num - 1) % MACRO_STEP == 0:
                    for a in AGENTS:
                        r = torch.tensor(
                            obs[a]["role_id"], dtype=torch.float32
                        ).unsqueeze(0)
                        m_act, _, _, _ = agent_model.get_manager_action_and_value(
                            state_t, r
                        )
                        current_goals[a] = m_act.squeeze(0)

                for a in AGENTS:
                    o = torch.tensor(
                        obs[a]["observation"], dtype=torch.float32
                    ).unsqueeze(0)
                    r = torch.tensor(obs[a]["role_id"], dtype=torch.float32).unsqueeze(
                        0
                    )
                    m = torch.tensor(
                        obs[a]["action_mask"], dtype=torch.float32
                    ).unsqueeze(0)
                    g = current_goals[a].unsqueeze(0)
                    act, _, _, _ = agent_model.get_worker_action_and_value(
                        o, r, m, g, state_t
                    )
                    actions[a] = int(act.item())
            elif algo == "marc":
                for a in AGENTS:
                    o = torch.tensor(
                        obs[a]["observation"], dtype=torch.float32
                    ).unsqueeze(0)
                    r = torch.tensor(obs[a]["role_id"], dtype=torch.float32).unsqueeze(
                        0
                    )
                    m = torch.tensor(
                        obs[a]["action_mask"], dtype=torch.float32
                    ).unsqueeze(0)
                    act, _, _, _ = agent_model.get_action_and_value(o, r, m, state_t)
                    actions[a] = int(act.item())
            elif algo == "coma":
                for a in AGENTS:
                    o = torch.tensor(
                        obs[a]["observation"], dtype=torch.float32
                    ).unsqueeze(0)
                    r = torch.tensor(obs[a]["role_id"], dtype=torch.float32).unsqueeze(
                        0
                    )
                    m = torch.tensor(
                        obs[a]["action_mask"], dtype=torch.float32
                    ).unsqueeze(0)
                    act, _, _, _ = agent_model.get_action(o, r, m)
                    actions[a] = int(act.item())

        render_ascii_frame(
            env=env,
            step_num=step_num,
            actions=actions,
            rewards=rewards,
            total_reward=total_reward,
            algo=algo,
            stage=stage,
            seed=seed,
            god_mode=god_mode,
        )

        if delay > 0:
            time.sleep(delay)
        else:
            input(f"\n{DIM}[Press Enter to step, Ctrl+C to exit]{RESET}")

        obs, rewards, terms, truncs, infos = env.step(actions)
        step_avg_reward = sum(rewards.values()) / N_AGENTS
        total_reward += step_avg_reward

        done = any(terms.values()) or any(truncs.values())
        if done:
            render_ascii_frame(
                env=env,
                step_num=step_num,
                actions=actions,
                rewards=rewards,
                total_reward=total_reward,
                algo=algo,
                stage=stage,
                seed=seed,
                god_mode=god_mode,
            )
            is_win = infos.get("scout", {}).get("win", False)
            if is_win:
                print(
                    f"\n{BOLD}{BRIGHT_GREEN}★★★ HEIST SUCCESSFUL! The team escaped with the loot! ★★★{RESET}\n"
                )
            else:
                print(
                    f"\n{BOLD}{BRIGHT_RED}✘ HEIST FAILED! (Caught, timed out, or max alarm reached) ✘{RESET}\n"
                )
            break


def main():
    parser = argparse.ArgumentParser(
        description="HEIST ASCII Terminal Visualizer & Debugger"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="mappo",
        choices=["mappo", "coop", "ecoop", "hmappo", "marc"],
        help="Algorithm network architecture to evaluate",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=0,
        help="Curriculum stage index (0 to 4)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt). Defaults to results/{algo}/stage_{stage}/model.pt",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.15,
        help="Delay in seconds between frames (0 for interactive enter key)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for environment reset",
    )
    parser.add_argument(
        "--fog",
        action="store_true",
        help="Enable fog-of-war (defaults to god-mode map overview)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max steps for the episode",
    )

    args = parser.parse_args()

    try:
        run_playback(
            algo=args.algo,
            stage=args.stage,
            checkpoint=args.checkpoint,
            delay=args.delay,
            seed=args.seed,
            god_mode=not args.fog,
            max_steps=args.max_steps,
        )
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Playback terminated by user.{RESET}")


if __name__ == "__main__":
    main()
