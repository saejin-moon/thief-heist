"""
HEIST Terminal ASCII Visualizer & Debugger.

Allows stepping through environment episodes with rich ANSI colors,
live status HUD, and model checkpoint inference for all supported algorithms.
Optimized for high throughput via native Rust environment backend and batched GPU inference.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
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
from train_coma import ComaNetwork
from train_coop import CoopNetwork
from train_ecoop import EcoopNetwork
from train_hmappo import HierarchicalNetwork
from train_mappo import MappoNetwork
from train_marc import MarcNetwork
from train_thief import ThiefNetwork

# Try importing the compiled Rust backend
RUST_AVAILABLE = False
try:
    rust_dir = os.path.join(os.path.dirname(__file__), "..", "rs", "target", "release")
    if os.path.exists(rust_dir) and rust_dir not in sys.path:
        sys.path.append(rust_dir)
    import heist_core_rs  # type: ignore

    RUST_AVAILABLE = True
except Exception:  # noqa: BLE001
    RUST_AVAILABLE = False

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


class RustEnvWrapper:
    """Wrapper exposing single-env semantics and render properties over PyHeistBatchEnv."""

    def __init__(self, config: dict, base_seed: int = 0):
        self.config = dict(config)
        self.max_steps = self.config.get("max_steps", 100)
        self.batch_env = heist_core_rs.PyHeistBatchEnv(
            1, json.dumps(self.config), base_seed
        )
        self.state_dim = self.batch_env.state_dim()

        self.obs_buf = np.zeros((N_AGENTS, 1, 7, 7), dtype=np.int32)
        self.mask_buf = np.zeros((N_AGENTS, 1, 6), dtype=np.int8)
        self.role_buf = np.zeros((N_AGENTS, 1, 4), dtype=np.int8)
        self.goal_buf = np.zeros((N_AGENTS, 1, 2), dtype=np.float32)
        self.state_buf = np.zeros((1, self.state_dim), dtype=np.float32)

        self.rew_buf = np.zeros((N_AGENTS, 1), dtype=np.float32)
        self.term_buf = np.zeros((N_AGENTS, 1), dtype=bool)
        self.trunc_buf = np.zeros((N_AGENTS, 1), dtype=bool)

        self._render_state = {}

    def reset(self, seed: int | None = None):
        use_seed = seed if seed is not None else 0
        self.batch_env.reset(
            use_seed,
            self.obs_buf,
            self.mask_buf,
            self.role_buf,
            self.goal_buf,
            self.state_buf,
        )
        self._render_state = self.batch_env.get_render_state(0)
        return self._get_obs_dict(), {}

    def step(self, actions: dict | np.ndarray | torch.Tensor):
        if isinstance(actions, dict):
            actions_arr = np.array([[actions[a]] for a in AGENTS], dtype=np.int32)
        elif isinstance(actions, torch.Tensor):
            actions_arr = actions.cpu().numpy().reshape(N_AGENTS, 1).astype(np.int32)
        else:
            actions_arr = np.array(actions, dtype=np.int32).reshape(N_AGENTS, 1)

        raw_infos = self.batch_env.step(
            actions_arr,
            self.obs_buf,
            self.mask_buf,
            self.role_buf,
            self.goal_buf,
            self.rew_buf,
            self.term_buf,
            self.trunc_buf,
            self.state_buf,
        )
        self._render_state = self.batch_env.get_render_state(0)

        rewards = {a: float(self.rew_buf[i, 0]) for i, a in enumerate(AGENTS)}
        terms = {a: bool(self.term_buf[i, 0]) for i, a in enumerate(AGENTS)}
        truncs = {a: bool(self.trunc_buf[i, 0]) for i, a in enumerate(AGENTS)}
        infos = raw_infos[0] if raw_infos else {}

        return self._get_obs_dict(), rewards, terms, truncs, infos

    def _get_obs_dict(self) -> dict:
        return {
            a: {
                "observation": self.obs_buf[i, 0].copy(),
                "action_mask": self.mask_buf[i, 0].copy(),
                "role_id": self.role_buf[i, 0].copy(),
                "goal_vector": self.goal_buf[i, 0].copy(),
            }
            for i, a in enumerate(AGENTS)
        }

    def state(self) -> np.ndarray:
        return self.state_buf[0].copy()

    @property
    def map_h(self) -> int:
        return self._render_state.get("map_h", 0)

    @property
    def map_w(self) -> int:
        return self._render_state.get("map_w", 0)

    @property
    def grid(self) -> np.ndarray:
        flat = self._render_state.get("grid", [])
        return np.array(flat, dtype=np.int32).reshape(self.map_h, self.map_w)

    @property
    def explored_map(self) -> np.ndarray:
        flat = self._render_state.get("explored_map", [])
        return np.array(flat, dtype=bool).reshape(self.map_h, self.map_w)

    @property
    def alarm(self) -> float:
        return float(self._render_state.get("alarm", 0.0))

    @property
    def alarm_max(self) -> float:
        return float(self._render_state.get("alarm_max", 100.0))

    @property
    def terminal_disabled(self) -> bool:
        return bool(self._render_state.get("terminal_disabled", False))

    @property
    def hack_progress(self) -> int:
        return int(self._render_state.get("hack_progress", 0))

    @property
    def loot_acquired(self) -> bool:
        return bool(self._render_state.get("loot_acquired", False))

    @property
    def extraction_triggered(self) -> bool:
        return bool(self._render_state.get("extraction_triggered", False))

    @property
    def extraction_countdown(self) -> int:
        return int(self._render_state.get("extraction_countdown", 0))

    @property
    def guard_positions(self) -> list:
        return self._render_state.get("guard_positions", [])

    @property
    def neutralized(self) -> list:
        return self._render_state.get("neutralized", [])

    @property
    def agent_positions(self) -> dict:
        return self._render_state.get("agent_positions", {})

    @property
    def terminal_pos(self) -> tuple:
        return self._render_state.get("terminal_pos", (0, 0))

    @property
    def loot_pos(self) -> tuple:
        return self._render_state.get("loot_pos", (0, 0))

    @property
    def extract_pos(self) -> tuple:
        return self._render_state.get("extract_pos", (0, 0))


def load_model(algo: str, state_dim: int, checkpoint_path: str, device: str = "cuda"):
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
    elif algo == "thief":
        agent = ThiefNetwork(state_dim, num_initial_experts=2).to(device)
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
            if algo in ("ecoop", "thief"):
                expert_indices = {
                    int(k.split(".")[1]) for k in state_dict if k.startswith("experts.")
                }
                needed = max(expert_indices) + 1 if expert_indices else 1
                while len(agent.experts) < needed:
                    agent.add_expert()
                if len(agent.experts) > needed:
                    agent.prune_experts(list(range(needed)))
            agent.load_state_dict(state_dict)
            print(
                f"{GREEN}[✓] Loaded model weights from: {checkpoint_path} (Device: {device}){RESET}"
            )
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


def render_alarm_bar(
    alarm: float, width: int = 24, alarm_max: float = ALARM_MAX
) -> str:
    """Return colored ANSI progress bar for alarm."""
    ratio = min(max(alarm / max(1.0, alarm_max), 0.0), 1.0)
    filled = round(ratio * width)
    empty = width - filled

    if alarm < 0.4 * alarm_max:
        bar_color = BRIGHT_GREEN
    elif alarm < 0.75 * alarm_max:
        bar_color = BRIGHT_YELLOW
    else:
        bar_color = BRIGHT_RED

    bar = f"{bar_color}{'█' * filled}{GRAY}{'░' * empty}{RESET}"
    return f"[{bar}] {bar_color}{alarm:5.1f} / {alarm_max:.1f}{RESET}"


def render_ascii_frame(
    env,
    step_num: int,
    actions: dict,
    rewards: dict,
    total_reward: float,
    algo: str,
    stage: int,
    seed: int | None,
    god_mode: bool = True,
    device: str = "cuda",
    backend: str = "rust",
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
    max_steps_val = env.config.get("max_steps", 100) if hasattr(env, "config") else 100
    print(
        f"{BOLD}{BRIGHT_BLUE}║ {BRIGHT_WHITE}HEIST DEBUGGER{BRIGHT_BLUE} │ {BRIGHT_CYAN}{algo.upper():<6}{BRIGHT_BLUE} │ Stg {BRIGHT_YELLOW}{stage} ({h}x{w}){BRIGHT_BLUE} │ Step: {BRIGHT_WHITE}{step_num:03d}/{max_steps_val:<4d}{BRIGHT_BLUE} │ {MAGENTA}{backend.upper()}+{device.upper():<8}{BRIGHT_BLUE} ║{RESET}"
    )
    print(
        f"{BOLD}{BRIGHT_BLUE}║ {WHITE}Alarm: {render_alarm_bar(env.alarm, alarm_max=env.alarm_max)}    Seed: {seed!s:<6}   {BRIGHT_BLUE}║{RESET}"
    )

    # Objectives Status
    term_status = (
        f"{BRIGHT_GREEN}[✓] Hacked{RESET}"
        if env.terminal_disabled
        else f"{GRAY}[ ] Locked ({env.hack_progress}/3){RESET}"
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
        f"{BOLD}{BRIGHT_BLUE}║ {WHITE}Terminal: {term_status} │ Loot: {loot_status} │ Escape: {ext_status} {BRIGHT_BLUE}║{RESET}"
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
            if gi < len(env.neutralized) and env.neutralized[gi] > 0:
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
    device: str = "cuda",
    backend: str = "rust",
):
    """Run interactive playback loop for a single episode using Rust env and GPU inference."""
    if stage < 0 or stage >= len(CURRICULUM_STAGES):
        raise ValueError(f"Stage must be between 0 and {len(CURRICULUM_STAGES) - 1}")

    config = dict(CURRICULUM_STAGES[stage])
    if max_steps is not None:
        config["max_steps"] = max_steps

    use_seed = seed if seed is not None else 0

    env = RustEnvWrapper(config, base_seed=use_seed)
    actual_backend = "rust"

    state_dim = 6 + (MAP_SIZE[0] * MAP_SIZE[1]) + (N_AGENTS * 2) + 24 + 12

    if checkpoint is None:
        final_ckpt = os.path.join("results", "final", algo, f"seed_{use_seed}", f"stage_{stage}", "model.pt")
        final_seed0 = os.path.join("results", "final", algo, "seed_0", f"stage_{stage}", "model.pt")
        legacy_ckpt = os.path.join("results", algo, f"stage_{stage}", "model.pt")
        if os.path.exists(final_ckpt):
            checkpoint = final_ckpt
        elif os.path.exists(final_seed0):
            checkpoint = final_seed0
        else:
            checkpoint = legacy_ckpt

    agent_model = load_model(algo, state_dim, checkpoint, device=device)

    _obs, _ = env.reset(seed=seed)
    step_num = 0
    total_reward = 0.0
    actions = {a: WAIT for a in AGENTS}
    rewards = {a: 0.0 for a in AGENTS}

    current_goals = torch.zeros(N_AGENTS, 2, device=device)
    prev_experts = torch.full((N_AGENTS,), -1, dtype=torch.long, device=device)

    while True:
        step_num += 1

        with torch.no_grad():
            obs_t = (
                torch.from_numpy(env.obs_buf.reshape(N_AGENTS, 7, 7)).float().to(device)
            )
            role_t = (
                torch.from_numpy(env.role_buf.reshape(N_AGENTS, 4)).float().to(device)
            )
            mask_t = (
                torch.from_numpy(env.mask_buf.reshape(N_AGENTS, 6)).float().to(device)
            )
            goal_t = (
                torch.from_numpy(env.goal_buf.reshape(N_AGENTS, 2)).float().to(device)
            )
            state_t = (
                torch.from_numpy(env.state_buf).float().to(device).repeat(N_AGENTS, 1)
            )

            # Single batched forward pass on GPU
            if algo in ("mappo", "marc"):
                act_t, _, _, _ = agent_model.get_action_and_value(
                    obs_t, role_t, mask_t, goal_t, state_t
                )
            elif algo == "coop":
                act_t, _, _, _, _ = agent_model.get_action_and_value(
                    obs_t, role_t, mask_t, goal_t, state_t
                )
            elif algo == "ecoop":
                act_t, _, _, _, chosen_exp = agent_model.get_action_and_value(
                    obs_t,
                    role_t,
                    mask_t,
                    goal_t,
                    state_t,
                    active_experts=len(agent_model.experts),
                    previous_expert=prev_experts,
                )
                if chosen_exp is not None:
                    prev_experts = chosen_exp
            elif algo == "thief":
                act_t, _, _, _, chosen_exp = agent_model.get_action_and_value(
                    obs_t,
                    role_t,
                    mask_t,
                    goal_t,
                    state_t,
                    active_experts=len(agent_model.experts),
                    previous_expert=prev_experts,
                    num_envs=1,
                )
                if chosen_exp is not None:
                    prev_experts = chosen_exp
            elif algo == "coma":
                act_t, _, _, _ = agent_model.get_action(obs_t, role_t, mask_t, goal_t)
            elif algo == "hmappo":
                if (step_num - 1) % MACRO_STEP == 0:
                    m_act, _, _, _ = agent_model.get_manager_action_and_value(
                        state_t, role_t
                    )
                    current_goals = m_act
                act_t, _, _, _ = agent_model.get_worker_action_and_value(
                    obs_t, role_t, mask_t, current_goals, state_t
                )

            act_np = act_t.cpu().numpy()
            actions = {a: int(act_np[i]) for i, a in enumerate(AGENTS)}

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
            device=device,
            backend=actual_backend,
        )

        if delay > 0:
            time.sleep(delay)
        else:
            input(f"\n{DIM}[Press Enter to step, Ctrl+C to exit]{RESET}")

        _obs, rewards, terms, truncs, infos = env.step(actions)
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
                device=device,
                backend=actual_backend,
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
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(
        description="HEIST ASCII Terminal Visualizer & Debugger (GPU + Rust Accelerated)"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="mappo",
        choices=["mappo", "coop", "ecoop", "hmappo", "marc", "coma", "thief"],
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
    parser.add_argument(
        "--device",
        type=str,
        default=default_device,
        choices=["cuda", "cpu"],
        help=f"Device for policy inference (default: {default_device})",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="rust",
        choices=["rust", "python"],
        help="Environment engine backend (default: rust)",
    )
    parser.add_argument(
        "--gif",
        type=str,
        default=None,
        help="Path to save animated GIF playback (e.g. docs/assets/demo.gif)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="Frames per second for saved GIF (default: 6)",
    )

    args = parser.parse_args()

    if args.gif:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from scripts.record_ascii_gif import record_gif

        record_gif(
            algo=args.algo,
            stage=args.stage,
            seed=args.seed if args.seed is not None else 3,
            checkpoint=args.checkpoint or (
                os.path.join("results", "final", args.algo, f"seed_{args.seed if args.seed is not None else 0}", f"stage_{args.stage}", "model.pt")
                if os.path.exists(os.path.join("results", "final", args.algo, f"seed_{args.seed if args.seed is not None else 0}", f"stage_{args.stage}", "model.pt"))
                else os.path.join("results", args.algo, f"stage_{args.stage}", "model.pt")
            ),
            output_path=args.gif,
            device=args.device,
            fps=args.fps,
            max_steps=args.max_steps or 300,
        )
        return

    try:
        run_playback(
            algo=args.algo,
            stage=args.stage,
            checkpoint=args.checkpoint,
            delay=args.delay,
            seed=args.seed,
            god_mode=not args.fog,
            max_steps=args.max_steps,
            device=args.device,
            backend=args.backend,
        )
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Playback terminated by user.{RESET}")


if __name__ == "__main__":
    main()
