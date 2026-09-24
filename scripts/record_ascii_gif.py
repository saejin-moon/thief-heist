"""
Record HEIST ASCII gameplay to an animated GIF using trained model checkpoints.
Features:
- Full color ANSI terminal rendering to true-type font image frames.
- Search mode (--until-win) to simulate seeds until a successful victory is achieved.
- Accurate terminal state capture: freezes on the winning map layout without leaking
  the auto-reset subsequent map.
"""

import argparse
import copy
import os
import re
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "py"))

from ascii import (
    ACTION_NAMES,
    AGENT_COLORS,
    AGENT_GLYPHS,
    AGENTS,
    BLUE,
    BOLD,
    BRIGHT_BLUE,
    BRIGHT_CYAN,
    BRIGHT_GREEN,
    BRIGHT_MAGENTA,
    BRIGHT_RED,
    BRIGHT_WHITE,
    BRIGHT_YELLOW,
    CAMERA,
    CYAN,
    DIM,
    DOOR,
    EXTRACT,
    GRAY,
    GREEN,
    INVERT,
    LOOT,
    MAGENTA,
    RED,
    RESET,
    TERMINAL,
    WAIT,
    WALL,
    WHITE,
    YELLOW,
    RustEnvWrapper,
    load_model,
)
from constants import CURRICULUM_STAGES, MAP_SIZE, N_AGENTS

ANSI_PATTERN = re.compile(r"\x1b\[([0-9;]*)m|\x1b\[H|\x1b\[J")

ANSI_COLOR_MAP = {
    "30": (40, 44, 52),
    "31": (230, 80, 80),
    "32": (90, 210, 110),
    "33": (240, 195, 60),
    "34": (55, 85, 135),
    "35": (210, 110, 230),
    "36": (60, 210, 230),
    "37": (210, 215, 225),
    "90": (95, 105, 120),
    "91": (255, 95, 95),
    "92": (90, 230, 120),
    "93": (255, 215, 70),
    "94": (90, 170, 255),
    "95": (230, 130, 255),
    "96": (70, 230, 255),
    "97": (255, 255, 255),
}

DEFAULT_FG = (205, 215, 225)
DEFAULT_BG = (13, 16, 23)


def strip_ansi(s: str) -> str:
    return ANSI_PATTERN.sub("", s)


def render_alarm_bar(alarm: float, width: int = 20, alarm_max: float = 125.0) -> str:
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
    return f"[{bar}] {bar_color}{alarm:5.1f}/{alarm_max:.0f}{RESET}"


def tokenize_ansi(line: str):
    line = line.replace("\x1b[H", "").replace("\x1b[J", "")
    tokens = []
    fg = DEFAULT_FG
    bg = None
    invert = False
    bold = False

    pos = 0
    for match in ANSI_PATTERN.finditer(line):
        text = line[pos : match.start()]
        for ch in text:
            cur_fg = (DEFAULT_BG if bg is None else bg) if invert else fg
            cur_bg = fg if invert else bg
            tokens.append((ch, cur_fg, cur_bg, bold))
        pos = match.end()
        raw = match.group(1)
        if raw is None:
            continue
        codes = raw.split(";") if raw else ["0"]
        for code in codes:
            if code in ("0", ""):
                fg = DEFAULT_FG
                bg = None
                invert = False
                bold = False
            elif code == "1":
                bold = True
            elif code == "2":
                fg = (110, 115, 125)
            elif code == "7":
                invert = True
            elif code in ANSI_COLOR_MAP:
                fg = ANSI_COLOR_MAP[code]
    for ch in line[pos:]:
        cur_fg = (DEFAULT_BG if bg is None else bg) if invert else fg
        cur_bg = fg if invert else bg
        tokens.append((ch, cur_fg, cur_bg, bold))
    return tokens


def build_frame_lines(
    env,
    step_num: int,
    actions: dict,
    rewards: dict,
    total_reward: float,
    algo: str,
    stage: int,
    seed: int,
    box_w: int = 78,
    state_override: dict = None,
):
    lines = []
    h = state_override.get("map_h", env.map_h) if state_override else env.map_h
    w = state_override.get("map_w", env.map_w) if state_override else env.map_w

    if state_override and "grid" in state_override:
        grid_data = state_override["grid"]
        if isinstance(grid_data, np.ndarray):
            grid = grid_data
        else:
            grid = np.array(grid_data, dtype=np.int32).reshape(h, w)
    else:
        grid = env.grid

    alarm = state_override.get("alarm", env.alarm) if state_override else env.alarm
    alarm_max = state_override.get("alarm_max", env.alarm_max) if state_override else env.alarm_max
    terminal_disabled = (
        state_override.get("terminal_disabled", env.terminal_disabled)
        if state_override
        else env.terminal_disabled
    )
    hack_progress = (
        state_override.get("hack_progress", env.hack_progress)
        if state_override
        else env.hack_progress
    )
    loot_acquired = (
        state_override.get("loot_acquired", env.loot_acquired)
        if state_override
        else env.loot_acquired
    )
    extraction_triggered = (
        state_override.get("extraction_triggered", env.extraction_triggered)
        if state_override
        else env.extraction_triggered
    )
    extraction_countdown = (
        state_override.get("extraction_countdown", env.extraction_countdown)
        if state_override
        else env.extraction_countdown
    )
    guard_positions = (
        state_override.get("guard_positions", env.guard_positions)
        if state_override
        else env.guard_positions
    )
    neutralized = (
        state_override.get("neutralized", env.neutralized)
        if state_override
        else env.neutralized
    )
    agent_positions = (
        state_override.get("agent_positions", env.agent_positions)
        if state_override
        else env.agent_positions
    )

    max_steps_val = env.config.get("max_steps", 900)

    def box_row(content: str) -> str:
        vis_len = len(strip_ansi(content))
        pad = max(0, box_w - 4 - vis_len)
        return f"{BOLD}{BRIGHT_BLUE}║ {RESET}{content}{' ' * pad}{BOLD}{BRIGHT_BLUE} ║{RESET}"

    # Header top border
    lines.append(f"{BOLD}{BRIGHT_BLUE}╔{'═' * (box_w - 2)}╗{RESET}")

    # Title row
    line1 = f"{BRIGHT_WHITE}{BOLD}HEIST DEBUGGER{RESET} │ {BRIGHT_CYAN}{algo.upper():<5}{RESET} │ Stg {BRIGHT_YELLOW}{stage}{RESET} ({h}x{w}) │ Step: {BRIGHT_WHITE}{step_num:03d}/{max_steps_val:<3d}{RESET} │ {MAGENTA}RUST+CUDA{RESET}"
    lines.append(box_row(line1))

    # Alarm and seed
    line2 = f"Alarm: {render_alarm_bar(alarm, 20, alarm_max)}   Seed: {WHITE}{seed}{RESET}"
    lines.append(box_row(line2))

    # Objective status
    term_status = (
        f"{BRIGHT_GREEN}[OK] Hacked{RESET}"
        if terminal_disabled
        else f"{GRAY}[..] Locked ({hack_progress}/3){RESET}"
    )
    loot_status = (
        f"{BRIGHT_GREEN}[OK] Secured{RESET}"
        if loot_acquired
        else f"{GRAY}[..] In Vault{RESET}"
    )
    if state_override and loot_acquired and extraction_triggered:
        ext_status = f"{BRIGHT_GREEN}[OK] Escaped{RESET}"
    elif extraction_triggered:
        ext_status = f"{BRIGHT_GREEN}[OK] Triggered ({extraction_countdown}s){RESET}"
    else:
        ext_status = f"{GRAY}[..] Inactive{RESET}"
    line3 = f"Terminal: {term_status}  Loot: {loot_status}  Escape: {ext_status}"
    lines.append(box_row(line3))

    # Divider
    lines.append(f"{BOLD}{BRIGHT_BLUE}╠{'═' * (box_w - 2)}╣{RESET}")

    # Map grid (2 columns per tile for balanced aspect ratio)
    char_grid = []
    for r in range(h):
        row = []
        for c in range(w):
            tile = grid[r, c]
            if tile == WALL:
                row.append(f"{BLUE}██{RESET}")
            elif tile == TERMINAL:
                row.append(
                    f"{DIM}T {RESET}"
                    if terminal_disabled
                    else f"{BRIGHT_BLUE}{BOLD}T {RESET}"
                )
            elif tile == LOOT:
                row.append(
                    f"{DIM}$ {RESET}"
                    if loot_acquired
                    else f"{BRIGHT_YELLOW}{BOLD}$ {RESET}"
                )
            elif tile == EXTRACT:
                row.append(f"{BRIGHT_GREEN}{BOLD}X {RESET}")
            elif tile == DOOR:
                row.append(f"{YELLOW}D {RESET}")
            elif tile == CAMERA:
                row.append(f"{MAGENTA}C {RESET}")
            else:
                row.append(f"{GRAY}· {RESET}")
        char_grid.append(row)

    # Overlay guards
    for gi, (gr, gc) in enumerate(guard_positions):
        if 0 <= gr < h and 0 <= gc < w:
            if gi < len(neutralized) and neutralized[gi] > 0:
                char_grid[gr][gc] = f"{GRAY}{BOLD}g {RESET}"
            else:
                char_grid[gr][gc] = f"{BRIGHT_RED}{BOLD}{INVERT}G{RESET} "

    # Overlay agents
    for agent, (ar, ac) in agent_positions.items():
        if 0 <= ar < h and 0 <= ac < w:
            color = AGENT_COLORS[agent]
            glyph = AGENT_GLYPHS[agent]
            char_grid[ar][ac] = f"{color}{BOLD}{INVERT}{glyph}{RESET} "

    # Center map rows inside box
    map_pad = max(0, (box_w - 4 - (w * 2)) // 2)
    for row in char_grid:
        lines.append(box_row(" " * map_pad + "".join(row)))

    # Divider
    lines.append(f"{BOLD}{BRIGHT_BLUE}╠{'═' * (box_w - 2)}╣{RESET}")

    # Agents breakdown
    for agent in AGENTS:
        pos = agent_positions.get(agent, (0, 0))
        act = actions.get(agent, 4)
        act_name = ACTION_NAMES.get(act, str(act))
        rew = rewards.get(agent, 0.0)
        color = AGENT_COLORS[agent]
        line_agent = f"{color}{BOLD}{agent.capitalize():<9}{RESET} pos: {pos!s:<8} action: {BRIGHT_WHITE}{act_name:<9}{RESET} reward: {rew:+5.2f}"
        lines.append(box_row(line_agent))

    # Bottom border
    lines.append(f"{BOLD}{BRIGHT_BLUE}╚{'═' * (box_w - 2)}╝{RESET}")
    lines.append(
        f"  {WHITE}Total Episode Return: {BOLD}{BRIGHT_GREEN if total_reward >= 0 else BRIGHT_RED}{total_reward:+.3f}{RESET}"
    )

    return lines


def run_episode_frames(
    env,
    agent_model,
    algo: str,
    stage: int,
    seed: int,
    device: str,
    max_steps: int,
    font,
    box_w: int,
    cell_w: int = 9,
    cell_h: int = 20,
    pad_x: int = 24,
    pad_y: int = 20,
):
    width = pad_x * 2 + box_w * cell_w

    def render_image(lines, victory_text=None):
        total_lines = len(lines) + (2 if victory_text else 0)
        height = pad_y * 2 + total_lines * cell_h
        img = Image.new("RGB", (width, height), color=DEFAULT_BG)
        draw = ImageDraw.Draw(img)

        for r, line in enumerate(lines):
            tokens = tokenize_ansi(line)
            y = pad_y + r * cell_h
            for c, (ch, fg, bg, bold) in enumerate(tokens):
                x = pad_x + c * cell_w
                if bg is not None:
                    draw.rectangle([x, y, x + cell_w, y + cell_h], fill=bg)
                if ch == "█":
                    draw.rectangle([x, y + 1, x + cell_w, y + cell_h - 1], fill=fg)
                elif ch != " ":
                    cw = font.getlength(ch)
                    offset_x = max(0, (cell_w - cw) / 2)
                    draw.text((x + offset_x, y), ch, font=font, fill=fg)

        if victory_text:
            y_vic = pad_y + len(lines) * cell_h + 4
            draw.text((pad_x + 10, y_vic), victory_text, font=font, fill=(90, 230, 120))
        return img

    frames = []
    step_num = 0
    total_reward = 0.0
    actions = {a: 4 for a in AGENTS}
    rewards = {a: 0.0 for a in AGENTS}

    prev_experts = torch.full((N_AGENTS,), -1, dtype=torch.long, device=device)

    initial_lines = build_frame_lines(
        env, 0, actions, rewards, total_reward, algo, stage, seed, box_w=box_w
    )
    frames.append(render_image(initial_lines))

    done = False
    won = False

    while not done and step_num < max_steps:
        step_num += 1
        pre_step_state = copy.deepcopy(env._render_state)

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

            if algo == "thief":
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
            else:
                act_t, _, _, _ = agent_model.get_action_and_value(
                    obs_t, role_t, mask_t, goal_t, state_t
                )

        actions = {a: int(act_t[i].cpu()) for i, a in enumerate(AGENTS)}
        _obs, rewards, terms, truncs, infos = env.step(actions)
        step_avg_reward = sum(rewards.values()) / N_AGENTS
        total_reward += step_avg_reward

        done = any(terms.values()) or any(truncs.values())

        if not done:
            lines = build_frame_lines(
                env, step_num, actions, rewards, total_reward, algo, stage, seed, box_w=box_w
            )
            frames.append(render_image(lines))
        else:
            won = infos.get("scout", {}).get("win", False)
            # CRITICAL: construct terminal frame using pre_step_state updated with terminal info.
            # Never read from env._render_state because Rust auto-resets on terminal steps!
            winning_state = copy.deepcopy(pre_step_state)
            winning_state["agent_positions"] = {a: infos[a]["pos"] for a in AGENTS}
            winning_state["alarm"] = infos.get("scout", {}).get("alarm", winning_state["alarm"])
            if won:
                winning_state["loot_acquired"] = True
                winning_state["terminal_disabled"] = True
                winning_state["extraction_triggered"] = True

            winning_lines = build_frame_lines(
                env,
                step_num,
                actions,
                rewards,
                total_reward,
                algo,
                stage,
                seed,
                box_w=box_w,
                state_override=winning_state,
            )
            frames.append(render_image(winning_lines))

            vic_msg = (
                "=== HEIST SUCCESSFUL! ALL AGENTS ESCAPED WITH LOOT ==="
                if won
                else "=== HEIST FAILED ==="
            )
            final_frame = render_image(winning_lines, victory_text=vic_msg)
            # Hold winning frame for 18 frames (~2.25 seconds at 8 fps)
            hold_count = 18 if won else 8
            for _ in range(hold_count):
                frames.append(final_frame)

    return frames, won, step_num, total_reward


def record_gif(
    algo: str = "thief",
    stage: int = 3,
    seed: int = 12,
    checkpoint: str = None,
    output_path: str = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    fps: int = 8,
    max_steps: int = 600,
    until_win: bool = True,
    max_attempts: int = 40,
):
    if checkpoint is None:
        checkpoint = f"results/final/{algo}/seed_0/stage_{stage}/model.pt"
    if output_path is None:
        output_path = f"docs/assets/heist_stage{stage}_{algo}.gif"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    state_dim = 6 + (MAP_SIZE[0] * MAP_SIZE[1]) + (N_AGENTS * 2) + 24 + 12
    agent_model = load_model(algo, state_dim, checkpoint, device=device)

    # Find available monospace font
    font_candidates = [
        "/usr/share/fonts/liberation/LiberationMono-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/TTF/JetBrainsMonoNerdFontMono-Bold.ttf",
    ]
    font_path = None
    for cand in font_candidates:
        if os.path.exists(cand):
            font_path = cand
            break

    if font_path:
        font = ImageFont.truetype(font_path, 15)
    else:
        font = ImageFont.load_default()

    config = dict(CURRICULUM_STAGES[stage])
    config["max_steps"] = max_steps
    w = config.get("map_w", 35)
    box_w = max(78, w * 2 + 8)

    current_seed = seed
    attempts = 0
    chosen_frames = None
    final_won = False

    while attempts < max_attempts:
        attempts += 1
        print(f"[{attempts}/{max_attempts}] Simulating Stage {stage} ({algo}) with seed {current_seed}...")
        env = RustEnvWrapper(config, base_seed=current_seed)
        env.reset(seed=current_seed)

        frames, won, steps, ret = run_episode_frames(
            env=env,
            agent_model=agent_model,
            algo=algo,
            stage=stage,
            seed=current_seed,
            device=device,
            max_steps=max_steps,
            font=font,
            box_w=box_w,
        )

        print(f" -> Result: won={won}, steps={steps}, return={ret:+.2f}")

        if won or not until_win:
            chosen_frames = frames
            final_won = won
            break
        else:
            current_seed += 1

    if chosen_frames is None:
        print("Warning: Max attempts reached without win, saving last attempt.")
        chosen_frames = frames

    duration_ms = int(1000 / fps)
    chosen_frames[0].save(
        output_path,
        save_all=True,
        append_images=chosen_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    size_kb = os.path.getsize(output_path) / 1024
    print(
        f"Recorded GIF successfully: {output_path} ({len(chosen_frames)} frames, {size_kb:.1f} KB, won={final_won}, seed={current_seed})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Record HEIST ASCII Visualizer episode to animated GIF"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="thief",
        choices=["thief", "ecoop", "coop", "mappo", "hmappo", "marc", "coma"],
    )
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/assets/heist_stage3_thief.gif",
    )
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--until-win", action="store_true", default=True)
    parser.add_argument("--max-attempts", type=int, default=30)
    args = parser.parse_args()

    record_gif(
        algo=args.algo,
        stage=args.stage,
        seed=args.seed,
        checkpoint=args.checkpoint,
        output_path=args.output,
        fps=args.fps,
        max_steps=args.max_steps,
        until_win=args.until_win,
        max_attempts=args.max_attempts,
    )


if __name__ == "__main__":
    main()
