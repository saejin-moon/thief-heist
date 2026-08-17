import argparse
import importlib
import os

from constants import CURRICULUM_STAGES


def parse_stages(stages_arg):
    """Parses stage string like '0', '0-2', '0,1,3', or 'all' into a list of stage integers."""
    if stages_arg is None or stages_arg.lower() == "all":
        return list(range(len(CURRICULUM_STAGES)))

    if "-" in stages_arg:
        parts = stages_arg.split("-")
        start, end = int(parts[0]), int(parts[1])
        return list(range(start, end + 1))

    if "," in stages_arg:
        return [int(x.strip()) for x in stages_arg.split(",") if x.strip()]

    return [int(stages_arg)]


def run_curriculum(algo_name, stages_to_run=None):
    if stages_to_run is None:
        stages_to_run = list(range(len(CURRICULUM_STAGES)))

    print(f"Starting Curriculum for {algo_name} on Stages: {stages_to_run}")

    # Dynamically import the requested trainer
    module_name = f"train_{algo_name}"
    try:
        trainer = importlib.import_module(module_name)
    except ImportError:
        print(f"Error: Could not import src/{module_name}.py")
        return

    base_dir = f"results/{algo_name}"
    os.makedirs(base_dir, exist_ok=True)

    # Automatically check if a checkpoint from the preceding stage exists
    first_stage = stages_to_run[0]
    load_ckpt_path = None
    if first_stage > 0:
        prev_ckpt = os.path.join(base_dir, f"stage_{first_stage - 1}", "model.pt")
        if os.path.exists(prev_ckpt):
            load_ckpt_path = prev_ckpt
            print(f"Found preceding checkpoint for Stage {first_stage} at {load_ckpt_path}")

    for stage_idx in stages_to_run:
        if stage_idx < 0 or stage_idx >= len(CURRICULUM_STAGES):
            print(f"Warning: Stage {stage_idx} out of bounds (0-{len(CURRICULUM_STAGES) - 1}). Skipping.")
            continue

        stage_config = CURRICULUM_STAGES[stage_idx]
        print(f"\n{'=' * 40}")
        print(
            f"Executing Stage {stage_idx} | Area: {stage_config['map_size'][0]}x{stage_config['map_size'][1]}"
        )
        print(f"{'=' * 40}")

        env_config = stage_config.copy()
        total_timesteps = env_config.pop("timesteps")

        stage_dir = os.path.join(base_dir, f"stage_{stage_idx}")
        os.makedirs(stage_dir, exist_ok=True)
        save_ckpt_dir = stage_dir
        log_dir = stage_dir

        # Call the train function
        trainer.train(
            algo_name=algo_name,
            stage_idx=stage_idx,
            env_config=env_config,
            total_timesteps=total_timesteps,
            load_ckpt_path=load_ckpt_path,
            save_ckpt_dir=save_ckpt_dir,
            log_dir=log_dir,
        )

        # Update checkpoint for next stage in sequence
        load_ckpt_path = os.path.join(save_ckpt_dir, "model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent RL Curriculum Runner")
    parser.add_argument(
        "--algo",
        type=str,
        default="all",
        choices=["mappo", "coop", "marc", "hmappo", "ecoop", "all"],
        help="Algorithm to train across curriculum, or 'all' to run all benchmarks",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Stages to train, e.g. '0', '0-2', '0,1,3', or 'all' (default: 'all')",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=None,
        help="Convenience alias to run a single stage (e.g. --stage 0)",
    )
    args = parser.parse_args()

    selected_stages = [args.stage] if args.stage is not None else parse_stages(args.stages)

    if args.algo == "all":
        for algo in ["mappo", "coop", "ecoop", "hmappo", "marc"]:
            run_curriculum(algo, stages_to_run=selected_stages)
    else:
        run_curriculum(args.algo, stages_to_run=selected_stages)
