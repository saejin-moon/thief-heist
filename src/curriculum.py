import argparse
import importlib
import os

from constants import CURRICULUM_STAGES


def run_curriculum(algo_name):
    print(f"Starting Curriculum for {algo_name}")

    # Dynamically import the requested trainer
    module_name = f"train_{algo_name}"
    try:
        trainer = importlib.import_module(module_name)
    except ImportError:
        print(f"Error: Could not import src/{module_name}.py")
        return

    base_dir = f"results/{algo_name}"
    os.makedirs(base_dir, exist_ok=True)

    load_ckpt_path = None

    for stage_idx, stage_config in enumerate(CURRICULUM_STAGES):
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

        # Update checkpoint for next stage
        load_ckpt_path = os.path.join(save_ckpt_dir, "model.pt")

        if not os.path.exists(load_ckpt_path):
            print(
                f"Error: Checkpoint not found at {load_ckpt_path}. Halting curriculum."
            )
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["mappo", "coop", "marc", "hmappo", "ecoop"],
    )
    args = parser.parse_args()
    run_curriculum(args.algo)
