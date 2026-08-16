import os
import re

# 1. Update src/constants.py
constants_path = "src/constants.py"
with open(constants_path, "r") as f:
    constants_content = f.read()

append_str = """
# --- HYPERPARAMETERS ---
LR = 2.5e-4
NUM_ENVS = 8
NUM_STEPS = 125
GAMMA = 0.99
GAE_LAMBDA = 0.95
UPDATE_EPOCHS = 4
CLIP_COEF = 0.2

# Algorithm specific
MACRO_STEP = 5
ECOOP_POOL_SIZE = 8
ALPHA_ALARM = 1.5
GAMMA_CAUSAL = 0.95
AFFORDANCE_COEF = 0.5

# --- CURRICULUM STAGES ---
# Based on HEIST paper Section 4.1
CURRICULUM_STAGES = [
    {"map_size": (11, 11), "guard_count": 0, "camera_count": 0, "door_count": 0, "max_steps": 100, "spawn_mode": "role", "timesteps": 120_000},
    {"map_size": (17, 17), "guard_count": 1, "camera_count": 0, "door_count": 1, "max_steps": 150, "spawn_mode": "role", "timesteps": 343_933},
    {"map_size": (25, 25), "guard_count": 2, "camera_count": 1, "door_count": 2, "max_steps": 200, "spawn_mode": "role", "timesteps": 929_752},
    {"map_size": (35, 35), "guard_count": 3, "camera_count": 2, "door_count": 3, "max_steps": 250, "spawn_mode": "role", "timesteps": 2_186_776},
    {"map_size": (50, 50), "guard_count": 4, "camera_count": 3, "door_count": 4, "max_steps": 300, "spawn_mode": "role", "timesteps": 5_206_611},
]
"""
if "CURRICULUM_STAGES" not in constants_content:
    with open(constants_path, "a") as f:
        f.write(append_str)

# 2. Write src/curriculum.py
curriculum_code = """
import os
import argparse
import importlib

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
        print(f"\\n{'='*40}")
        print(f"Executing Stage {stage_idx} | Area: {stage_config['map_size'][0]}x{stage_config['map_size'][1]}")
        print(f"{'='*40}")

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
            log_dir=log_dir
        )

        # Update checkpoint for next stage
        load_ckpt_path = os.path.join(save_ckpt_dir, "model.pt")
        
        if not os.path.exists(load_ckpt_path):
            print(f"Error: Checkpoint not found at {load_ckpt_path}. Halting curriculum.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, required=True, choices=["mappo", "coop", "marc", "hmappo", "ecoop"])
    args = parser.parse_args()
    run_curriculum(args.algo)
"""
with open("src/curriculum.py", "w") as f:
    f.write(curriculum_code.strip())

# 3. Refactor all train scripts
train_files = [
    "src/train_mappo.py",
    "src/train_coop.py",
    "src/train_marc.py",
    "src/train_hmappo.py",
    "src/train_ecoop.py",
]

for tf in train_files:
    if not os.path.exists(tf):
        continue
    with open(tf, "r") as f:
        content = f.read()

    # 1. Update imports for constants
    content = re.sub(
        r"from constants import (.*?)\n",
        r"from constants import \1, LR, NUM_ENVS, NUM_STEPS, GAMMA, GAE_LAMBDA, UPDATE_EPOCHS, CLIP_COEF, MACRO_STEP, ECOOP_POOL_SIZE, ALPHA_ALARM, GAMMA_CAUSAL, AFFORDANCE_COEF\nimport os\nimport json\nimport logging\n",
        content,
    )

    # 2. Remove hardcoded Hyperparameters blocks
    content = re.sub(
        r"# Hyperparameters\n.*?(?=class |def |#)", "", content, flags=re.DOTALL
    )
    content = re.sub(
        r"# MARC Specific Hyperparameters\n.*?(?=class |def |#)",
        "",
        content,
        flags=re.DOTALL,
    )

    # 3. Modify def train() signature
    content = re.sub(
        r"def train\(\):",
        'def train(algo_name="test", stage_idx=0, env_config=None, total_timesteps=1000, load_ckpt_path=None, save_ckpt_dir=None, log_dir=None):',
        content,
    )

    # 4. Remove hardcoded env_config and TOTAL_TIMESTEPS logic
    content = re.sub(
        r"    env_config = {.*?}\n",
        '    if env_config is None:\n        env_config = {"map_size": (11, 11), "guard_count": 0, "camera_count": 0, "door_count": 0, "max_steps": 100, "spawn_mode": "role"}\n',
        content,
        flags=re.DOTALL,
    )

    # 5. Insert logging and loading
    init_str = """
    if log_dir is not None:
        logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO, format='%(asctime)s %(message)s', force=True)
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', force=True)
    
    logging.info(f"Training {algo_name} Stage {stage_idx} on {device}...")
"""
    content = re.sub(
        r'    print\(f"Training .*? on {device}\.\.\."\)\n', init_str, content
    )

    # Load ckpt
    load_str = """    if load_ckpt_path and os.path.exists(load_ckpt_path):
        agent.load_state_dict(torch.load(load_ckpt_path, map_location=device))
        logging.info(f"Loaded checkpoint from {load_ckpt_path}")
"""
    content = re.sub(
        r"(    optimizer = torch.optim.Adam\(agent.parameters\(\), lr=LR\)\n)",
        r"\1\n" + load_str,
        content,
    )

    # Convert print statements to logging.info
    content = re.sub(
        r'            print\(f"Update:', r'            logging.info(f"Update:', content
    )
    content = re.sub(
        r'            print\(f"Evolutionary',
        r'            logging.info(f"Evolutionary',
        content,
    )

    # Replace TOTAL_TIMESTEPS with total_timesteps
    content = re.sub(r"TOTAL_TIMESTEPS", "total_timesteps", content)

    # Save logic at the end
    # We find the end of the update loop and append save logic
    save_str = """
    # Save checkpoint and results
    if save_ckpt_dir:
        torch.save(agent.state_dict(), os.path.join(save_ckpt_dir, "model.pt"))
        results = {
            "algo": algo_name,
            "stage": stage_idx,
            "win_rate": float(win_rate) if 'win_rate' in locals() else 0.0,
            "mean_reward": float(avg_reward) if 'avg_reward' in locals() else 0.0
        }
        with open(os.path.join(save_ckpt_dir, "results.json"), "w") as jf:
            json.dump(results, jf, indent=4)
        logging.info(f"Saved checkpoint and results to {save_ckpt_dir}")
"""
    # Replace the if __name__ part to ensure save_str goes right before it
    content = re.sub(
        r'if __name__ == "__main__":',
        save_str + '\nif __name__ == "__main__":',
        content,
    )

    with open(tf, "w") as f:
        f.write(content)
