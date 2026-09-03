"""
Global constants for the HEIST environment.
"""

import numpy as np

# Tile types (values observed by agents / used by the renderer)
FOG = -1  # Hidden behind fog of war
EMPTY = 0  # Walkable floor
WALL = 1  # Solid wall (blocks movement and line of sight)
TERMINAL = 2  # Security terminal - Hacker disables cameras/vault here
LOOT = 3  # The heist loot - Extractor must secure it
EXTRACT = 4  # Extraction point - All agents must end here with loot
GUARD = 5  # Rule-based adversary (dynamic entity)
ALLY = 6  # Other agent (dynamic entity)
CAMERA = 7  # Security camera (line-of-sight alarm source)
DOOR = 8  # Locked door (blocks movement; hacker can bypass)
WAYPOINT = 9  # Directional beacon for tagged objectives

# Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
WAIT = 4
INTERACT = 5

ACTION_DELTAS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
    WAIT: (0, 0),
}

# Agents
AGENTS = ["scout", "hacker", "muscle", "extractor"]
N_AGENTS = len(AGENTS)
AGENT_CHAR = {"scout": "S", "hacker": "H", "muscle": "M", "extractor": "E"}
ROLE_ONEHOT_ARRAYS = {
    a: np.array([1 if i == j else 0 for j in range(N_AGENTS)], dtype=np.int8)
    for i, a in enumerate(AGENTS)
}

# Observation / layout dimensions
MAP_SIZE = (50, 50)
OBSERVATION_SIZE = (7, 7)  # Agent local view window
ACTION_SPACE_SIZE = 6  # |A| for every agent (UP, DOWN, LEFT, RIGHT, WAIT, INTERACT)
TILE_SIZE = 20  # Renderer pixels per tile
SCOUT_VISION_RADIUS = 8
AGENT_VISION_RADIUS = 3
SCOUT_TAG_DISTANCE = 8  # Full line-of-sight stand-off tagging distance
MIN_GUARD_SPAWN_DIST = 6

# Reward structure
REWARD_WIN = 15.0
REWARD_LOSE = -10.0
REWARD_TASK = 2.0
REWARD_TAG = 1.0
REWARD_BYPASS = 1.0
REWARD_HACK_PROGRESS = 0.5
TOTAL_TIME_BLEED = -2.0
REWARD_TIME_BLEED = -0.005
CONVERGE_BONUS = 0.20
CONVERGE_RADIUS = 4
WIN_CONVERGE_RADIUS = 3

# Mechanics & Alarms
HACK_TURNS = 3
EXTRACTION_COUNTDOWN_RATIO = 0.5  # Countdown steps as a percentage of max_steps
ALARM_MAX = 100.0
ALARM_CAMERA = 0.10
ALARM_HACK_TURN = 1.0
ALARM_BYPASS = 3.0
ALARM_NEUTRALIZE = 5.0
ALARM_GUARD_SPOT = 10.0
ALARM_GUARD_CONTINUOUS = 1.5
ALARM_EXTRACTION_TIMEOUT = 15.0
CAMERA_RANGE = 12
CATCH_DISTANCE = 1
CONVERGE_ALARM = 50.0
GUARD_LOS_RANGE = 8
SEARCH_RADIUS = 5
SEARCH_TURNS = 6

# Renderer Palette
COLORS = {
    WALL: (20, 20, 30),
    EMPTY: (245, 245, 245),
    TERMINAL: (0, 100, 255),
    LOOT: (255, 215, 0),
    EXTRACT: (0, 200, 90),
    GUARD: (255, 60, 60),
    CAMERA: (120, 40, 180),
    DOOR: (160, 110, 40),
    "AGENT": {
        "scout": (0, 255, 255),
        "hacker": (150, 60, 220),
        "muscle": (180, 60, 60),
        "extractor": (255, 150, 40),
    },
    "EXPLORED": (0, 0, 0, 90),
}
# --- HYPERPARAMETERS ---
LR = 2.5e-4
NUM_ENVS = 8
NUM_STEPS = 125
GAMMA = 0.99
GAE_LAMBDA = 0.95
UPDATE_EPOCHS = 4
CLIP_COEF = 0.2

# Algorithm specific
COOP_NUM_EXPERTS = 2
MACRO_STEP = 5
ALPHA_ALARM = 1.5
GAMMA_CAUSAL = 0.95
AFFORDANCE_COEF = 0.5

# --- CURRICULUM STAGES ---
# Right-sized timesteps matching true empirical convergence horizons (3.2x faster training)
CURRICULUM_STAGES = [
    {
        "map_size": (11, 11),
        "guard_count": 0,
        "camera_count": 0,
        "door_count": 0,
        "max_steps": 300,
        "alarm_max": 100.0,
        "spawn_mode": "role",
        "timesteps": 200_000,
    },
    {
        "map_size": (17, 17),
        "guard_count": 1,
        "camera_count": 0,
        "door_count": 1,
        "max_steps": 400,
        "alarm_max": 100.0,
        "spawn_mode": "role",
        "timesteps": 200_000,
    },
    {
        "map_size": (25, 25),
        "guard_count": 2,
        "camera_count": 1,
        "door_count": 2,
        "max_steps": 900,
        "alarm_max": 125.0,
        "spawn_mode": "role",
        "timesteps": 500_000,
    },
    {
        "map_size": (35, 35),
        "guard_count": 3,
        "camera_count": 2,
        "door_count": 3,
        "max_steps": 2000,
        "alarm_max": 150.0,
        "spawn_mode": "role",
        "timesteps": 900_000,
    },
    {
        "map_size": (50, 50),
        "guard_count": 4,
        "camera_count": 3,
        "door_count": 4,
        "max_steps": 5000,
        "alarm_max": 175.0,
        "spawn_mode": "role",
        "timesteps": 1_600_000,
    },
]

# Further RL & Training Constants
ENT_COEF = 0.01
VF_COEF = 0.5
WIN_REWARD_THRESHOLD = 5.0

# Hardware Thermal Protection (Celsius)
MAX_GPU_TEMP = 85.0
MAX_CPU_TEMP = 85.0

# E-COOP Specific (Aligned with THIEF for fair head-to-head benchmarking)
ECOOP_NUM_ENVS = 16
ECOOP_EVOLUTION_INTERVAL = 125
ECOOP_GRACE_UPDATES = 20
ECOOP_RAMP_UPDATES = 15
ECOOP_CULL_WINDOW_UPDATES = (
    20  # Consecutive 0% usage updates before an expert goes extinct
)
ECOOP_MUTATION_NOISE = 0.03  # Calibrated micro-exploration along flat Fisher manifolds
ECOOP_CROSSOVER_DAMPING = 1e-4
ECOOP_MUTANT_ENVS = (
    4  # 4 dedicated exploration envs during burn-in grace period (25% of 16 envs)
)
ECOOP_HYSTERESIS_EPSILON = (
    0.05  # Calibrated threshold to enforce temporal policy coherence
)
ECOOP_PROGRESS_COEF = 0.05  # Potential-based sub-goal progress shaping coefficient

# MAHIRO / H-MAPPO Specific
MAHIRO_INTRINSIC_REWARD_COEF = 0.015

# --- THIEF ALGORITHM CONSTANTS ---
THIEF_NUM_ENVS = 16
THIEF_INITIAL_EXPERTS = 1
THIEF_MAX_EXPERTS = 8
THIEF_MAX_SANDBOX_EXPERTS = 1  # 1 dedicated specialist in sandbox
THIEF_ENVS_PER_MUTANT = 8  # 8 dedicated environments for the child specialist (50% of cluster)
THIEF_WARMUP_UPDATES = 20
THIEF_POST_GRACE_COOLDOWN_UPDATES = 30
THIEF_ISOLATION_UPDATES = 20
THIEF_TARGETED_LR = 0.08
THIEF_GRADIENT_CONFLICT_THRESHOLD = -0.20  # Trigger when cos(g+, g-) < -0.20
THIEF_DEFICIT_MIN_SAMPLES = 32  # Minimum failure transition count
THIEF_HYSTERESIS_EPSILON = 0.05
THIEF_TARGET_VECTOR_DIM = 2  # 2D continuous unit orientation vector (ux, uy)
THIEF_HER_GOAL_DIM = 2  # Alias for backward compatibility
GOAL_VECTOR_DIM = 2  # 2D continuous unit orientation vector (ux, uy) for all MARL algos
THIEF_PROGRESS_COEF = (
    0.05  # Dense potential-based sub-goal progress shaping coefficient
)
THIEF_DORMANCY_WINDOW = (
    20  # Consecutive zero-usage updates before specialist enters dormancy
)
THIEF_CULL_WINDOW_UPDATES = 20
THIEF_MACRO_HORIZON = (
    5  # Sub-goal duration (K steps per macro action, 125 / 5 = 25 macro steps)
)
THIEF_HER_REWARD_COEF = 0.05  # Intrinsic distance reduction reward weight
THIEF_HER_AUX_COEF = 0.10  # Weight of auxiliary HER hindsight navigation policy loss
THIEF_HER_REACH_DIST = 1.5  # Distance threshold in tiles to consider sub-goal reached
