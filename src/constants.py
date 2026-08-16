"""
Global constants for the HEIST environment.
"""
import numpy as np

# Tile types (values observed by agents / used by the renderer)
FOG = -1       # Hidden behind fog of war
EMPTY = 0      # Walkable floor
WALL = 1       # Solid wall (blocks movement and line of sight)
TERMINAL = 2   # Security terminal - Hacker disables cameras/vault here
LOOT = 3       # The heist loot - Extractor must secure it
EXTRACT = 4    # Extraction point - All agents must end here with loot
GUARD = 5      # Rule-based adversary (dynamic entity)
ALLY = 6       # Other agent (dynamic entity)
CAMERA = 7     # Security camera (line-of-sight alarm source)
DOOR = 8       # Locked door (blocks movement; hacker can bypass)
WAYPOINT = 9   # Directional beacon for tagged objectives

# Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
WAIT = 4
INTERACT = 5
BREACH = 6

ACTION_DELTAS = {
    UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1), WAIT: (0, 0),
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
ACTION_SPACE_SIZE = 7      # |A| for every agent
TILE_SIZE = 20             # Renderer pixels per tile
SCOUT_VISION_RADIUS = 8    
AGENT_VISION_RADIUS = 3    

# Reward structure
REWARD_WIN = 15.0
REWARD_LOSE = -10.0
REWARD_TASK = 2.0          
REWARD_TAG = 1.0           
REWARD_TIME_BLEED = -0.01  
CONVERGE_BONUS = 1.0       
CONVERGE_RADIUS = 4        
WIN_CONVERGE_RADIUS = 3    

# Mechanics & Alarms
HACK_TURNS = 3
EXTRACTION_COUNTDOWN = 60
ALARM_MAX = 100.0
ALARM_CAMERA = 0.20
ALARM_HACK_TURN = 2.0
ALARM_BYPASS = 6.0
ALARM_NEUTRALIZE = 10.0
ALARM_BREACH = 10.0
ALARM_GUARD_SPOT = 25.0
ALARM_EXTRACTION_TIMEOUT = 25.0
CAMERA_RANGE = 12
CATCH_DISTANCE = 1
CONVERGE_ALARM = 50.0
NEUTRALIZE_TURNS = 8
GUARD_LOS_RANGE = 8
SEARCH_RADIUS = 5
SEARCH_TURNS = 6

# Renderer Palette
COLORS = {
    WALL: (20, 20, 30), EMPTY: (245, 245, 245), TERMINAL: (0, 100, 255),
    LOOT: (255, 215, 0), EXTRACT: (0, 200, 90), GUARD: (255, 60, 60),
    CAMERA: (120, 40, 180), DOOR: (160, 110, 40),
    "AGENT": {"scout": (0, 255, 255), "hacker": (150, 60, 220), "muscle": (180, 60, 60), "extractor": (255, 150, 40)},
    "EXPLORED": (0, 0, 0, 90),
}