pub const FOG: i32 = -1;
pub const EMPTY: i32 = 0;
pub const WALL: i32 = 1;
pub const TERMINAL: i32 = 2;
pub const LOOT: i32 = 3;
pub const EXTRACT: i32 = 4;
pub const GUARD: i32 = 5;
pub const ALLY: i32 = 6;
pub const CAMERA: i32 = 7;
pub const DOOR: i32 = 8;
pub const WAYPOINT: i32 = 9;

pub const UP: i32 = 0;
pub const DOWN: i32 = 1;
pub const LEFT: i32 = 2;
pub const RIGHT: i32 = 3;
pub const WAIT: i32 = 4;
pub const INTERACT: i32 = 5;

pub const ACTION_DELTAS: [(i32, i32); 5] = [
    (-1, 0), // UP
    (1, 0),  // DOWN
    (0, -1), // LEFT
    (0, 1),  // RIGHT
    (0, 0),  // WAIT
];

pub const N_AGENTS: usize = 4;
pub const AGENT_NAMES: [&str; N_AGENTS] = ["scout", "hacker", "muscle", "extractor"];

pub const OBSERVATION_DIM: usize = 7;
pub const OBSERVATION_PAD: usize = 3; // 7 // 2
pub const ACTION_SPACE_SIZE: usize = 6;
pub const GOAL_VECTOR_DIM: usize = 2;

pub const SCOUT_VISION_RADIUS: i32 = 8;
pub const AGENT_VISION_RADIUS: i32 = 3;
pub const SCOUT_TAG_DISTANCE: i32 = 8;
pub const MIN_GUARD_SPAWN_DIST: i32 = 6;

pub const REWARD_WIN: f32 = 15.0;
pub const REWARD_LOSE: f32 = -10.0;
pub const REWARD_TASK: f32 = 2.0;
pub const REWARD_TAG: f32 = 1.0;
pub const REWARD_BYPASS: f32 = 1.0;
pub const REWARD_HACK_PROGRESS: f32 = 0.5;
pub const TOTAL_TIME_BLEED: f32 = -2.0;
pub const CONVERGE_BONUS: f32 = 0.20;
pub const WIN_CONVERGE_RADIUS: i32 = 3;

pub const HACK_TURNS: i32 = 3;
pub const EXTRACTION_COUNTDOWN_RATIO: f32 = 0.5;
pub const ALARM_MAX: f32 = 100.0;
pub const ALARM_CAMERA: f32 = 0.10;
pub const ALARM_HACK_TURN: f32 = 1.0;
pub const ALARM_BYPASS: f32 = 3.0;
pub const ALARM_NEUTRALIZE: f32 = 5.0;
pub const ALARM_GUARD_SPOT: f32 = 10.0;
pub const ALARM_GUARD_CONTINUOUS: f32 = 1.5;
pub const ALARM_EXTRACTION_TIMEOUT: f32 = 15.0;
pub const CAMERA_RANGE: i32 = 12;
pub const CATCH_DISTANCE: i32 = 1;
pub const CONVERGE_ALARM: f32 = 50.0;
pub const GUARD_LOS_RANGE: i32 = 8;
pub const SEARCH_RADIUS: i32 = 5;
pub const SEARCH_TURNS: i32 = 6;

pub const MAX_MAP_H: usize = 50;
pub const MAX_MAP_W: usize = 50;
pub const STATE_DIM: usize = 6 + (MAX_MAP_H * MAX_MAP_W) + (N_AGENTS * 2) + (12 * 2) + 12;
