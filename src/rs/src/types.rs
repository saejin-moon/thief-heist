use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnvConfig {
    #[serde(default = "default_map_size")]
    pub map_size: (usize, usize),
    #[serde(default = "default_num_rooms_range")]
    pub num_rooms_range: (usize, usize),
    #[serde(default = "default_guard_count")]
    pub guard_count: usize,
    #[serde(default = "default_camera_count")]
    pub camera_count: usize,
    #[serde(default = "default_door_count")]
    pub door_count: usize,
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    #[serde(default = "default_alarm_max")]
    pub alarm_max: f32,
    #[serde(default = "default_spawn_mode")]
    pub spawn_mode: String,
}

fn default_map_size() -> (usize, usize) {
    (50, 50)
}
fn default_num_rooms_range() -> (usize, usize) {
    (8, 12)
}
fn default_guard_count() -> usize {
    4
}
fn default_camera_count() -> usize {
    3
}
fn default_door_count() -> usize {
    4
}
fn default_max_steps() -> usize {
    300
}
fn default_alarm_max() -> f32 {
    100.0
}
fn default_spawn_mode() -> String {
    "role".to_string()
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            map_size: default_map_size(),
            num_rooms_range: default_num_rooms_range(),
            guard_count: default_guard_count(),
            camera_count: default_camera_count(),
            door_count: default_door_count(),
            max_steps: default_max_steps(),
            alarm_max: default_alarm_max(),
            spawn_mode: default_spawn_mode(),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum GuardState {
    Patrol,
    Search,
    Converge,
}

#[inline]
pub fn manhattan(a: (i32, i32), b: (i32, i32)) -> i32 {
    (a.0 - b.0).abs() + (a.1 - b.1).abs()
}
