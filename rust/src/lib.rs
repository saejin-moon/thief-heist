pub mod constants;
pub mod types;
pub mod grid;
pub mod vision;
pub mod pathfinding;
pub mod env;
pub mod batch;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn};

use crate::batch::HeistBatchEnv;
use crate::types::EnvConfig;

#[pyclass]
pub struct PyHeistBatchEnv {
    inner: HeistBatchEnv,
}

#[pymethods]
impl PyHeistBatchEnv {
    #[new]
    #[pyo3(signature = (num_envs=16, config_json="{}".to_string(), base_seed=0))]
    pub fn new(num_envs: usize, config_json: String, base_seed: u64) -> PyResult<Self> {
        let config: EnvConfig = serde_json::from_str(&config_json)
            .unwrap_or_else(|_| EnvConfig::default());
        let inner = HeistBatchEnv::new(num_envs, config, base_seed);
        Ok(Self { inner })
    }

    pub fn state_dim(&self) -> usize {
        self.inner.state_dim()
    }

    pub fn reset(
        &mut self,
        _py: Python,
        base_seed: u64,
        mut out_obs: PyReadwriteArrayDyn<i32>,
        mut out_mask: PyReadwriteArrayDyn<i8>,
        mut out_role: PyReadwriteArrayDyn<i8>,
        mut out_goal: PyReadwriteArrayDyn<f32>,
        mut out_state: PyReadwriteArrayDyn<f32>,
    ) -> PyResult<()> {
        let obs_slice = out_obs.as_slice_mut()?;
        let mask_slice = out_mask.as_slice_mut()?;
        let role_slice = out_role.as_slice_mut()?;
        let goal_slice = out_goal.as_slice_mut()?;
        let state_slice = out_state.as_slice_mut()?;

        self.inner.reset_parallel(
            base_seed,
            obs_slice,
            mask_slice,
            role_slice,
            goal_slice,
            state_slice,
        );

        Ok(())
    }

    pub fn step(
        &mut self,
        py: Python,
        actions: PyReadonlyArrayDyn<i32>,
        mut out_obs: PyReadwriteArrayDyn<i32>,
        mut out_mask: PyReadwriteArrayDyn<i8>,
        mut out_role: PyReadwriteArrayDyn<i8>,
        mut out_goal: PyReadwriteArrayDyn<f32>,
        mut out_rewards: PyReadwriteArrayDyn<f32>,
        mut out_terms: PyReadwriteArrayDyn<bool>,
        mut out_truncs: PyReadwriteArrayDyn<bool>,
        mut out_state: PyReadwriteArrayDyn<f32>,
    ) -> PyResult<PyObject> {
        let act_slice = actions.as_slice()?;
        let obs_slice = out_obs.as_slice_mut()?;
        let mask_slice = out_mask.as_slice_mut()?;
        let role_slice = out_role.as_slice_mut()?;
        let goal_slice = out_goal.as_slice_mut()?;
        let rew_slice = out_rewards.as_slice_mut()?;
        let term_slice = out_terms.as_slice_mut()?;
        let trunc_slice = out_truncs.as_slice_mut()?;
        let state_slice = out_state.as_slice_mut()?;

        let step_infos = self.inner.step_parallel(
            act_slice,
            obs_slice,
            mask_slice,
            role_slice,
            goal_slice,
            rew_slice,
            term_slice,
            trunc_slice,
            state_slice,
        );

        let py_infos = PyList::empty(py);
        for info in step_infos {
            let env_dict = PyDict::new(py);
            for ag in &["scout", "hacker", "muscle", "extractor"] {
                let d = PyDict::new(py);
                d.set_item("win", info.win)?;
                d.set_item("alarm", info.alarm)?;
                d.set_item("steps", info.steps)?;
                d.set_item("scout_pois_tagged", info.scout_pois_tagged)?;
                d.set_item("scout_interact_success", info.scout_interact_success)?;
                d.set_item("hacker_hack_success", info.hacker_hack_success)?;
                d.set_item("muscle_neutralize_success", info.muscle_neutralize_success)?;
                d.set_item("muscle_guards_neutralized", info.muscle_guards_neutralized)?;
                d.set_item("extractor_loot_success", info.extractor_loot_success)?;
                d.set_item("agents_at_extract", info.agents_at_extract)?;
                d.set_item("terminal_pos", info.terminal_pos)?;
                d.set_item("loot_pos", info.loot_pos)?;
                d.set_item("extract_pos", info.extract_pos)?;

                let ag_idx = match *ag {
                    "scout" => 0,
                    "hacker" => 1,
                    "muscle" => 2,
                    _ => 3,
                };
                d.set_item("pos", info.agent_positions[ag_idx])?;
                env_dict.set_item(*ag, d)?;
            }
            py_infos.append(env_dict)?;
        }

        Ok(py_infos.into())
    }
}

#[pymodule]
fn heist_core_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHeistBatchEnv>()?;
    Ok(())
}
