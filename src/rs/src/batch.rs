use rayon::prelude::*;
use crate::constants::*;
use crate::env::{HeistEnv, StepInfo};
use crate::types::EnvConfig;

pub struct HeistBatchEnv {
    pub envs: Vec<HeistEnv>,
    pub num_envs: usize,
    pub config: EnvConfig,
}

impl HeistBatchEnv {
    pub fn new(num_envs: usize, config: EnvConfig, base_seed: u64) -> Self {
        let envs: Vec<HeistEnv> = (0..num_envs)
            .map(|i| HeistEnv::new(config.clone(), base_seed + (i as u64) * 1000 + 1))
            .collect();

        Self {
            envs,
            num_envs,
            config,
        }
    }

    pub fn state_dim(&self) -> usize {
        STATE_DIM
    }

    pub fn reset_parallel(
        &mut self,
        base_seed: u64,
        out_obs: &mut [i32],    // [N_AGENTS, NUM_ENVS, 7, 7]
        out_mask: &mut [i8],    // [N_AGENTS, NUM_ENVS, 6]
        out_role: &mut [i8],    // [N_AGENTS, NUM_ENVS, 4]
        out_goal: &mut [f32],   // [N_AGENTS, NUM_ENVS, 2]
        out_state: &mut [f32],  // [NUM_ENVS, STATE_DIM]
    ) {
        let num_envs = self.num_envs;

        // Initialize role one-hot array
        for ag_idx in 0..N_AGENTS {
            for e in 0..num_envs {
                let role_offset = (ag_idx * num_envs + e) * N_AGENTS;
                for j in 0..N_AGENTS {
                    out_role[role_offset + j] = if ag_idx == j { 1 } else { 0 };
                }
            }
        }

        self.envs.par_iter_mut().enumerate().for_each(|(e, env)| {
            env.reset(Some(base_seed + (e as u64) * 1000 + 1));
        });

        // Write outputs per env
        for (e, env) in self.envs.iter().enumerate() {
            let mut temp_obs = [0i32; N_AGENTS * OBSERVATION_DIM * OBSERVATION_DIM];
            let mut temp_goals = [0f32; N_AGENTS * 2];
            let mut temp_masks = [0i8; N_AGENTS * ACTION_SPACE_SIZE];

            env.compute_observations_and_goals(&mut temp_obs, &mut temp_goals);
            env.compute_action_masks(&mut temp_masks);

            for ag_idx in 0..N_AGENTS {
                // Obs copy
                let obs_src = &temp_obs[ag_idx * 49..(ag_idx + 1) * 49];
                let obs_dest_start = (ag_idx * num_envs + e) * 49;
                out_obs[obs_dest_start..obs_dest_start + 49].copy_from_slice(obs_src);

                // Mask copy
                let mask_src = &temp_masks[ag_idx * 6..(ag_idx + 1) * 6];
                let mask_dest_start = (ag_idx * num_envs + e) * 6;
                out_mask[mask_dest_start..mask_dest_start + 6].copy_from_slice(mask_src);

                // Goal copy
                let goal_src = &temp_goals[ag_idx * 2..(ag_idx + 1) * 2];
                let goal_dest_start = (ag_idx * num_envs + e) * 2;
                out_goal[goal_dest_start..goal_dest_start + 2].copy_from_slice(goal_src);
            }

            let state_dest = &mut out_state[e * STATE_DIM..(e + 1) * STATE_DIM];
            env.write_state(state_dest);
        }
    }

    pub fn step_parallel(
        &mut self,
        actions: &[i32],        // [N_AGENTS, NUM_ENVS]
        out_obs: &mut [i32],    // [N_AGENTS, NUM_ENVS, 7, 7]
        out_mask: &mut [i8],    // [N_AGENTS, NUM_ENVS, 6]
        out_role: &mut [i8],    // [N_AGENTS, NUM_ENVS, 4]
        out_goal: &mut [f32],   // [N_AGENTS, NUM_ENVS, 2]
        out_rewards: &mut [f32],// [N_AGENTS, NUM_ENVS]
        out_terms: &mut [bool], // [N_AGENTS, NUM_ENVS]
        out_truncs: &mut [bool],// [N_AGENTS, NUM_ENVS]
        out_state: &mut [f32],  // [NUM_ENVS, STATE_DIM]
    ) -> Vec<StepInfo> {
        let num_envs = self.num_envs;

        // Ensure role buffer is filled
        for ag_idx in 0..N_AGENTS {
            for e in 0..num_envs {
                let role_offset = (ag_idx * num_envs + e) * N_AGENTS;
                for j in 0..N_AGENTS {
                    out_role[role_offset + j] = if ag_idx == j { 1 } else { 0 };
                }
            }
        }

        // Step all environments in parallel with rayon
        let step_results: Vec<(StepInfo, [f32; N_AGENTS], [bool; N_AGENTS], [bool; N_AGENTS])> = self
            .envs
            .par_iter_mut()
            .enumerate()
            .map(|(e, env)| {
                let env_acts = [
                    actions[0 * num_envs + e],
                    actions[1 * num_envs + e],
                    actions[2 * num_envs + e],
                    actions[3 * num_envs + e],
                ];

                let mut r = [0.0f32; N_AGENTS];
                let mut t = [false; N_AGENTS];
                let mut tr = [false; N_AGENTS];

                let info = env.step(env_acts, &mut r, &mut t, &mut tr);

                // If episode terminated or truncated, auto-reset the env
                if t[0] || tr[0] {
                    env.reset(None);
                }

                (info, r, t, tr)
            })
            .collect();

        // Write step outputs into contiguous buffers
        let mut infos = Vec::with_capacity(num_envs);

        for (e, (info, r, t, tr)) in step_results.into_iter().enumerate() {
            for ag_idx in 0..N_AGENTS {
                out_rewards[ag_idx * num_envs + e] = r[ag_idx];
                out_terms[ag_idx * num_envs + e] = t[ag_idx];
                out_truncs[ag_idx * num_envs + e] = tr[ag_idx];
            }
            infos.push(info);

            // Compute next observation and state (from auto-reset or ongoing step)
            let env = &self.envs[e];
            let mut temp_obs = [0i32; N_AGENTS * OBSERVATION_DIM * OBSERVATION_DIM];
            let mut temp_goals = [0f32; N_AGENTS * 2];
            let mut temp_masks = [0i8; N_AGENTS * ACTION_SPACE_SIZE];

            env.compute_observations_and_goals(&mut temp_obs, &mut temp_goals);
            env.compute_action_masks(&mut temp_masks);

            for ag_idx in 0..N_AGENTS {
                let obs_src = &temp_obs[ag_idx * 49..(ag_idx + 1) * 49];
                let obs_dest_start = (ag_idx * num_envs + e) * 49;
                out_obs[obs_dest_start..obs_dest_start + 49].copy_from_slice(obs_src);

                let mask_src = &temp_masks[ag_idx * 6..(ag_idx + 1) * 6];
                let mask_dest_start = (ag_idx * num_envs + e) * 6;
                out_mask[mask_dest_start..mask_dest_start + 6].copy_from_slice(mask_src);

                let goal_src = &temp_goals[ag_idx * 2..(ag_idx + 1) * 2];
                let goal_dest_start = (ag_idx * num_envs + e) * 2;
                out_goal[goal_dest_start..goal_dest_start + 2].copy_from_slice(goal_src);
            }

            let state_dest = &mut out_state[e * STATE_DIM..(e + 1) * STATE_DIM];
            env.write_state(state_dest);
        }

        infos
    }
}
