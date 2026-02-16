"""
Simulation loop: one world + two agents running simultaneously.

Both agents share the same world (same dynamics, same noise sequence)
but maintain independent latent state estimates and parameter estimates.
"""

import numpy as np
from world import World, observe, reward, S0, TRUE_THETA
from agent_base import AgentBase


class Simulation:
    """Manages a single world with two agents for comparison."""

    def __init__(self, agent_a: AgentBase, agent_b: AgentBase, seed: int = 42):
        self.world = World(seed=seed)
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.step_count = 0

        # Each agent has its own true latent state (independent trajectories)
        self.s_a = S0.copy()
        self.s_b = S0.copy()

    def reset(self, seed: int = 42):
        """Reset world and both agents."""
        self.world = World(seed=seed)
        self.agent_a.reset()
        self.agent_b.reset()
        self.s_a = S0.copy()
        self.s_b = S0.copy()
        self.step_count = 0

    def step(self) -> dict:
        """Advance one simulation step for both agents."""
        # --- Agent A ---
        obs_a = observe(self.s_a)
        r_a = reward(self.s_a)
        idx_a, action_a, scores_a = self.agent_a.select_action()
        # Save RNG state so Agent B gets the same noise realization
        rng_state = self.world.rng.bit_generator.state
        new_s_a = self.world.step(self.s_a, action_a)
        result_a = self.agent_a.step(self.s_a, action_a, new_s_a)
        self.s_a = new_s_a

        # --- Agent B (restore RNG for same noise) ---
        self.world.rng.bit_generator.state = rng_state
        obs_b = observe(self.s_b)
        r_b = reward(self.s_b)
        idx_b, action_b, scores_b = self.agent_b.select_action()
        new_s_b = self.world.step(self.s_b, action_b)
        result_b = self.agent_b.step(self.s_b, action_b, new_s_b)
        self.s_b = new_s_b

        self.step_count += 1

        return {
            'step': self.step_count,
            'agent_a': {
                'name': self.agent_a.name,
                'obs': observe(self.s_a).tolist(),
                'prev_obs': obs_a.tolist(),
                'true_s': self.s_a.tolist(),
                'reward': r_a,
                'action_idx': idx_a,
                'action_label': scores_a[idx_a]['label'],
                'scores': [{'label': s['label'], 'expected_reward': s['expected_reward']}
                           for s in scores_a],
                'theta': self.agent_a.theta.tolist(),
                'theta_error': result_a['theta_error'],
                'obs_error': result_a['err_mag'],
                'threat': self.agent_a.threat,
                'cumulative_reward': self.agent_a._cum_reward,
            },
            'agent_b': {
                'name': self.agent_b.name,
                'obs': observe(self.s_b).tolist(),
                'prev_obs': obs_b.tolist(),
                'true_s': self.s_b.tolist(),
                'reward': r_b,
                'action_idx': idx_b,
                'action_label': scores_b[idx_b]['label'],
                'scores': [{'label': s['label'], 'expected_reward': s['expected_reward']}
                           for s in scores_b],
                'theta': self.agent_b.theta.tolist(),
                'theta_error': result_b['theta_error'],
                'obs_error': result_b['err_mag'],
                'threat': self.agent_b.threat,
                'cumulative_reward': self.agent_b._cum_reward,
            },
            'true_theta': TRUE_THETA.tolist(),
        }

    def get_history(self) -> dict:
        """Return full history for both agents."""
        return {
            'agent_a': self.agent_a.history,
            'agent_b': self.agent_b.history,
        }
