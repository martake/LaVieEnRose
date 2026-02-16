"""
Base agent class for La Vie En Rose comparison experiment.

Each agent independently estimates θ and selects actions to maximize reward.
"""

from abc import ABC, abstractmethod
import numpy as np
from world import (
    build_A, observe, reward, get_scaled_actions,
    P, B, P_pinv, S0, THETA0, ACTION_LABELS
)


class AgentBase(ABC):
    """Abstract base class for parameter-estimating agents."""

    def __init__(self, name: str, lr: float = 0.02, exploration_range: float = 1.0):
        self.name = name
        self.lr = lr
        self.base_lr = lr
        self.exploration_range = exploration_range
        self.reset()

    def reset(self):
        """Reset agent to initial conditions."""
        self.theta = THETA0.copy()
        self.s_hat = S0.copy()
        self.threat = 0.0
        self.lr = self.base_lr
        self.recent_rewards: list[float] = []
        self.recent_pred_errors: list[float] = []
        self.history: dict = {
            'theta_errors': [],
            'obs_errors': [],
            'rewards': [],
            'cumulative_reward': [],
        }
        self._cum_reward = 0.0

    def select_action(self) -> tuple[int, np.ndarray, list[dict]]:
        """Evaluate all candidate actions and return (chosen_idx, action, scores)."""
        actions = get_scaled_actions(self.exploration_range)
        scores = []
        for i, a in enumerate(actions):
            s_next = self._predict(a)
            expected_r = reward(s_next)
            scores.append({
                'label': ACTION_LABELS[i],
                'action': a,
                'expected_reward': expected_r,
            })

        best_idx = max(range(len(scores)), key=lambda i: scores[i]['expected_reward'])
        return best_idx, scores[best_idx]['action'], scores

    def _predict(self, action: np.ndarray) -> np.ndarray:
        """Predict next latent state using current estimates."""
        A = build_A(self.theta)
        return A @ self.s_hat + B @ action

    def step(self, true_s: np.ndarray, action: np.ndarray, new_s: np.ndarray):
        """Perform one learning step after the world has transitioned.

        Args:
            true_s: latent state before action (for reference, not used by agent)
            action: the action that was taken
            new_s: latent state after action (agent sees only observation)
        """
        new_obs = observe(new_s)

        # Predicted observation
        s_next_hat = self._predict(action)
        o_next_hat = observe(s_next_hat)
        error = new_obs - o_next_hat
        err_mag = float(np.linalg.norm(error))

        # Threat detection
        predicted_r = reward(s_next_hat)
        actual_r = reward(new_s)
        reward_pred_err = abs(actual_r - predicted_r)

        self.recent_rewards.append(actual_r)
        self.recent_pred_errors.append(reward_pred_err)
        win = 10
        if len(self.recent_rewards) > win:
            self.recent_rewards.pop(0)
        if len(self.recent_pred_errors) > win:
            self.recent_pred_errors.pop(0)

        avg_reward = np.mean(self.recent_rewards)
        avg_pred_err = np.mean(self.recent_pred_errors)

        low_reward_threat = max(0, 0.3 - avg_reward) / 0.8
        pred_err_threat = min(1.0, avg_pred_err / 0.8)
        self.threat = min(1.0, max(low_reward_threat, pred_err_threat))
        self.lr = self.base_lr * (1 + 4 * self.threat)

        # Delegate to subclass for parameter update
        self._learn(action, new_obs, s_next_hat, o_next_hat, error)

        # Update estimated latent state via pseudo-inverse correction
        ds = P_pinv @ error
        self.s_hat = s_next_hat + ds

        # Record history
        from world import TRUE_THETA
        theta_err = float(np.linalg.norm(self.theta - TRUE_THETA))
        self.history['theta_errors'].append(theta_err)
        self.history['obs_errors'].append(err_mag)
        self.history['rewards'].append(actual_r)
        self._cum_reward += actual_r
        self.history['cumulative_reward'].append(self._cum_reward)

        # Decay base learning rate
        self.base_lr = max(0.003, self.base_lr * 0.998)

        return {
            'error': error,
            'err_mag': err_mag,
            'predicted_obs': o_next_hat,
            'actual_reward': actual_r,
            'theta_error': theta_err,
        }

    @abstractmethod
    def _learn(self, action: np.ndarray, new_obs: np.ndarray,
               s_next_hat: np.ndarray, o_next_hat: np.ndarray,
               error: np.ndarray):
        """Update theta estimate. Implemented by subclasses."""
        ...
