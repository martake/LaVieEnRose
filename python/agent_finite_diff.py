"""Finite difference gradient descent agent."""

import numpy as np
from agent_base import AgentBase
from world import build_A, observe, B, P


class AgentFiniteDiff(AgentBase):
    """Estimate theta via central finite-difference gradient descent."""

    def __init__(self, lr: float = 0.02, eps: float = 0.001):
        self.eps = eps
        super().__init__(name="FiniteDiff", lr=lr)

    def _learn(self, action: np.ndarray, new_obs: np.ndarray,
               s_next_hat: np.ndarray, o_next_hat: np.ndarray,
               error: np.ndarray):
        """Numerical gradient descent on each theta component."""
        for k in range(3):
            # +eps perturbation
            theta_plus = self.theta.copy()
            theta_plus[k] += self.eps
            s_p = build_A(theta_plus) @ self.s_hat + B @ action
            err_plus = float(np.linalg.norm(new_obs - observe(s_p)))

            # -eps perturbation
            theta_minus = self.theta.copy()
            theta_minus[k] -= self.eps
            s_m = build_A(theta_minus) @ self.s_hat + B @ action
            err_minus = float(np.linalg.norm(new_obs - observe(s_m)))

            # Central difference gradient and update
            grad = (err_plus - err_minus) / (2 * self.eps)
            self.theta[k] -= self.lr * grad
