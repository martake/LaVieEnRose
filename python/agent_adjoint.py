"""Analytical gradient agent using direct sensitivity method.

Computes exact analytical gradients of the observation error with respect to
theta, using ∂A/∂θ_k. This is equivalent to what finite differences approximates,
but without the numerical perturbation (no extra forward evaluations needed).

The key advantage: exact gradient computation in O(N) instead of O(2N) evaluations.
"""

import numpy as np
from agent_base import AgentBase
from world import build_A, dA_dtheta, observe, P, B


class AgentAdjoint(AgentBase):
    """Estimate theta via analytical (direct sensitivity) gradient computation.

    For each step, computes the exact gradient:
        ∂||e||/∂θ_k = -(e^T · P · (∂A/∂θ_k · s_hat)) / ||e||

    This is the analytical equivalent of the central-difference approximation
    used by AgentFiniteDiff. Same loss function (||e||), same learning rate,
    same update rule — only the gradient computation method differs.
    """

    def __init__(self, lr: float = 0.02, window_size: int = 10):
        self.window_size = window_size  # kept for UI slider compatibility
        super().__init__(name="Adjoint", lr=lr)

    def _learn(self, action: np.ndarray, new_obs: np.ndarray,
               s_next_hat: np.ndarray, o_next_hat: np.ndarray,
               error: np.ndarray):
        err_norm = float(np.linalg.norm(error))
        if err_norm < 1e-10:
            return

        # Analytical gradient of ||e|| w.r.t. θ_k:
        #   ∂||e||²/∂θ_k = -2 * e^T * P * (∂A/∂θ_k * s_hat)
        #   ∂||e||/∂θ_k  = ∂||e||²/∂θ_k / (2 * ||e||)
        #                 = -(e^T * P * (∂A/∂θ_k * s_hat)) / ||e||
        #
        # This matches what finite differences computes:
        #   (||e+|| - ||e-||) / (2ε) ≈ ∂||e||/∂θ_k
        dA_list = dA_dtheta(self.theta)
        for k in range(3):
            grad_k = -(error @ P @ (dA_list[k] @ self.s_hat)) / err_norm
            self.theta[k] -= self.lr * grad_k
