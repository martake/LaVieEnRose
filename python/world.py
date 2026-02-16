"""
World model for La Vie En Rose adjoint method comparison.

Latent state s_t ∈ R^3 follows:
    s_{t+1} = A(θ) · s_t + B · a_t + ε_t

Observation is 2D projection:
    o_t = P · s_t
"""

import numpy as np

# --- True parameters ---
TRUE_THETA = np.array([0.15, 0.98, 0.12])  # [θ_r, θ_s, θ_d]

# Projection matrix P (2×3): latent 3D → observed 2D
P = np.array([
    [1.0, 0.0, 0.3],
    [0.0, 1.0, 0.5]
])

# Action lift matrix B (3×2): 2D action → 3D latent
B = np.array([
    [1.0,  0.0],
    [0.0,  1.0],
    [0.2, -0.1]
])

# Pseudo-inverse of P: P^+ = P^T (P P^T)^{-1}
P_pinv = np.linalg.pinv(P)

NOISE_STD = 0.02
OBS_BOUND = 2.5

# Initial conditions
S0 = np.array([1.0, 0.5, 0.0])
THETA0 = np.array([0.0, 1.0, 0.0])

# Candidate actions: 8 directions + stay
ACTION_LABELS = ['↑', '↗', '→', '↘', '↓', '↙', '←', '↖', '·']
_d = 0.707
BASE_ACTIONS = np.array([
    [0, -1], [_d, -_d], [1, 0], [_d, _d],
    [0,  1], [-_d, _d], [-1, 0], [-_d, -_d],
    [0,  0]
])


def build_A(theta: np.ndarray) -> np.ndarray:
    """Build 3×3 dynamics matrix A(θ)."""
    r, s, d = theta
    c, sn = np.cos(r), np.sin(r)
    return np.array([
        [s * c,  -s * sn, 0.0],
        [s * sn,  s * c,  0.0],
        [0.0,     0.0,    1.0 + 0.5 * d]
    ])


def dA_dtheta(theta: np.ndarray) -> list[np.ndarray]:
    """Analytical partial derivatives ∂A/∂θ_k for k=0,1,2 (r,s,d)."""
    r, s, _ = theta
    c, sn = np.cos(r), np.sin(r)

    dA_dr = np.array([
        [-s * sn, -s * c, 0.0],
        [ s * c,  -s * sn, 0.0],
        [ 0.0,     0.0,    0.0]
    ])

    dA_ds = np.array([
        [c,  -sn, 0.0],
        [sn,  c,  0.0],
        [0.0, 0.0, 0.0]
    ])

    dA_dd = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5]
    ])

    return [dA_dr, dA_ds, dA_dd]


def observe(s: np.ndarray) -> np.ndarray:
    """Project latent state to observation: o = P · s"""
    return P @ s


def reward(s: np.ndarray) -> float:
    """Reward function over latent state."""
    return float(np.sin(5 * s[2] + s[0]) * np.cos(3 * s[2] - s[1]))


def get_scaled_actions(exploration_range: float = 1.0) -> np.ndarray:
    """Scale base actions by exploration range."""
    return BASE_ACTIONS * (0.5 * exploration_range)


class World:
    """Shared world with true dynamics, noise generation, and deterministic seeding."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.true_theta = TRUE_THETA.copy()

    def step(self, s: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Advance latent state: s_{t+1} = A(θ*) · s_t + B · a_t + ε_t"""
        A = build_A(self.true_theta)
        noise = self.rng.normal(0, NOISE_STD, size=3)
        s_next = A @ s + B @ action + noise

        # Clamp so observed position stays within bounds
        obs = P @ s_next
        if np.abs(obs[0]) > OBS_BOUND or np.abs(obs[1]) > OBS_BOUND:
            scale = 1.0
            if np.abs(obs[0]) > OBS_BOUND:
                scale = min(scale, OBS_BOUND / np.abs(obs[0]))
            if np.abs(obs[1]) > OBS_BOUND:
                scale = min(scale, OBS_BOUND / np.abs(obs[1]))
            s_next *= scale

        return s_next
