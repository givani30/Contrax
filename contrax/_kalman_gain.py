"""Steady-state linear estimator design through the dual Riccati equation."""

import jax.numpy as jnp
from jax import Array

from contrax._riccati import dare
from contrax.core import DiscLTI
from contrax.solvers import DARE
from contrax.types import KalmanGainResult


def kalman_gain(
    sys: DiscLTI,
    Q: Array,
    R: Array,
    *,
    solver: DARE | None = None,
    throw: bool = False,
) -> KalmanGainResult:
    """Design a steady-state Kalman filter gain for a discrete LTI system.

    This is the estimator dual of discrete LQR. It solves the DARE for
    `(A.T, C.T, Q, R)` and returns the measurement-update gain used
    by `kalman_update()`.

    Args:
        sys: Discrete-time linear system.
        Q: Process noise covariance. Shape: `(n, n)`.
        R: Measurement noise covariance. Shape: `(p, p)`.
        solver: Dual DARE solver configuration.
        throw: Raise on failed Riccati design.

    Returns:
        [KalmanGainResult][contrax.types.KalmanGainResult]: A bundle with
            update gain `K`, steady-state predicted covariance `P`, and
            estimator error-dynamics poles.

    Examples:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> sys = cx.dss(
        ...     jnp.array([[1.0, 0.1], [0.0, 1.0]]),
        ...     jnp.array([[0.0], [0.1]]),
        ...     jnp.eye(2),
        ...     jnp.zeros((2, 1)),
        ...     dt=0.1,
        ... )
        >>> design = cx.kalman_gain(sys, 0.001 * jnp.eye(2), 0.01 * jnp.eye(2))
    """
    if not isinstance(sys, DiscLTI):
        raise TypeError("kalman_gain requires a discrete LTI system")
    design = dare(sys.A.T, sys.C.T, Q, R, solver=solver, throw=throw)
    P_pred = design.S
    innovation_cov = sys.C @ P_pred @ sys.C.T + R
    K = jnp.linalg.solve(innovation_cov.T, (P_pred @ sys.C.T).T).T
    closed_loop = sys.A - sys.A @ K @ sys.C
    return KalmanGainResult(
        K=K,
        P=P_pred,
        closed_loop=closed_loop,
        status=design.status,
        residual_norm=design.residual_norm,
        num_steps=design.num_steps,
    )
