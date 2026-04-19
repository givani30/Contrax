"""contrax._ekf — Extended Kalman filter."""

import jax
import jax.numpy as jnp
from jax import Array

from contrax._kalman import _measurement_flag, _symmetrize
from contrax.nonlinear import (
    DynamicsLike,
    ObservationFn,
    ObservationLike,
    _coerce_dynamics,
    _coerce_observation,
    _is_system_model,
    _system_dt,
)
from contrax.types import KalmanResult


def ekf_predict(
    model_or_f: DynamicsLike,
    x: Array,
    P: Array,
    u: Array,
    Q_noise: Array,
    *,
    t: Array | float = 0.0,
) -> tuple[Array, Array]:
    """Predict one extended Kalman filter step.

    Args:
        model_or_f: A `NonlinearSystem` or a plain dynamics function with
            signature `(t, x, u) -> x_next`. Both signatures are now the same,
            so `sys.dynamics` can be passed directly when using a system model.
        x: Filtered mean at the previous time step. Shape: `(n,)`.
        P: Filtered covariance at the previous time step. Shape: `(n, n)`.
        u: Current input. Shape: `(m,)`.
        Q_noise: Process noise covariance. Shape: `(n, n)`.
        t: Current sample time.

    Returns:
        Tuple `(x_pred, P_pred)` with the predicted mean and covariance.
    """
    f = _coerce_dynamics(model_or_f)
    x_pred = f(t, x, u)
    F = jax.jacfwd(f, argnums=1)(t, x, u)
    P_pred = _symmetrize(F @ P @ F.T + Q_noise)
    return x_pred, P_pred


def ekf_update(
    model_or_h: ObservationLike,
    x_pred: Array,
    P_pred: Array,
    y: Array,
    R_noise: Array,
    *,
    u: Array | None = None,
    t: Array | float = 0.0,
    has_measurement: Array | bool = True,
    num_iter: int = 1,
) -> tuple[Array, Array, Array]:
    """Update one extended Kalman filter step.

    Args:
        model_or_h: Nonlinear system model or observation function.
        x_pred: Predicted mean. Shape: `(n,)`.
        P_pred: Predicted covariance. Shape: `(n, n)`.
        y: Measurement vector. Shape: `(p,)`.
        R_noise: Measurement noise covariance. Shape: `(p, p)`.
        u: Current input used by the observation model. Shape: `(m,)`.
        t: Current sample time.
        has_measurement: Whether to apply the measurement update.
        num_iter: Number of iterated-EKF measurement linearization passes.
            `1` gives the ordinary EKF update.

    Returns:
        Tuple `(x, P, innovation)` with the filtered mean, filtered covariance,
        and final measurement residual. Missing-measurement steps return the
        prediction and a zero innovation.
    """
    if num_iter < 1:
        raise ValueError("num_iter must be at least 1")

    n = x_pred.shape[0]
    if u is None:
        u = jnp.zeros((0,), dtype=x_pred.dtype)
    h = _coerce_observation(
        model_or_h,
        None if _is_system_model(model_or_h) else model_or_h,
    )

    # Reference: Jazwinski (1970), "Stochastic Processes and Filtering
    # Theory"; the iterated EKF relinearizes the measurement model around the
    # current update iterate while keeping the prediction covariance fixed.
    def one_iteration(x_lin, _):
        H = jax.jacfwd(h, argnums=1)(t, x_lin, u)
        S = H @ P_pred @ H.T + R_noise
        K = jnp.linalg.solve(S.T, (P_pred @ H.T).T).T
        innov = y - h(t, x_lin, u) + H @ (x_lin - x_pred)
        x_next = x_pred + K @ innov
        return x_next, (x_next, H, K, innov)

    x_final, (_, Hs, Ks, innovations) = jax.lax.scan(
        one_iteration,
        x_pred,
        xs=None,
        length=num_iter,
    )
    H_final = Hs[-1]
    K_final = Ks[-1]
    innov_final = innovations[-1]
    P_final = _symmetrize((jnp.eye(n, dtype=P_pred.dtype) - K_final @ H_final) @ P_pred)

    def apply_update():
        return x_final, P_final, innov_final

    def skip_update():
        return x_pred, P_pred, jnp.zeros_like(y)

    return jax.lax.cond(
        _measurement_flag(has_measurement),
        apply_update,
        skip_update,
    )


def ekf_step(
    model_or_f: DynamicsLike,
    x: Array,
    P: Array,
    u: Array,
    y: Array,
    Q_noise: Array,
    R_noise: Array,
    *,
    t: Array | float = 0.0,
    has_measurement: Array | bool = True,
    num_iter: int = 1,
    observation: ObservationFn | None = None,
) -> tuple[Array, Array, Array]:
    """Run one predict-update step of an extended Kalman filter.

    Args:
        model_or_f: Nonlinear system model or dynamics function.
        x: Filtered mean at the previous time step. Shape: `(n,)`.
        P: Filtered covariance at the previous time step. Shape: `(n, n)`.
        u: Current input. Shape: `(m,)`.
        y: Measurement vector. Shape: `(p,)`.
        Q_noise: Process noise covariance. Shape: `(n, n)`.
        R_noise: Measurement noise covariance. Shape: `(p, p)`.
        has_measurement: Whether to apply the measurement update.
        num_iter: Number of iterated-EKF measurement linearization passes.
        observation: Observation function `h(x)` for plain dynamics callables.
            Omit this when passing a system model with its own output map.

    Returns:
        Tuple `(x, P, innovation)` after the step.
    """
    if _is_system_model(model_or_f):
        if observation is not None:
            raise ValueError("ekf_step(sys, ...) does not accept observation=.")
        obs = model_or_f
    else:
        if observation is None:
            raise ValueError(
                "ekf_step(f, ..., observation=...) requires an observation function."
            )
        obs = observation
    x_pred, P_pred = ekf_predict(model_or_f, x, P, u, Q_noise, t=t)
    return ekf_update(
        obs,
        x_pred,
        P_pred,
        y,
        R_noise,
        u=u,
        t=t,
        has_measurement=has_measurement,
        num_iter=num_iter,
    )


def ekf(
    model_or_f: DynamicsLike,
    Q_noise: Array,
    R_noise: Array,
    ys: Array,
    us: Array,
    x0: Array,
    P0: Array,
    *,
    observation: ObservationFn | None = None,
) -> KalmanResult:
    """Run an extended Kalman filter for nonlinear dynamics and observations.

    Uses automatic Jacobians of the transition and observation functions rather
    than requiring a hand-derived local linearization.

    This is the main nonlinear recursive estimation primitive in the current
    library. It works well with JAX-defined model functions, but the API is
    intentionally simple and does not yet expose multiple EKF variants.

    Args:
        model_or_f: Nonlinear system model or dynamics function.
        Q_noise: Process noise covariance. Shape: `(n, n)`.
        R_noise: Measurement noise covariance. Shape: `(p, p)`.
        ys: Measurement sequence. Shape: `(T, p)`.
        us: Input sequence. Shape: `(T, m)`.
        x0: Initial mean. Shape: `(n,)`.
        P0: Initial covariance. Shape: `(n, n)`.
        observation: Observation function `h(x)` for plain dynamics callables.
            Omit this when passing a system model with its own output map.

    Returns:
        [KalmanResult][contrax.types.KalmanResult]: A bundle containing
            filtered means, covariances, and innovations.

    Examples:
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> def f(t, x, u):
        ...     return jnp.array([x[0] + 0.1 * x[1], x[1] + u[0]])
        >>> def h(x):
        ...     return x[:1]
        >>> result = cx.ekf(
        ...     f,
        ...     Q_noise=1e-3 * jnp.eye(2),
        ...     R_noise=1e-2 * jnp.eye(1),
        ...     ys=jnp.zeros((10, 1)),
        ...     us=jnp.zeros((10, 1)),
        ...     x0=jnp.zeros(2),
        ...     P0=jnp.eye(2),
        ...     observation=h,
        ... )
    """
    if _is_system_model(model_or_f):
        if observation is not None:
            raise ValueError("ekf(sys, ...) does not accept observation=.")
        obs = model_or_f
    else:
        if observation is None:
            raise ValueError(
                "ekf(f, ..., observation=...) requires an observation function."
            )
        obs = observation

    dt = _system_dt(model_or_f)
    ts = jnp.arange(ys.shape[0], dtype=x0.dtype)
    if dt is not None:
        ts = ts * dt

    # Update-then-predict: x0/P0 is the prior on x_0.
    def update_predict(carry, inputs):
        x_prior, P_prior = carry
        y, u, t = inputs
        x_post, P_post, innov = ekf_update(
            obs,
            x_prior,
            P_prior,
            y,
            R_noise,
            u=u,
            t=t,
        )
        x_next, P_next = ekf_predict(model_or_f, x_post, P_post, u, Q_noise, t=t)
        return (x_next, P_next), (x_post, P_post, innov)

    _, (x_hats, Ps, innovations) = jax.lax.scan(update_predict, (x0, P0), (ys, us, ts))
    return KalmanResult(x_hat=x_hats, P=Ps, innovations=innovations)
