"""contrax._ekf — Extended Kalman filter."""

import jax
import jax.numpy as jnp
from jax import Array

from contrax._filter_common import prepare_filter, zero_input
from contrax._kalman import _covariance_update, _measurement_flag, _symmetrize
from contrax.nonlinear import (
    DynamicsLike,
    ObservationFn,
    ObservationLike,
    _coerce_dynamics,
    _coerce_observation,
    _is_system_model,
    _system_dt,
)
from contrax.solvers import EKF
from contrax.types import KalmanResult


def ekf_predict(
    model_or_f: DynamicsLike,
    x: Array,
    P: Array,
    *,
    Q: Array,
    u: Array | None = None,
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
        Q: Process noise covariance. Shape: `(n, n)`.
        t: Current sample time.

    Returns:
        Tuple `(x_pred, P_pred)` with the predicted mean and covariance.
    """
    u = zero_input(model_or_f, x.dtype) if u is None else u
    f = _coerce_dynamics(model_or_f)
    x_pred = f(t, x, u)
    F = jax.jacfwd(f, argnums=1)(t, x, u)
    P_pred = _symmetrize(F @ P @ F.T + Q)
    return x_pred, P_pred


def ekf_update(
    model_or_h: ObservationLike,
    x_pred: Array,
    P_pred: Array,
    y: Array,
    *,
    R: Array,
    u: Array | None = None,
    t: Array | float = 0.0,
    has_measurement: Array | bool = True,
    solver: EKF | None = None,
) -> tuple[Array, Array, Array]:
    """Update one extended Kalman filter step.

    Args:
        model_or_h: Nonlinear system model or observation function.
        x_pred: Predicted mean. Shape: `(n,)`.
        P_pred: Predicted covariance. Shape: `(n, n)`.
        y: Measurement vector. Shape: `(p,)`.
        R: Measurement noise covariance. Shape: `(p, p)`.
        u: Current input used by the observation model. Shape: `(m,)`.
        t: Current sample time.
        has_measurement: Whether to apply the measurement update.
        solver: Joseph covariance form and number of iterated linearizations.
            `EKF(num_iter=1)` gives the ordinary EKF update.

    Returns:
        Tuple `(x, P, innovation)` with the filtered mean, filtered covariance,
        and final measurement residual. Missing-measurement steps return the
        prediction and a zero innovation.
    """
    solver = EKF() if solver is None else solver
    if not isinstance(solver, EKF):
        raise TypeError("ekf_update solver must be contrax.solvers.EKF")

    if u is None:
        u = zero_input(model_or_h, x_pred.dtype)
    h = _coerce_observation(
        model_or_h,
        None if _is_system_model(model_or_h) else model_or_h,
    )

    # Reference: Jazwinski (1970), "Stochastic Processes and Filtering
    # Theory"; the iterated EKF relinearizes the measurement model around the
    # current update iterate while keeping the prediction covariance fixed.
    def one_iteration(x_lin, _):
        H = jax.jacfwd(h, argnums=1)(t, x_lin, u)
        if H.shape != (y.shape[0], x_pred.shape[0]):
            raise ValueError("The observation map must return a vector matching y")
        S = H @ P_pred @ H.T + R
        K = jnp.linalg.solve(S.T, (P_pred @ H.T).T).T
        observed = jnp.where(_measurement_flag(has_measurement), y, jnp.zeros_like(y))
        innov = observed - h(t, x_lin, u) + H @ (x_lin - x_pred)
        x_next = x_pred + K @ innov
        return x_next, (x_next, H, K, innov)

    def apply_update():
        x_final, (_, Hs, Ks, innovations) = jax.lax.scan(
            one_iteration,
            x_pred,
            xs=None,
            length=solver.num_iter,
        )
        P_final = _covariance_update(
            P_pred,
            Hs[-1],
            Ks[-1],
            R,
            solver.covariance_update,
        )
        return x_final, P_final, innovations[-1]

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
    y: Array,
    *,
    Q: Array,
    R: Array,
    u: Array | None = None,
    t: Array | float = 0.0,
    has_measurement: Array | bool = True,
    solver: EKF | None = None,
    observation: ObservationFn | None = None,
    u_measurement: Array | None = None,
    measurement_time: Array | float | None = None,
) -> tuple[Array, Array, Array]:
    """Run one predict-update step of an extended Kalman filter.

    Args:
        model_or_f: Nonlinear system model or dynamics function.
        x: Filtered mean at the previous time step. Shape: `(n,)`.
        P: Filtered covariance at the previous time step. Shape: `(n, n)`.
        u: Current input. Shape: `(m,)`.
        t: Transition start time.
        y: Measurement vector. Shape: `(p,)`.
        Q: Process noise covariance. Shape: `(n, n)`.
        R: Measurement noise covariance. Shape: `(p, p)`.
        has_measurement: Whether to apply the measurement update.
        solver: Covariance update and number of iterated measurement passes.
        observation: Observation function `h(x)` for plain dynamics callables.
            Omit this when passing a system model with its own output map.
        u_measurement: Input at the measurement time, defaulting to u.
        measurement_time: Observation time. Defaults to t plus the model's
            sample time when available, otherwise t.

    Returns:
        Tuple `(x, P, innovation)` after the step.
    """
    u = zero_input(model_or_f, x.dtype) if u is None else u
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
    x_pred, P_pred = ekf_predict(model_or_f, x, P, u=u, Q=Q, t=t)
    if measurement_time is None:
        dt = _system_dt(model_or_f)
        measurement_time = t if dt is None else t + dt
    return ekf_update(
        obs,
        x_pred,
        P_pred,
        y,
        R=R,
        u=u if u_measurement is None else u_measurement,
        t=measurement_time,
        has_measurement=has_measurement,
        solver=solver,
    )


def ekf(
    model_or_f: DynamicsLike,
    ys: Array,
    *,
    Q: Array,
    R: Array,
    us: Array | None = None,
    x0: Array | None = None,
    P0: Array | None = None,
    observation: ObservationFn | None = None,
    has_measurement: Array | None = None,
    solver: EKF | None = None,
) -> KalmanResult:
    """Run an extended Kalman filter for nonlinear dynamics and observations.

    Uses automatic Jacobians of the transition and observation functions rather
    than requiring a hand-derived local linearization.

    Joseph covariance updates and one measurement linearization are the defaults.
    Solver configuration selects compact covariance form or iterated updates.
    Supports `jit`, `vmap` and derivatives through JAX-defined model functions.

    Args:
        model_or_f: Nonlinear system model or dynamics function.
        Q: Process noise covariance. Shape: `(n, n)`.
        R: Measurement noise covariance. Shape: `(p, p)`.
        ys: Measurement sequence. Shape: `(T, p)`.
        us: Optional controls, shape `(T, m)` or `(T, 2, m)` for sampled FOH.
            Defaults to zero controls; declare model.input_dim or supply us for
            controlled nonlinear models.
        x0: Initial prior mean, shape `(n,)`. Defaults to zeros.
        P0: Initial prior covariance, shape `(n, n)`. Defaults to identity.
        observation: Observation function `h(x)` for plain dynamics callables.
            Omit this when passing a system model with its own output map.
        has_measurement: Whole-observation availability flags. Shape: `(T,)`.
            Missing observations may contain NaNs and return zero innovations.
        solver: Joseph covariance form and iterated measurement-update settings.

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
        ...     f, jnp.zeros((10, 1)), Q=1e-3 * jnp.eye(2), R=1e-2 * jnp.eye(1),
        ...     us=jnp.zeros((10, 1)), observation=h,
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

    ys, Q, R, us, x0, P0, flags = prepare_filter(
        model_or_f, ys, Q, R, us, x0, P0, has_measurement
    )
    dt = _system_dt(model_or_f)
    ts = jnp.arange(ys.shape[0], dtype=x0.dtype)
    if dt is not None:
        ts = ts * jnp.asarray(dt, dtype=x0.dtype)

    # Update-then-predict: x0/P0 is the prior on x_0.
    def update_predict(carry, inputs):
        x_prior, P_prior = carry
        y, u, t, available = inputs
        x_post, P_post, innov = ekf_update(
            obs,
            x_prior,
            P_prior,
            y,
            R=R,
            u=u,
            t=t,
            has_measurement=available,
            solver=solver,
        )
        x_next, P_next = ekf_predict(model_or_f, x_post, P_post, u=u, Q=Q, t=t)
        return (x_next, P_next), (x_post, P_post, innov)

    _, (x_hats, Ps, innovations) = jax.lax.scan(
        update_predict, (x0, P0), (ys, us, ts, flags)
    )
    return KalmanResult(
        xs=x_hats, Ps=Ps, innovations=innovations, has_measurement=flags
    )
