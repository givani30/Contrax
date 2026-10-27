"""contrax._ekf — Extended Kalman filter."""

import jax
import jax.numpy as jnp
from jax import Array

from contrax._filter_checks import checked, input_status, prepare_state
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
from contrax.types import FilterPredictionResult, FilterUpdateResult, KalmanResult


def _predict(model, x, P, Q, u, t):
    f = _coerce_dynamics(model)

    def value_and_status(state):
        if callable(getattr(model, "step", None)):
            step = model.step(t, state, u)
            value = jnp.where(step.successful, step.x, jnp.full_like(step.x, jnp.nan))
            code = step.status
        else:
            value = f(t, state, u)
            code = jnp.where(jnp.all(jnp.isfinite(value)), 0, 3)
        return value, (value, code)

    # has_aux reuses the transition evaluation for its value, Jacobian and
    # native integration status; AD still differentiates model data in closures.
    F, (x_pred, code) = jax.jacfwd(value_and_status, has_aux=True)(x)
    P_pred = _symmetrize(F @ P @ F.T + Q)
    return FilterPredictionResult(x_pred, P_pred, code)


def ekf_predict(
    model_or_f: DynamicsLike,
    x: Array,
    P: Array,
    *,
    Q: Array,
    u: Array | None = None,
    t: Array | float = 0.0,
    throw: bool = False,
) -> FilterPredictionResult:
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

        throw: Raise on numerical failure; otherwise inspect status.

    Returns:
        FilterPredictionResult: Mean x, covariance P, status and successful.
        Unpacks as (x_pred, P_pred).
    """
    x, P, Q = prepare_state(x, P, Q)
    u = zero_input(model_or_f, x.dtype) if u is None else u
    prediction = _predict(model_or_f, x, P, Q, u, t)
    valid = input_status(P, Q)
    code = jnp.where(valid != 0, valid, prediction.solver_status)
    result = FilterPredictionResult(prediction.x, prediction.P, code)
    return checked(result, throw, "EKF prediction failed; inspect status")


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
    throw: bool = False,
) -> FilterUpdateResult:
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

        throw: Raise on numerical failure; otherwise inspect status.

    Returns:
        FilterUpdateResult: Mean x, covariance P, innovation and status; unpacks
        as (x, P, innovation), with the filtered mean, filtered covariance,
        and final measurement residual. Missing-measurement steps return the
        prediction and a zero innovation.
    """
    x_pred, P_pred, R = prepare_state(x_pred, P_pred, R, measurement=y)
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

    x, P, innovation = jax.lax.cond(
        _measurement_flag(has_measurement),
        apply_update,
        skip_update,
    )
    result = FilterUpdateResult(x, P, innovation, input_status(P_pred, R))
    return checked(result, throw, "EKF update failed; inspect status")


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
    throw: bool = False,
) -> FilterUpdateResult:
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

        throw: Raise on numerical failure; otherwise inspect status.

    Returns:
        FilterUpdateResult: Mean x, covariance P, innovation, status and successful.
        Unpacks as (x, P, innovation).
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
    prediction = ekf_predict(model_or_f, x, P, u=u, Q=Q, t=t)
    x_pred, P_pred = prediction
    if measurement_time is None:
        dt = _system_dt(model_or_f)
        measurement_time = t if dt is None else t + dt
    update = ekf_update(
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
    code = jnp.where(
        prediction.solver_status != 0, prediction.solver_status, update.solver_status
    )
    result = FilterUpdateResult(update.x, update.P, update.innovation, code)
    return checked(result, throw, "EKF step failed; inspect status")


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
    throw: bool = False,
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

        throw: Raise on numerical failure; otherwise inspect status.

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
        x_prior, P_prior, code = carry
        y, u, t, available, k = inputs
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

        def advance():
            prediction = _predict(model_or_f, x_post, P_post, Q, u, t)
            return (
                prediction.x,
                prediction.P,
                prediction.solver_status.astype(code.dtype),
            )

        x_next, P_next, next_code = jax.lax.cond(
            k + 1 < ys.shape[0],
            advance,
            lambda: (x_post, P_post, jnp.zeros_like(code)),
        )
        code = jnp.where(code != 0, code, next_code)
        return (x_next, P_next, code), (x_post, P_post, innov)

    (_, _, code), (x_hats, Ps, innovations) = jax.lax.scan(
        update_predict,
        (x0, P0, input_status(P0, Q, R)),
        (ys, us, ts, flags, jnp.arange(ys.shape[0])),
    )
    result = KalmanResult(
        xs=x_hats,
        Ps=Ps,
        innovations=innovations,
        has_measurement=flags,
        solver_status=code,
    )
    return checked(result, throw, "EKF filter failed; inspect status")
