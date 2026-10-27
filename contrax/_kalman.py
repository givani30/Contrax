"""Linear Kalman filtering and online prediction/update helpers."""

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import cho_solve

from contrax._filter_checks import checked, input_status, prepare_state
from contrax._filter_common import prepare_filter
from contrax._linear_filter_options import (
    cached_filter,
    cached_update,
    information_update,
    noise_status,
)
from contrax.core import DiscLTI
from contrax.solvers import InformationKalman, Kalman, SteadyStateKalman
from contrax.types import (
    FilterPredictionResult,
    FilterUpdateResult,
    KalmanResult,
)


def _symmetrize(X: Array) -> Array:
    return (X + X.T) / 2


def _zero_input(sys: DiscLTI, x: Array) -> Array:
    return jnp.zeros(sys.B.shape[1], dtype=x.dtype)


def _measurement_flag(has_measurement: Array | bool) -> Array:
    return jnp.asarray(has_measurement, dtype=bool)


def _covariance_update(P, H, K, R, method):
    # Reference: Joseph & Bucy (1968), Filtering for Stochastic Processes with
    # Applications to Guidance. No eigenvalue clipping or covariance repair.
    I_KH = jnp.eye(P.shape[0], dtype=P.dtype) - K @ H
    if method == "joseph":
        return _symmetrize(I_KH @ P @ I_KH.T + K @ R @ K.T)
    return _symmetrize(I_KH @ P)


def kalman_predict(
    sys: DiscLTI,
    x: Array,
    P: Array,
    *,
    Q: Array,
    u: Array | None = None,
    throw: bool = False,
) -> FilterPredictionResult:
    """Predict one linear Kalman filter step.

    Args:
        sys: Discrete-time linear system.
        x: Filtered mean at the previous time step. Shape: `(n,)`.
        P: Filtered covariance at the previous time step. Shape: `(n, n)`.
        Q: Process noise covariance. Shape: `(n, n)`.
        u: Optional input for the transition. Shape: `(m,)`. If omitted, the
            input term is zero.

        throw: Raise on numerical failure; otherwise inspect status.

    Returns:
        FilterPredictionResult: Mean x, covariance P, status and successful.
    """
    x, P, Q = prepare_state(x, P, Q)
    u = _zero_input(sys, x) if u is None else u
    x_pred = sys.A @ x + sys.B @ u
    P_pred = _symmetrize(sys.A @ P @ sys.A.T + Q)
    result = FilterPredictionResult(x_pred, P_pred, input_status(P, Q))
    return checked(result, throw, "Kalman prediction failed; inspect status")


def kalman_update(
    sys: DiscLTI,
    x_pred: Array,
    P_pred: Array,
    y: Array,
    *,
    R: Array,
    u: Array | None = None,
    has_measurement: Array | bool = True,
    solver: Kalman | SteadyStateKalman | InformationKalman | None = None,
    throw: bool = False,
) -> FilterUpdateResult:
    """Update one linear Kalman filter step.

    When `has_measurement=False`, returns the predicted state and covariance
    with a zero innovation. The flag may be a JAX scalar boolean, making
    this helper usable inside `jax.lax.scan` for streams with missing samples.

    Args:
        sys: Discrete-time linear system.
        x_pred: Predicted mean. Shape: `(n,)`.
        P_pred: Predicted covariance. Shape: `(n, n)`.
        y: Measurement vector. Shape: `(p,)`.
        R: Measurement noise covariance, shape `(p, p)`, or a positive variance
            vector `(p,)` with InformationKalman.
        u: Optional input for direct-feedthrough measurement models. Shape:
            `(m,)`. If omitted, the feedthrough term is zero.
        has_measurement: Whether to apply the measurement update.
        solver: Kalman (Joseph by default), InformationKalman for diagonal R,
            or SteadyStateKalman for an explicit cached design. Cached mode
            requires P_pred to match design.P and an available observation.

        throw: Raise on numerical failure; otherwise inspect status.

    Returns:
        FilterUpdateResult: Mean x, covariance P, innovation and status; unpacks
        as (x, P, innovation), with the filtered mean, filtered covariance,
        and measurement residual. Missing-measurement steps return a zero
        innovation with the same shape as `y`.
    """
    solver = Kalman() if solver is None else solver
    if not isinstance(solver, (Kalman, SteadyStateKalman, InformationKalman)):
        raise TypeError("solver must be Kalman, SteadyStateKalman or InformationKalman")
    information = isinstance(solver, InformationKalman)
    x_pred, P_pred, R = prepare_state(
        x_pred,
        P_pred,
        R,
        measurement=y,
        diagonal_noise=information,
    )
    u = _zero_input(sys, x_pred) if u is None else u
    available = _measurement_flag(has_measurement)
    if information:
        result = information_update(sys, x_pred, P_pred, y, R, u, available)
        return checked(result, throw, "Kalman update failed; inspect status")
    if isinstance(solver, SteadyStateKalman):
        result = cached_update(sys, x_pred, P_pred, y, R, u, available, solver)
        return checked(result, throw, "Kalman update failed; inspect status")

    def apply_update():
        y_pred = sys.C @ x_pred + sys.D @ u
        S = sys.C @ P_pred @ sys.C.T + R
        factor = jnp.linalg.cholesky(S)
        K = cho_solve((factor, True), (P_pred @ sys.C.T).T).T
        # vmap lowers cond to selection; keep inactive NaN measurements out of
        # arithmetic as well as keeping all work inside the scalar branch.
        observed = jnp.where(_measurement_flag(has_measurement), y, jnp.zeros_like(y))
        innov = observed - y_pred
        x_new = x_pred + K @ innov
        P_new = _covariance_update(P_pred, sys.C, K, R, solver.covariance_update)
        return x_new, P_new, innov

    def skip_update():
        return x_pred, P_pred, jnp.zeros_like(y)

    x, P, innovation = jax.lax.cond(
        _measurement_flag(has_measurement),
        apply_update,
        skip_update,
    )
    result = FilterUpdateResult(x, P, innovation, input_status(P_pred, R))
    return checked(result, throw, "Kalman update failed; inspect status")


def kalman_step(
    sys: DiscLTI,
    x: Array,
    P: Array,
    y: Array,
    *,
    Q: Array,
    R: Array,
    u: Array | None = None,
    has_measurement: Array | bool = True,
    solver: Kalman | SteadyStateKalman | InformationKalman | None = None,
    u_measurement: Array | None = None,
    throw: bool = False,
) -> FilterUpdateResult:
    """Run one predict-update step of a linear Kalman filter.

    The step can be called directly from a service loop or placed inside
    `jax.lax.scan` for a batch workflow.

    Args:
        sys: Discrete-time linear system.
        x: Filtered mean at the previous time step. Shape: `(n,)`.
        P: Filtered covariance at the previous time step. Shape: `(n, n)`.
        y: Measurement vector. Shape: `(p,)`.
        Q: Process noise covariance. Shape: `(n, n)`.
        R: Measurement covariance `(p, p)`, or variances `(p,)` in information mode.
        u: Optional input. Shape: `(m,)`.
        has_measurement: Whether to apply the measurement update.
        solver: Covariance, information or explicit cached-design configuration,
            shared with batch filtering.
        u_measurement: Input at the measurement time for direct feedthrough.
            Defaults to u; pass separately when consecutive controls differ.

        throw: Raise on numerical failure; otherwise inspect status.

    Returns:
        FilterUpdateResult: Mean x, covariance P, innovation, status and successful.
    """
    prediction = kalman_predict(sys, x, P, Q=Q, u=u)
    x_pred, P_pred = prediction
    update = kalman_update(
        sys,
        x_pred,
        P_pred,
        y,
        R=R,
        u=u if u_measurement is None else u_measurement,
        has_measurement=has_measurement,
        solver=solver,
    )
    code = jnp.where(
        prediction.solver_status != 0, prediction.solver_status, update.solver_status
    )
    result = FilterUpdateResult(update.x, update.P, update.innovation, code)
    return checked(result, throw, "Kalman step failed; inspect status")


def kalman(
    sys: DiscLTI,
    ys: Array,  # (T, p) measurement sequence
    *,
    Q: Array,
    R: Array,
    x0: Array | None = None,
    P0: Array | None = None,
    us: Array | None = None,
    has_measurement: Array | None = None,
    solver: Kalman | SteadyStateKalman | InformationKalman | None = None,
    throw: bool = False,
) -> KalmanResult:
    """Run a linear discrete-time Kalman filter.

    Filters a measurement sequence for a
    [DiscLTI][contrax.systems.DiscLTI] system using the standard
    predict-update recursion expressed as a pure `lax.scan`.

    Innovation gains use Cholesky solves and posterior covariances use Joseph
    form by default. Innovation covariances must be positive definite. Gradients
    flow through the filter recursion, including covariance tuning. Inputs are
    promoted to a common floating dtype, including cached design data.

    Args:
        sys: Discrete-time linear system.
        Q: Process noise covariance. Shape: `(n, n)`.
        R: Measurement covariance `(p, p)`, or a positive variance vector `(p,)`
            with InformationKalman, which requires positive-definite priors.
        ys: Measurement sequence. Shape: `(T, p)`.
        x0: Optional initial mean. Shape: `(n,)`. Defaults to zeros.
        P0: Initial predicted covariance `(n, n)`. Defaults to identity for
            covariance/information filtering, or design.P with SteadyStateKalman.
            A provided cached-mode P0 must match the design's predicted covariance.
        us: Controls for feedthrough at k and transition k to k+1. Shape:
            `(T, m)`. Defaults to zeros; the final transition is not returned.
        has_measurement: Whole-observation availability flags. Shape: `(T,)`.
            Missing observations may contain NaNs and return zero innovations.
        solver: Kalman (Joseph default, compact optional), InformationKalman for
            diagonal R and state-size solves, or SteadyStateKalman(design).
            Cached mode requires a fully observed stream and a gain/covariance
            consistent with A, C, Q and R; mismatch reports invalid_problem.

        throw: Raise on numerical failure; otherwise inspect status.

    Returns:
        [KalmanResult][contrax.types.KalmanResult]: A bundle containing
            filtered means `xs`, filtered covariances `Ps`, and measurement
            innovations `innovations`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> sys = cx.dss(
        ...     jnp.array([[1.0, 0.1], [0.0, 1.0]]),
        ...     jnp.array([[0.0], [0.1]]),
        ...     jnp.eye(2),
        ...     jnp.zeros((2, 1)),
        ...     dt=0.1,
        ... )
        >>> result = cx.kalman(
        ...     sys, jnp.zeros((20, 2)), Q=1e-3 * jnp.eye(2), R=1e-2 * jnp.eye(2)
        ... )
    """
    if not isinstance(sys, DiscLTI):
        raise TypeError("kalman requires a discrete LTI system")
    solver = Kalman() if solver is None else solver
    if not isinstance(solver, (Kalman, SteadyStateKalman, InformationKalman)):
        raise TypeError("solver must be Kalman, SteadyStateKalman or InformationKalman")
    information = isinstance(solver, InformationKalman)
    extra = ()
    if isinstance(solver, SteadyStateKalman):
        extra = (solver.design.K, solver.design.P)
        P0 = solver.design.P if P0 is None else P0
    ys, Q, R, us, x0, P0, flags = prepare_filter(
        sys,
        ys,
        Q,
        R,
        us,
        x0,
        P0,
        has_measurement,
        diagonal_R=information,
        extra_arrays=extra,
    )
    if isinstance(solver, SteadyStateKalman):
        result = cached_filter(sys, ys, Q, R, us, x0, P0, flags, solver)
        return checked(result, throw, "Kalman filter failed; inspect status")

    # Update-then-predict: x0/P0 is the prior on x_0.
    # Each scan step updates with y_k first (using the current prior), then
    # predicts forward to give the prior on x_{k+1}.
    def update_predict(carry, inputs):
        x_prior, P_prior = carry
        y, u, available, k = inputs
        x_post, P_post, innov = kalman_update(
            sys, x_prior, P_prior, y, R=R, u=u, has_measurement=available, solver=solver
        )
        x_next, P_next = jax.lax.cond(
            k + 1 < ys.shape[0],
            lambda: tuple(kalman_predict(sys, x_post, P_post, Q=Q, u=u)),
            lambda: (x_post, P_post),
        )
        return (x_next, P_next), (x_post, P_post, innov)

    _, (x_hats, Ps, innovations) = jax.lax.scan(
        update_predict, (x0, P0), (ys, us, flags, jnp.arange(ys.shape[0]))
    )
    result = KalmanResult(
        xs=x_hats,
        Ps=Ps,
        innovations=innovations,
        has_measurement=flags,
        solver_status=jnp.where(
            input_status(P0, Q) != 0, 2, noise_status(R, information)
        ),
    )
    return checked(result, throw, "Kalman filter failed; inspect status")
