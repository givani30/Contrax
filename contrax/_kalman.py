"""contrax._kalman — Linear Kalman filter, gain design, and RTS smoother."""

import jax
import jax.numpy as jnp
from jax import Array

from contrax._filter_common import prepare_filter
from contrax._riccati import dare
from contrax.core import DiscLTI
from contrax.solvers import DARE, Kalman
from contrax.types import KalmanGainResult, KalmanResult, RTSResult


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
) -> tuple[Array, Array]:
    """Predict one linear Kalman filter step.

    Args:
        sys: Discrete-time linear system.
        x: Filtered mean at the previous time step. Shape: `(n,)`.
        P: Filtered covariance at the previous time step. Shape: `(n, n)`.
        Q: Process noise covariance. Shape: `(n, n)`.
        u: Optional input for the transition. Shape: `(m,)`. If omitted, the
            input term is zero.

    Returns:
        Tuple `(x_pred, P_pred)` with the predicted mean and covariance.
    """
    u = _zero_input(sys, x) if u is None else u
    x_pred = sys.A @ x + sys.B @ u
    P_pred = _symmetrize(sys.A @ P @ sys.A.T + Q)
    return x_pred, P_pred


def kalman_update(
    sys: DiscLTI,
    x_pred: Array,
    P_pred: Array,
    y: Array,
    *,
    R: Array,
    u: Array | None = None,
    has_measurement: Array | bool = True,
    solver: Kalman | None = None,
) -> tuple[Array, Array, Array]:
    """Update one linear Kalman filter step.

    `has_measurement=False` returns the predicted state and covariance without
    applying a measurement update. The flag may be a JAX scalar boolean, making
    this helper usable inside `jax.lax.scan` for streams with missing samples.

    Args:
        sys: Discrete-time linear system.
        x_pred: Predicted mean. Shape: `(n,)`.
        P_pred: Predicted covariance. Shape: `(n, n)`.
        y: Measurement vector. Shape: `(p,)`.
        R: Measurement noise covariance. Shape: `(p, p)`.
        u: Optional input for direct-feedthrough measurement models. Shape:
            `(m,)`. If omitted, the feedthrough term is zero.
        has_measurement: Whether to apply the measurement update.
        solver: Joseph (default) or compact covariance-update configuration.

    Returns:
        Tuple `(x, P, innovation)` with the filtered mean, filtered covariance,
        and measurement residual. Missing-measurement steps return a zero
        innovation with the same shape as `y`.
    """
    u = _zero_input(sys, x_pred) if u is None else u
    solver = Kalman() if solver is None else solver
    if not isinstance(solver, Kalman):
        raise TypeError("kalman_update solver must be contrax.solvers.Kalman")

    def apply_update():
        y_pred = sys.C @ x_pred + sys.D @ u
        S = sys.C @ P_pred @ sys.C.T + R
        K = jnp.linalg.solve(S.T, (P_pred @ sys.C.T).T).T
        # vmap lowers cond to selection; keep inactive NaN measurements out of
        # arithmetic as well as keeping all work inside the scalar branch.
        observed = jnp.where(_measurement_flag(has_measurement), y, jnp.zeros_like(y))
        innov = observed - y_pred
        x_new = x_pred + K @ innov
        P_new = _covariance_update(P_pred, sys.C, K, R, solver.covariance_update)
        return x_new, P_new, innov

    def skip_update():
        return x_pred, P_pred, jnp.zeros_like(y)

    return jax.lax.cond(
        _measurement_flag(has_measurement),
        apply_update,
        skip_update,
    )


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
    solver: Kalman | None = None,
    u_measurement: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Run one predict-update step of a linear Kalman filter.

    This is the online counterpart to `kalman()`. It is a pure array function,
    so service loops can call it directly and batch workflows can place it
    inside `jax.lax.scan`.

    Args:
        sys: Discrete-time linear system.
        x: Filtered mean at the previous time step. Shape: `(n,)`.
        P: Filtered covariance at the previous time step. Shape: `(n, n)`.
        y: Measurement vector. Shape: `(p,)`.
        Q: Process noise covariance. Shape: `(n, n)`.
        R: Measurement noise covariance. Shape: `(p, p)`.
        u: Optional input. Shape: `(m,)`.
        has_measurement: Whether to apply the measurement update.
        solver: Covariance-update configuration shared with batch filtering.
        u_measurement: Input at the measurement time for direct feedthrough.
            Defaults to u; pass separately when consecutive controls differ.

    Returns:
        Tuple `(x, P, innovation)` after the step.
    """
    x_pred, P_pred = kalman_predict(sys, x, P, Q=Q, u=u)
    return kalman_update(
        sys,
        x_pred,
        P_pred,
        y,
        R=R,
        u=u if u_measurement is None else u_measurement,
        has_measurement=has_measurement,
        solver=solver,
    )


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
    )


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
    solver: Kalman | None = None,
) -> KalmanResult:
    """Run a linear discrete-time Kalman filter.

    Filters a measurement sequence for a
    [DiscLTI][contrax.systems.DiscLTI] system using the standard
    predict-update recursion expressed as a pure `lax.scan`.

    Innovation gains use linear solves and posterior covariances use Joseph
    form by default. Supports `jit`, `vmap` and differentiable covariance tuning.
    All input arrays should share a floating dtype; filtering preserves it.

    Args:
        sys: Discrete-time linear system.
        Q: Process noise covariance. Shape: `(n, n)`.
        R: Measurement noise covariance. Shape: `(p, p)`.
        ys: Measurement sequence. Shape: `(T, p)`.
        x0: Optional initial mean. Shape: `(n,)`. Defaults to zeros.
        P0: Optional initial covariance. Shape: `(n, n)`. Defaults to the
            identity matrix.
        us: Controls for feedthrough at k and transition k to k+1. Shape:
            `(T, m)`. Defaults to zeros; the final transition is not returned.
        has_measurement: Whole-observation availability flags. Shape: `(T,)`.
            Missing observations may contain NaNs and return zero innovations.
        solver: Joseph (default) or compact covariance-update configuration.

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
    ys, Q, R, us, x0, P0, flags = prepare_filter(
        sys, ys, Q, R, us, x0, P0, has_measurement
    )
    if Q.shape != sys.A.shape or R.shape != (sys.C.shape[0], sys.C.shape[0]):
        raise ValueError("Noise dimensions must match the model state and output")

    # Update-then-predict: x0/P0 is the prior on x_0.
    # Each scan step updates with y_k first (using the current prior), then
    # predicts forward to give the prior on x_{k+1}.
    def update_predict(carry, inputs):
        x_prior, P_prior = carry
        y, u, available = inputs
        x_post, P_post, innov = kalman_update(
            sys, x_prior, P_prior, y, R=R, u=u, has_measurement=available, solver=solver
        )
        x_next, P_next = kalman_predict(sys, x_post, P_post, Q=Q, u=u)
        return (x_next, P_next), (x_post, P_post, innov)

    _, (x_hats, Ps, innovations) = jax.lax.scan(
        update_predict, (x0, P0), (ys, us, flags)
    )
    return KalmanResult(
        xs=x_hats, Ps=Ps, innovations=innovations, has_measurement=flags
    )


def rts(
    sys: DiscLTI,
    result: KalmanResult,
    *,
    Q: Array,
    us: Array | None = None,
) -> RTSResult:
    """Run a Rauch-Tung-Striebel smoother on filtered Kalman results.

    Applies the standard backward smoothing pass to the output of `kalman()`
    using the same process model and process noise covariance.

    This is an offline smoothing primitive rather than an online filter.
    `Q` should match the covariance used in the forward `kalman()` pass;
    for deterministic dynamics, pass zeros rather than omitting the term.

    Args:
        sys: Discrete-time linear system used in the forward filter.
        result: Output of `kalman()`.
        Q: Process noise covariance used by the forward filter.
        us: Controls from the forward filter. Shape: `(T, m)`. The last input
            is unused; defaults to zeros.

    Returns:
        [RTSResult][contrax.types.RTSResult]: A bundle containing smoothed
            state means and covariances.

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
        >>> filtered = cx.kalman(
        ...     sys, jnp.zeros((20, 2)), Q=1e-3 * jnp.eye(2), R=1e-2 * jnp.eye(2)
        ... )
        >>> smoothed = cx.rts(sys, filtered, Q=0.001 * jnp.eye(2))
    """
    x_hats = result.xs  # (T, n)
    Ps = result.Ps  # (T, n, n)
    if x_hats.shape[0] == 0:
        return RTSResult(xs=x_hats, Ps=Ps)
    us = (
        jnp.zeros((x_hats.shape[0], sys.B.shape[1]), dtype=x_hats.dtype)
        if us is None
        else us
    )
    if us.shape != (x_hats.shape[0], sys.B.shape[1]):
        raise ValueError("us must have shape (T, m)")

    def smooth_step(carry, inputs):
        x_s, P_s = carry
        x_f, P_f, u = inputs

        # Predicted mean/cov from forward pass
        x_pred = sys.A @ x_f + sys.B @ u
        P_pred = sys.A @ P_f @ sys.A.T + Q

        G = jnp.linalg.solve(
            P_pred.T, sys.A @ P_f.T
        ).T  # smoother gain: P_f A^T P_pred^{-1}
        x_s_new = x_f + G @ (x_s - x_pred)
        P_s_new = P_f + G @ (P_s - P_pred) @ G.T
        return (x_s_new, P_s_new), (x_s_new, P_s_new)

    # Initialise smoother at last filtered estimate
    init = (x_hats[-1], Ps[-1])
    # Scan backwards over t = T-2, ..., 0
    _, (x_smooth_rev, P_smooth_rev) = jax.lax.scan(
        smooth_step, init, (x_hats[:-1][::-1], Ps[:-1][::-1], us[:-1][::-1])
    )
    # Prepend the last (unsmoothed) estimate and reverse
    x_smooth = jnp.concatenate([x_smooth_rev[::-1], x_hats[-1:]], axis=0)
    P_smooth = jnp.concatenate([P_smooth_rev[::-1], Ps[-1:]], axis=0)
    return RTSResult(xs=x_smooth, Ps=P_smooth)
