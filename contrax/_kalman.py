"""contrax._kalman — Linear Kalman filter, gain design, and RTS smoother."""

import jax
import jax.numpy as jnp
from jax import Array

from contrax._riccati import dare
from contrax.core import DiscLTI
from contrax.types import KalmanGainResult, KalmanResult, RTSResult


def _symmetrize(X: Array) -> Array:
    return (X + X.T) / 2


def _zero_input(sys: DiscLTI, x: Array) -> Array:
    return jnp.zeros(sys.B.shape[1], dtype=x.dtype)


def _measurement_flag(has_measurement: Array | bool) -> Array:
    return jnp.asarray(has_measurement, dtype=bool)


def kalman_predict(
    sys: DiscLTI,
    x: Array,
    P: Array,
    Q_noise: Array,
    u: Array | None = None,
) -> tuple[Array, Array]:
    """Predict one linear Kalman filter step.

    Args:
        sys: Discrete-time linear system.
        x: Filtered mean at the previous time step. Shape: `(n,)`.
        P: Filtered covariance at the previous time step. Shape: `(n, n)`.
        Q_noise: Process noise covariance. Shape: `(n, n)`.
        u: Optional input for the transition. Shape: `(m,)`. If omitted, the
            input term is zero.

    Returns:
        Tuple `(x_pred, P_pred)` with the predicted mean and covariance.
    """
    u = _zero_input(sys, x) if u is None else u
    x_pred = sys.A @ x + sys.B @ u
    P_pred = _symmetrize(sys.A @ P @ sys.A.T + Q_noise)
    return x_pred, P_pred


def kalman_update(
    sys: DiscLTI,
    x_pred: Array,
    P_pred: Array,
    y: Array,
    R_noise: Array,
    u: Array | None = None,
    *,
    has_measurement: Array | bool = True,
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
        R_noise: Measurement noise covariance. Shape: `(p, p)`.
        u: Optional input for direct-feedthrough measurement models. Shape:
            `(m,)`. If omitted, the feedthrough term is zero.
        has_measurement: Whether to apply the measurement update.

    Returns:
        Tuple `(x, P, innovation)` with the filtered mean, filtered covariance,
        and measurement residual. Missing-measurement steps return a zero
        innovation with the same shape as `y`.
    """
    u = _zero_input(sys, x_pred) if u is None else u
    n = x_pred.shape[0]
    y_pred = sys.C @ x_pred + sys.D @ u
    S = sys.C @ P_pred @ sys.C.T + R_noise
    K = jnp.linalg.solve(S.T, (P_pred @ sys.C.T).T).T
    innov = y - y_pred
    x_new = x_pred + K @ innov
    P_new = _symmetrize((jnp.eye(n, dtype=P_pred.dtype) - K @ sys.C) @ P_pred)

    def apply_update():
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
    Q_noise: Array,
    R_noise: Array,
    u: Array | None = None,
    *,
    has_measurement: Array | bool = True,
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
        Q_noise: Process noise covariance. Shape: `(n, n)`.
        R_noise: Measurement noise covariance. Shape: `(p, p)`.
        u: Optional input. Shape: `(m,)`.
        has_measurement: Whether to apply the measurement update.

    Returns:
        Tuple `(x, P, innovation)` after the step.
    """
    x_pred, P_pred = kalman_predict(sys, x, P, Q_noise, u)
    return kalman_update(
        sys,
        x_pred,
        P_pred,
        y,
        R_noise,
        u,
        has_measurement=has_measurement,
    )


def kalman_gain(
    sys: DiscLTI,
    Q_noise: Array,
    R_noise: Array,
) -> KalmanGainResult:
    """Design a steady-state Kalman filter gain for a discrete LTI system.

    This is the estimator dual of discrete LQR. It solves the DARE for
    `(A.T, C.T, Q_noise, R_noise)` and returns the measurement-update gain used
    by `kalman_update()`.

    Args:
        sys: Discrete-time linear system.
        Q_noise: Process noise covariance. Shape: `(n, n)`.
        R_noise: Measurement noise covariance. Shape: `(p, p)`.

    Returns:
        [KalmanGainResult][contrax.types.KalmanGainResult]: A bundle with
            update gain `K`, steady-state predicted covariance `P`, and
            estimator error-dynamics poles.

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
        >>> design = cx.kalman_gain(sys, 1e-3 * jnp.eye(2), 1e-2 * jnp.eye(2))
    """
    P_pred = dare(sys.A.T, sys.C.T, Q_noise, R_noise).S
    innovation_cov = sys.C @ P_pred @ sys.C.T + R_noise
    K = jnp.linalg.solve(innovation_cov.T, (P_pred @ sys.C.T).T).T
    poles = jnp.linalg.eigvals(sys.A - sys.A @ K @ sys.C)
    return KalmanGainResult(K=K, P=P_pred, poles=poles)


def kalman(
    sys: DiscLTI,
    Q_noise: Array,  # (n, n) process noise covariance
    R_noise: Array,  # (p, p) measurement noise covariance
    ys: Array,  # (T, p) measurement sequence
    x0: Array | None = None,
    P0: Array | None = None,
) -> KalmanResult:
    """Run a linear discrete-time Kalman filter.

    Filters a measurement sequence for a
    [DiscLTI][contrax.systems.DiscLTI] system using the standard
    predict-update recursion expressed as a pure `lax.scan`.

    This is the baseline linear-Gaussian estimation path in Contrax. Innovation
    gains are computed with linear solves rather than explicit inverses, and
    the scan structure makes small differentiable covariance-tuning loops
    practical.

    Args:
        sys: Discrete-time linear system.
        Q_noise: Process noise covariance. Shape: `(n, n)`.
        R_noise: Measurement noise covariance. Shape: `(p, p)`.
        ys: Measurement sequence. Shape: `(T, p)`.
        x0: Optional initial mean. Shape: `(n,)`. Defaults to zeros.
        P0: Optional initial covariance. Shape: `(n, n)`. Defaults to the
            identity matrix.

    Returns:
        [KalmanResult][contrax.types.KalmanResult]: A bundle containing
            filtered means `x_hat`, filtered covariances `P`, and measurement
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
        ...     sys,
        ...     Q_noise=1e-3 * jnp.eye(2),
        ...     R_noise=1e-2 * jnp.eye(2),
        ...     ys=jnp.zeros((20, 2)),
        ... )
    """
    n = sys.A.shape[0]
    if x0 is None:
        x0 = jnp.zeros(n, dtype=ys.dtype)
    if P0 is None:
        P0 = jnp.eye(n, dtype=ys.dtype)

    # Update-then-predict: x0/P0 is the prior on x_0.
    # Each scan step updates with y_k first (using the current prior), then
    # predicts forward to give the prior on x_{k+1}.
    def update_predict(carry, y):
        x_prior, P_prior = carry
        x_post, P_post, innov = kalman_update(sys, x_prior, P_prior, y, R_noise)
        x_next, P_next = kalman_predict(sys, x_post, P_post, Q_noise)
        return (x_next, P_next), (x_post, P_post, innov)

    _, (x_hats, Ps, innovations) = jax.lax.scan(update_predict, (x0, P0), ys)
    return KalmanResult(x_hat=x_hats, P=Ps, innovations=innovations)


def rts(
    sys: DiscLTI,
    result: KalmanResult,
    Q_noise: Array,
) -> RTSResult:
    """Run a Rauch-Tung-Striebel smoother on filtered Kalman results.

    Applies the standard backward smoothing pass to the output of `kalman()`
    using the same process model and process noise covariance.

    This is an offline smoothing primitive rather than an online filter.
    `Q_noise` should match the covariance used in the forward `kalman()` pass;
    for deterministic dynamics, pass zeros rather than omitting the term.

    Args:
        sys: Discrete-time linear system used in the forward filter.
        result: Output of `kalman()`.
        Q_noise: Process noise covariance used by the forward filter.

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
        ...     sys,
        ...     Q_noise=1e-3 * jnp.eye(2),
        ...     R_noise=1e-2 * jnp.eye(2),
        ...     ys=jnp.zeros((20, 2)),
        ... )
        >>> smoothed = cx.rts(sys, filtered, Q_noise=1e-3 * jnp.eye(2))
    """
    x_hats = result.x_hat  # (T, n)
    Ps = result.P  # (T, n, n)

    def smooth_step(carry, inputs):
        x_s, P_s = carry
        x_f, P_f = inputs

        # Predicted mean/cov from forward pass
        x_pred = sys.A @ x_f
        P_pred = sys.A @ P_f @ sys.A.T + Q_noise

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
        smooth_step, init, (x_hats[:-1][::-1], Ps[:-1][::-1])
    )
    # Prepend the last (unsmoothed) estimate and reverse
    x_smooth = jnp.concatenate([x_smooth_rev[::-1], x_hats[-1:]], axis=0)
    P_smooth = jnp.concatenate([P_smooth_rev[::-1], Ps[-1:]], axis=0)
    return RTSResult(x_smooth=x_smooth, P_smooth=P_smooth)
