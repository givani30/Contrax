"""Linear Rauch–Tung–Striebel smoothing with checked numerical status."""

import jax
import jax.numpy as jnp
from jax import Array

from contrax._filter_checks import checked, input_status, state_status
from contrax.core import DiscLTI
from contrax.types import KalmanResult, RTSResult


def rts(
    sys: DiscLTI,
    result: KalmanResult,
    *,
    Q: Array,
    us: Array | None = None,
    throw: bool = False,
) -> RTSResult:
    """Run a Rauch-Tung-Striebel smoother on filtered Kalman results.

    Applies the standard backward smoothing pass to the output of `kalman()`
    using the same process model and process noise covariance.

    Smooths a completed `kalman()` pass using the same process model and noise
    covariance. For deterministic dynamics, set `Q` to zero.

    Args:
        sys: Discrete-time linear system used in the forward filter.
        result: Output of `kalman()`.
        Q: Process noise covariance used by the forward filter.
        us: Controls from the forward filter. Shape: `(T, m)`. The last input
            is unused; defaults to zeros.

        throw: Raise on numerical failure; otherwise inspect status.

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
    if not isinstance(sys, DiscLTI):
        raise TypeError("rts requires a discrete LTI system")
    Q = jnp.asarray(Q)
    if Q.shape != sys.A.shape:
        raise ValueError("Q must match the model state dimensions")
    # Validate the incoming trajectory once, using a PD factorization fast
    # path with spectral fallback for singular PSD matrices.
    forward_status = state_status(
        result.xs, result.Ps, result.solver_status, trajectory=True, cholesky=True
    )
    forward_status = jnp.where(
        forward_status != 0,
        forward_status,
        jnp.where(jnp.all(jnp.isfinite(result.innovations)), 0, 3),
    )
    code = jnp.where(forward_status != 0, forward_status, input_status(Q))
    x_hats = result.xs  # (T, n)
    Ps = result.Ps  # (T, n, n)
    if x_hats.shape[0] == 0:
        return checked(
            RTSResult(xs=x_hats, Ps=Ps, solver_status=code),
            throw,
            "RTS smoother failed; inspect status",
        )
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
        P_s_new = (P_s_new + P_s_new.T) / 2
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
    smoothed = RTSResult(xs=x_smooth, Ps=P_smooth, solver_status=code)
    return checked(smoothed, throw, "RTS smoother failed; inspect status")
