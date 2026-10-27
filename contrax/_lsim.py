"""Exact sampled open-loop continuous LTI responses."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from contrax.core import ContLTI, DiscLTI
from contrax.types import SimResult


def continuous_lsim(sys, us, x0, ts, interpolation, throw):
    if ts is None:
        raise ValueError("Continuous lsim requires an explicit ts sample-time vector")
    if interpolation not in ("zoh", "foh"):
        raise ValueError("input_interpolation must be zoh or foh")
    ts = jnp.asarray(ts)
    if ts.ndim != 1 or us.ndim != 2 or us.shape != (ts.shape[0], sys.B.shape[1]):
        raise ValueError("ts and us must have shapes (N,) and (N, m)")
    dtype = jnp.result_type(sys.A, sys.B, us, x0)
    x0, us = x0.astype(dtype), us.astype(dtype)
    n, m = sys.B.shape
    if x0.shape != (n,):
        raise ValueError("x0 must have shape (n,)")
    if ts.shape[0] == 0:
        return SimResult(
            ts,
            jnp.empty((0, n), dtype=dtype),
            jnp.empty((0, sys.C.shape[0]), dtype=dtype),
            state_times=ts,
        )

    def step(x, inputs):
        h, u0, u1 = inputs
        h = h.astype(dtype)
        if interpolation == "zoh":
            block = jnp.block(
                [[sys.A * h, sys.B * h], [jnp.zeros((m, n + m), dtype=dtype)]]
            )
            exponential = jax.scipy.linalg.expm(block)
            next_x = exponential[:n, :n] @ x + exponential[:n, n:] @ u0
        else:
            # Reference: Van Loan (1978), block-exponential integral identity.
            # Normalize each interval to [0,1]: u'=u1-u0, avoiding inverse A
            # and division by h, including for marginal/singular dynamics.
            block = jnp.block(
                [
                    [sys.A * h, sys.B * h, jnp.zeros((n, m), dtype=dtype)],
                    [jnp.zeros((m, n + m), dtype=dtype), jnp.eye(m, dtype=dtype)],
                    [jnp.zeros((m, n + 2 * m), dtype=dtype)],
                ]
            )
            exponential = jax.scipy.linalg.expm(block)
            next_x = (
                exponential[:n, :n] @ x
                + exponential[:n, n : n + m] @ u0
                + exponential[:n, n + m :] @ (u1 - u0)
            )
        return next_x, next_x

    _, states = jax.lax.scan(step, x0, (ts[1:] - ts[:-1], us[:-1], us[1:]))
    states = jnp.concatenate([x0[None], states], axis=0)
    outputs = states @ sys.C.T + us @ sys.D.T
    valid = jnp.all(jnp.isfinite(ts)) & jnp.all(ts[1:] > ts[:-1])
    result = SimResult(
        ts,
        states,
        outputs,
        state_times=ts,
        solver_status=jnp.where(valid, 0, 2),
        num_steps=jnp.asarray(ts.shape[0] - 1),
    )
    if throw:
        result = eqx.error_if(
            result, ~result.successful, "Continuous lsim failed; inspect status"
        )
    return result


def lsim(
    sys: DiscLTI | ContLTI,
    us: Array,  # (T, m) input sequence
    x0: Array | None = None,  # (n,) initial state; defaults to zeros
    *,
    ts: Array | None = None,
    input_interpolation: str = "zoh",
    throw: bool = False,
) -> SimResult:
    """Simulate a discrete or continuous LTI system from sampled open-loop inputs.

    Discrete indexing returns T inputs/outputs and T+1 states. Continuous
    systems require N sample times and inputs and return N sampled states and
    outputs, with x0 at ts[0]. Exact block exponentials implement ZOH or FOH
    inputs on each interval; nonuniform grids are supported.

    The input sequence has a fixed shape and each interval uses the selected
    zero-order or first-order hold convention.

    Args:
        sys: Continuous or discrete LTI system.
        us: Input sequence. Shape: `(T, m)` discrete or `(N, m)` continuous.
        x0: Initial state. Shape: `(n,)`. Defaults to zeros.
        ts: Continuous-only sample times. Shape: `(N,)`, strictly increasing.
        input_interpolation: Continuous interval interpretation, zoh or foh.
        throw: Raise on numerical failure, otherwise inspect result.status.

    Returns:
        SimResult: Simulation result with `.ts`, `.xs`, `.ys`. Shapes:
            `(T,)`, `(T + 1, n)`, `(T, p)` for discrete systems and
            `(N,)`, `(N, n)`, `(N, p)` for continuous systems. `status` and
            `successful` report numerical validity.
            state_times aligns with xs, output_times (also ts) aligns with ys.

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
        >>> result = cx.lsim(sys, jnp.zeros((20, 1)))
    """
    if not isinstance(sys, (DiscLTI, ContLTI)):
        raise TypeError("lsim requires a continuous or discrete LTI system")
    us = jnp.asarray(us)
    n = sys.A.shape[0]
    if x0 is None:
        x0 = jnp.zeros(n, dtype=sys.A.dtype)
    x0 = jnp.asarray(x0)
    if isinstance(sys, ContLTI):
        return continuous_lsim(sys, us, x0, ts, input_interpolation, throw)
    if ts is not None or input_interpolation != "zoh":
        raise ValueError(
            "Discrete lsim uses sys.dt and sample inputs; ts/FOH are continuous-only"
        )

    if us.ndim != 2 or us.shape[1] != sys.B.shape[1] or x0.shape != (n,):
        raise ValueError("us and x0 must have shapes (T, m) and (n,)")
    T = us.shape[0]
    dtype = jnp.result_type(sys.A, us, x0)
    x0, us = x0.astype(dtype), us.astype(dtype)
    ts = jnp.arange(T, dtype=sys.A.real.dtype) * sys.dt

    def step(x, u):
        x_next = sys.A @ x + sys.B @ u
        y = sys.C @ x + sys.D @ u
        return x_next, (x_next, y)

    _, (xs, ys) = jax.lax.scan(step, x0, us)
    xs_full = jnp.concatenate([x0[None], xs], axis=0)
    result = SimResult(
        ts,
        xs_full,
        ys,
        state_times=jnp.arange(T + 1, dtype=ts.dtype) * sys.dt,
        num_steps=jnp.asarray(T),
    )
    if throw:
        result = eqx.error_if(result, ~result.successful, "Discrete lsim failed")
    return result
