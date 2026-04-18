"""contrax.control — LQR, pole placement, closed-loop construction."""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from contrax._place import place  # noqa: F401 — re-exported
from contrax._riccati import care, dare  # noqa: F401 — re-exported
from contrax.core import ContLTI, DiscLTI
from contrax.types import LQRResult

__all__ = [
    "lqr",
    "lqi",
    "dare",
    "care",
    "place",
    "state_feedback",
    "feedback",
    "augment_integrator",
]


def lqr(sys: DiscLTI | ContLTI, Q: Array, R: Array) -> LQRResult:
    """Solve the infinite-horizon linear quadratic regulator problem.

    Dispatches to `dare()` for discrete systems and `care()` for continuous
    systems, returning a unified [LQRResult][contrax.types.LQRResult].

    The discrete path is more benchmarked than the continuous path, but both
    routes are first-class public solvers. Type dispatch happens outside traced
    runtime values, so `jit` works as long as the system type is fixed at trace
    time.

    Args:
        sys: [DiscLTI][contrax.systems.DiscLTI] or
            [ContLTI][contrax.systems.ContLTI] system.
        Q: State cost matrix. Shape: `(n, n)`.
        R: Input cost matrix. Shape: `(m, m)`.

    Returns:
        [LQRResult][contrax.types.LQRResult]: A bundle containing the optimal
            gain, Riccati solution, and closed-loop poles.

    Raises:
        TypeError: If `sys` is not a
            [DiscLTI][contrax.systems.DiscLTI] or
            [ContLTI][contrax.systems.ContLTI].

    Examples:
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> sys = cx.dss(
        ...     jnp.array([[1.0, 0.05], [0.0, 1.0]]),
        ...     jnp.array([[0.0], [0.05]]),
        ...     jnp.eye(2),
        ...     jnp.zeros((2, 1)),
        ...     dt=0.05,
        ... )
        >>> result = cx.lqr(sys, jnp.eye(2), jnp.array([[1.0]]))
    """
    if isinstance(sys, DiscLTI):
        return dare(sys.A, sys.B, Q, R)
    elif isinstance(sys, ContLTI):
        return care(sys.A, sys.B, Q, R)
    else:
        raise TypeError(f"Expected DiscLTI or ContLTI, got {type(sys)}")


def augment_integrator(
    sys: DiscLTI | ContLTI,
    C_integral: Array | None = None,
    D_integral: Array | None = None,
    *,
    sign: float = 1.0,
    dt_scale: ArrayLike | None = None,
) -> DiscLTI | ContLTI:
    """Augment an LTI system with integral-of-output states.

    This is the small design utility behind LQI-style workflows. It returns an
    augmented system that can be passed to `lqr()` with a state cost that also
    weights the added integral states.

    For continuous systems, the added state evolves as
    `z_dot = sign * (C_integral x + D_integral u)`. For discrete systems, the
    added state evolves as
    `z[k+1] = z[k] + dt_scale * sign * (C_integral x[k] + D_integral u[k])`.
    When `dt_scale` is omitted for a discrete system, `sys.dt` is used.

    Args:
        sys: Continuous or discrete LTI system to augment.
        C_integral: Output map to integrate. Shape: `(r, n)`. Defaults to
            `sys.C`.
        D_integral: Feedthrough map for the integrated output. Shape: `(r, m)`.
            Defaults to `sys.D`.
        sign: Sign applied to the integrated output. Use `-1.0` for the common
            `reference - output` convention when the reference is handled
            outside the design model.
        dt_scale: Discrete integration scale. Defaults to `sys.dt` for
            discrete systems and is ignored for continuous systems.

    Returns:
        [DiscLTI][contrax.systems.DiscLTI] |
            [ContLTI][contrax.systems.ContLTI]: System with augmented state
            `[x; z]`.
    """
    C_int = sys.C if C_integral is None else jnp.asarray(C_integral, dtype=sys.A.dtype)
    D_int = sys.D if D_integral is None else jnp.asarray(D_integral, dtype=sys.A.dtype)
    scale = jnp.asarray(sign, dtype=sys.A.dtype)

    n = sys.A.shape[0]
    r = C_int.shape[0]
    zeros_xz = jnp.zeros((n, r), dtype=sys.A.dtype)
    eye_z = jnp.eye(r, dtype=sys.A.dtype)
    C_aug = jnp.hstack([sys.C, jnp.zeros((sys.C.shape[0], r), dtype=sys.A.dtype)])

    if isinstance(sys, ContLTI):
        A_aug = jnp.block(
            [
                [sys.A, zeros_xz],
                [scale * C_int, jnp.zeros((r, r), dtype=sys.A.dtype)],
            ]
        )
        B_aug = jnp.vstack([sys.B, scale * D_int])
        return ContLTI(A=A_aug, B=B_aug, C=C_aug, D=sys.D)

    if isinstance(sys, DiscLTI):
        dt = sys.dt if dt_scale is None else jnp.asarray(dt_scale, dtype=sys.A.dtype)
        row_scale = scale * dt
        A_aug = jnp.block([[sys.A, zeros_xz], [row_scale * C_int, eye_z]])
        B_aug = jnp.vstack([sys.B, row_scale * D_int])
        return DiscLTI(A=A_aug, B=B_aug, C=C_aug, D=sys.D, dt=sys.dt)

    raise TypeError(f"Expected DiscLTI or ContLTI, got {type(sys)}")


def lqi(
    sys: DiscLTI | ContLTI,
    Q: Array,
    R: Array,
    C_integral: Array | None = None,
    D_integral: Array | None = None,
    *,
    sign: float = 1.0,
    dt_scale: ArrayLike | None = None,
) -> LQRResult:
    """Design an LQI gain by augmenting integral states and calling `lqr()`.

    This is intentionally a thin composition of `augment_integrator()` and
    `lqr()`. The returned gain acts on the augmented state `[x; z]`, where `z`
    is the integral state.

    Args:
        sys: Continuous or discrete LTI system.
        Q: Augmented state cost matrix. Shape: `(n + r, n + r)`.
        R: Input cost matrix. Shape: `(m, m)`.
        C_integral: Output map to integrate. Shape: `(r, n)`. Defaults to
            `sys.C`.
        D_integral: Feedthrough map for the integrated output. Shape: `(r, m)`.
            Defaults to `sys.D`.
        sign: Sign applied to the integrated output.
        dt_scale: Discrete integration scale. Defaults to `sys.dt` for
            discrete systems and is ignored for continuous systems.

    Returns:
        [LQRResult][contrax.types.LQRResult]: LQR result for the
            integrator-augmented design model.
    """
    augmented = augment_integrator(
        sys,
        C_integral=C_integral,
        D_integral=D_integral,
        sign=sign,
        dt_scale=dt_scale,
    )
    return lqr(augmented, Q, R)


def state_feedback(sys: DiscLTI | ContLTI, K: Array) -> DiscLTI | ContLTI:
    """Construct the closed-loop system under state feedback.

    Returns a new LTI system with `A_cl = A - B @ K` and unchanged `B`, `C`,
    `D`, and, for discrete systems, `dt`.

    This is a pure functional constructor rather than a simulation routine. It
    dispatches on either [DiscLTI][contrax.systems.DiscLTI] or
    [ContLTI][contrax.systems.ContLTI] and behaves like normal array algebra
    under JAX transforms.

    Args:
        sys: Continuous or discrete-time system.
        K: State-feedback gain. Shape: `(m, n)`.

    Returns:
        [DiscLTI][contrax.systems.DiscLTI] |
            [ContLTI][contrax.systems.ContLTI]: A new system representing the
            closed-loop dynamics.

    Examples:
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> sys = cx.dss(
        ...     jnp.array([[1.0, 0.05], [0.0, 1.0]]),
        ...     jnp.array([[0.0], [0.05]]),
        ...     jnp.eye(2),
        ...     jnp.zeros((2, 1)),
        ...     dt=0.05,
        ... )
        >>> result = cx.lqr(sys, jnp.eye(2), jnp.array([[1.0]]))
        >>> closed_loop = cx.state_feedback(sys, result.K)
    """
    if isinstance(sys, DiscLTI):
        return DiscLTI(
            A=sys.A - sys.B @ K,
            B=sys.B,
            C=sys.C,
            D=sys.D,
            dt=sys.dt,
        )
    elif isinstance(sys, ContLTI):
        return ContLTI(
            A=sys.A - sys.B @ K,
            B=sys.B,
            C=sys.C,
            D=sys.D,
        )
    else:
        raise TypeError(f"Expected DiscLTI or ContLTI, got {type(sys)}")


def feedback(sys: DiscLTI | ContLTI, K: Array) -> DiscLTI | ContLTI:
    """Alias for `state_feedback()`.

    This alias remains available for MATLAB-style ergonomics, but the more
    explicit `state_feedback()` name is preferred in new Contrax code because it
    distinguishes gain application from general closed-loop interconnection.
    """
    return state_feedback(sys, K)
