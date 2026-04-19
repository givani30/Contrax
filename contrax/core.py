"""contrax.core — system types, discretization, and linearization."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from contrax._precision import require_x64
from contrax.nonlinear import (
    DynamicsLike,
    OutputFn,
    _coerce_dynamics,
    _coerce_observation,
    _is_system_model,
)


class ContLTI(eqx.Module):
    """Continuous-time LTI system: ẋ = Ax + Bu, y = Cx + Du."""

    A: Array  # (n, n)
    B: Array  # (n, m)
    C: Array  # (p, n)
    D: Array  # (p, m)

    def __matmul__(self, other):
        from contrax.interconnect import series

        return series(self, other)

    def __add__(self, other):
        from contrax.interconnect import parallel

        return parallel(self, other)

    def __sub__(self, other):
        from contrax.interconnect import parallel

        return parallel(self, other, sign=-1.0)


class DiscLTI(eqx.Module):
    """Discrete-time LTI system: x_{k+1} = Ax_k + Bu_k, y_k = Cx_k + Du_k."""

    A: Array  # (n, n) — state transition matrix (Phi)
    B: Array  # (n, m) — input matrix (Gamma)
    C: Array  # (p, n)
    D: Array  # (p, m)
    dt: Array  # scalar — sampling period as plain JAX array, NOT static

    def __matmul__(self, other):
        from contrax.interconnect import series

        return series(self, other)

    def __add__(self, other):
        from contrax.interconnect import parallel

        return parallel(self, other)

    def __sub__(self, other):
        from contrax.interconnect import parallel

        return parallel(self, other, sign=-1.0)


def ss(A: ArrayLike, B: ArrayLike, C: ArrayLike, D: ArrayLike) -> ContLTI:
    """Construct a continuous-time state-space system.

    Wraps continuous-time matrices in a
    [ContLTI][contrax.systems.ContLTI] pytree with MATLAB-familiar
    `ss(A, B, C, D)` ergonomics.

    This is a lightweight constructor rather than a solver. It is safe under
    normal JAX transforms with fixed-shape array inputs because it only wraps
    validated arrays into a data-only `eqx.Module`.

    Args:
        A: State matrix. Shape: `(n, n)`.
        B: Input matrix. Shape: `(n, m)`.
        C: Output matrix. Shape: `(p, n)`.
        D: Feedthrough matrix. Shape: `(p, m)`.

    Returns:
        [ContLTI][contrax.systems.ContLTI]: A continuous-time system with
            array-valued `A`, `B`, `C`, and `D` fields.

    Raises:
        AssertionError: If the matrix shapes are inconsistent.

    Examples:
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> sys = cx.ss(
        ...     jnp.array([[0.0, 1.0], [0.0, 0.0]]),
        ...     jnp.array([[0.0], [1.0]]),
        ...     jnp.eye(2),
        ...     jnp.zeros((2, 1)),
        ... )
    """
    A = jnp.asarray(A, dtype=float)
    B = jnp.asarray(B, dtype=float)
    C = jnp.asarray(C, dtype=float)
    D = jnp.asarray(D, dtype=float)
    n = A.shape[0]
    assert A.shape == (n, n), f"A must be square, got {A.shape}"
    assert B.shape[0] == n, f"B rows must match A, got {B.shape}"
    assert C.shape[1] == n, f"C cols must match A, got {C.shape}"
    assert D.shape == (C.shape[0], B.shape[1]), f"D shape mismatch, got {D.shape}"
    return ContLTI(A=A, B=B, C=C, D=D)


def dss(
    A: ArrayLike,
    B: ArrayLike,
    C: ArrayLike,
    D: ArrayLike,
    dt: ArrayLike,
) -> DiscLTI:
    """Construct a discrete-time state-space system.

    Wraps discrete-time matrices in a
    [DiscLTI][contrax.systems.DiscLTI] pytree with MATLAB-familiar
    `ss(A, B, C, D, dt)` ergonomics.

    This is a lightweight constructor rather than a solver. The sampling period
    is stored as a plain array leaf instead of static metadata, which keeps
    batching over systems more natural in JAX.

    Args:
        A: State transition matrix. Shape: `(n, n)`.
        B: Input matrix. Shape: `(n, m)`.
        C: Output matrix. Shape: `(p, n)`.
        D: Feedthrough matrix. Shape: `(p, m)`.
        dt: Sampling period. Shape: scalar.

    Returns:
        [DiscLTI][contrax.systems.DiscLTI]: A discrete-time system with
            array-valued `A`, `B`, `C`, `D`, and `dt` fields.

    Raises:
        AssertionError: If the matrix shapes are inconsistent.

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
    """
    A = jnp.asarray(A, dtype=float)
    B = jnp.asarray(B, dtype=float)
    C = jnp.asarray(C, dtype=float)
    D = jnp.asarray(D, dtype=float)
    dt = jnp.asarray(dt, dtype=float)
    n = A.shape[0]
    assert A.shape == (n, n)
    assert B.shape[0] == n
    assert C.shape[1] == n
    assert D.shape == (C.shape[0], B.shape[1])
    return DiscLTI(A=A, B=B, C=C, D=D, dt=dt)


# ── custom VJP for matrix exponential ──────────────────────────────────────
# Native autodiff through jax.scipy.linalg.expm (scaling-and-squaring)
# accumulates catastrophic numerical error for stiff systems.
# We use the exact analytical gradient via the augmented block matrix method.
# Reference: Al-Mohy & Higham (2009) — Fréchet derivative of the matrix exp.


@jax.custom_vjp
def _safe_expm(M: Array) -> Array:
    return jax.scipy.linalg.expm(M)


def _safe_expm_fwd(M: Array):
    eM = _safe_expm(M)
    return eM, (M, eM)


def _safe_expm_bwd(res, g):
    M, eM = res
    n = M.shape[0]
    # VJP is the adjoint Frechet derivative. Under the Frobenius inner product,
    # L_exp(M)^*(G) = L_exp(M.T, G).
    aug = jnp.block([[M.T, g], [jnp.zeros_like(M), M.T]])
    eaug = jax.scipy.linalg.expm(aug)
    return (eaug[:n, n:],)


_safe_expm.defvjp(_safe_expm_fwd, _safe_expm_bwd)


# ── c2d ────────────────────────────────────────────────────────────────────


def c2d(sys: ContLTI, dt: float, method: str = "zoh") -> DiscLTI:
    """Discretize a continuous-time state-space system.

    Converts a [ContLTI][contrax.systems.ContLTI] into a
    [DiscLTI][contrax.systems.DiscLTI] using either zero-order hold or the
    bilinear Tustin transform.

    The `zoh` path is the stronger current implementation and is differentiated
    through a custom VJP on the matrix exponential. `tustin` is available as a
    convenience discretization.

    Args:
        sys: Continuous-time system.
        dt: Sampling period. Shape: scalar.
        method: Discretization method. Supported values are `"zoh"` and
            `"tustin"`.

    Returns:
        [DiscLTI][contrax.systems.DiscLTI]: A discrete-time realization with
            the same output matrices and the requested sampling period.

    Raises:
        ValueError: If `method` is not one of the supported discretizations.

    Examples:
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> sys_c = cx.ss(
        ...     jnp.array([[0.0, 1.0], [0.0, 0.0]]),
        ...     jnp.array([[0.0], [1.0]]),
        ...     jnp.eye(2),
        ...     jnp.zeros((2, 1)),
        ... )
        >>> sys_d = cx.c2d(sys_c, dt=0.05)
    """
    require_x64("c2d")
    if method == "zoh":
        return _c2d_zoh(sys, dt)
    elif method == "tustin":
        return _c2d_tustin(sys, dt)
    else:
        raise ValueError(f"Unknown discretization method: {method!r}")


def _c2d_zoh(sys: ContLTI, dt: float) -> DiscLTI:
    n, m = sys.A.shape[0], sys.B.shape[1]
    # Augmented matrix: [[A, B], [0, 0]] * dt
    # expm of this gives [[Phi, Gamma], [0, I]]
    zeros_bot = jnp.zeros((m, n + m))
    aug = jnp.block([[sys.A, sys.B], [zeros_bot]]) * dt
    eM = _safe_expm(aug)
    Phi = eM[:n, :n]
    Gamma = eM[:n, n:]
    dt_arr = jnp.asarray(dt, dtype=sys.A.dtype)
    return DiscLTI(A=Phi, B=Gamma, C=sys.C, D=sys.D, dt=dt_arr)


def _c2d_tustin(sys: ContLTI, dt: float) -> DiscLTI:
    """Bilinear (Tustin) discretization: s ≈ (2/dt)(z-1)/(z+1).

    This uses the state-space bilinear transform that preserves the continuous
    transfer function under the Tustin frequency mapping. Leaving ``C`` and
    ``D`` unchanged would produce the correct state update map but the wrong
    discrete transfer behavior.

    Reference: the standard bilinear state-space realization used by
    MATLAB/Octave/Scipy ``c2d(..., "tustin"/"bilinear")``.
    """
    n = sys.A.shape[0]
    eye = jnp.eye(n)
    alpha = dt / 2.0
    LHS = eye - alpha * sys.A
    RHS = eye + alpha * sys.A
    Phi = jnp.linalg.solve(LHS, RHS)
    Gamma = jnp.linalg.solve(LHS, sys.B * dt)
    C_d = jnp.linalg.solve(LHS.T, sys.C.T).T
    D_d = sys.D + 0.5 * sys.C @ Gamma
    dt_arr = jnp.asarray(dt, dtype=sys.A.dtype)
    return DiscLTI(A=Phi, B=Gamma, C=C_d, D=D_d, dt=dt_arr)


# ── linearization ──────────────────────────────────────────────────────────
# Exact Jacobians via forward-mode AD (jacfwd). For control-scale systems
# (n, m typically < 100), forward mode costs n+m JVPs — cheaper than
# jacrev's p cotangent solves when n < output_dim.
# Both functions are jit- and vmap-compatible; the equilibrium arrays are
# treated as regular tracers.


def linearize(
    model_or_f: DynamicsLike,
    x0: Array,
    u0: Array,
    *,
    output: OutputFn | None = None,
    t: ArrayLike = 0.0,
) -> ContLTI:
    """Linearize dynamics and output maps into a continuous state-space model.

    Computes `A`, `B`, `C`, and `D` Jacobians at `(x0, u0)` using forward-mode
    automatic differentiation and returns them as a
    [ContLTI][contrax.systems.ContLTI].

    This is the main bridge from nonlinear plant code into the linear control
    stack. A common workflow is:

    ```python
    sys_lin = cx.linearize(nonlinear_sys, x_eq, u_eq)
    sys_d   = cx.c2d(sys_lin, dt=0.05)
    K       = cx.lqr(sys_d, Q, R).K
    cx.simulate(nonlinear_sys, x0, lambda t, x: -K @ x, ...)
    ```

    Args:
        model_or_f: A `NonlinearSystem` or a pure JAX dynamics function with
            signature `(t, x, u) → x_dot`.
        x0: State operating point. Shape: `(n,)`.
        u0: Input operating point. Shape: `(m,)`.
        output: Output function `(x, u) → y` for plain dynamics callables.
            When omitted, defaults to the full state (`C = I`, `D = 0`).
            Omit this when passing a system model with its own output map.
        t: Operating-point time. Relevant for time-varying system models.

    Returns:
        [ContLTI][contrax.systems.ContLTI]: Linearized continuous-time
            state-space model with Jacobian matrices `A`, `B`, `C`, `D`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> def dynamics(t, x, u):
        ...     return jnp.array([x[1], -jnp.sin(x[0]) + u[0]])
        >>> sys_lin = cx.linearize(dynamics, jnp.zeros(2), jnp.zeros(1))
    """
    t_arr = jnp.asarray(t, dtype=x0.dtype)

    if _is_system_model(model_or_f):
        if output is not None:
            raise ValueError("linearize(sys, x0, u0) does not accept output=.")
        f = model_or_f.dynamics
        h = model_or_f.output
    else:
        f = model_or_f
        if output is not None:
            def h(t_: Array, x: Array, u_: Array) -> Array:
                del t_
                return output(x, u_)
        else:
            def h(t_: Array, x: Array, u_: Array) -> Array:
                del t_, u_
                return x

    A = jax.jacfwd(f, argnums=1)(t_arr, x0, u0)
    B = jax.jacfwd(f, argnums=2)(t_arr, x0, u0)
    C = jax.jacfwd(h, argnums=1)(t_arr, x0, u0)
    D = jax.jacfwd(h, argnums=2)(t_arr, x0, u0)
    return ContLTI(A=A, B=B, C=C, D=D)


linearize_ss = linearize
