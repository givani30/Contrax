"""contrax.analysis — structural and transfer analysis helpers."""

import jax
import jax.numpy as jnp
from jax import Array

from contrax._gramians import ctrb_gramian, obsv_gramian  # noqa: F401
from contrax._lyapunov import lyapunov
from contrax._quadratic import quadratic_rollout_cost  # noqa: F401
from contrax._zeros import zeros  # noqa: F401
from contrax.core import ContLTI, DiscLTI
from contrax.solvers import Lyapunov
from contrax.types import LyapunovResult

__all__ = [
    "poles",
    "zeros",
    "ctrb",
    "obsv",
    "evalfr",
    "freqresp",
    "dcgain",
    "ctrb_gramian",
    "obsv_gramian",
    "lyap",
    "dlyap",
    "quadratic_rollout_cost",
]


def poles(sys: DiscLTI | ContLTI) -> Array:
    """Return the poles of a state-space system.

    For the current state-space-only Contrax surface, poles are the eigenvalues
    of the state matrix `A`.

    This is a lightweight analysis helper that composes with `jit` and `vmap`
    for fixed-size systems. The return dtype is generally complex, even for
    real-valued systems with real poles.

    Args:
        sys: Continuous or discrete LTI system.

    Returns:
        Array: System poles. Shape: `(n,)`.
    """
    return jnp.linalg.eigvals(sys.A)


def ctrb(sys: DiscLTI | ContLTI) -> Array:
    """Construct the controllability matrix of a state-space system.

    Forms `[B, AB, A^2 B, ..., A^{n-1} B]` for a continuous or discrete linear
    system.

    This is a structural analysis primitive rather than a yes-or-no
    controllability test by itself. Users typically inspect the downstream rank
    or conditioning of the returned matrix. The implementation uses
    `jax.lax.scan` so it composes with transforms, but it is still intended
    primarily for design-time analysis rather than tight inner loops.

    Args:
        sys: Continuous or discrete LTI system.

    Returns:
        Array: Controllability matrix. Shape: `(n, n * m)`.
    """
    A, B = sys.A, sys.B
    n = A.shape[0]

    def step(block, _):
        return A @ block, block

    _, blocks = jax.lax.scan(step, B, None, length=n)
    return jnp.transpose(blocks, (1, 0, 2)).reshape(n, n * B.shape[1])


def obsv(sys: DiscLTI | ContLTI) -> Array:
    """Construct the observability matrix of a state-space system.

    Forms `[C; CA; CA^2; ...; CA^{n-1}]` for a continuous or discrete linear
    system.

    This is a structural analysis primitive rather than a yes-or-no
    observability test by itself. Users typically inspect the downstream rank
    or conditioning of the returned matrix. The implementation uses
    `jax.lax.scan` so it composes with transforms, but it is still intended
    primarily for design-time analysis rather than tight inner loops.

    Args:
        sys: Continuous or discrete LTI system.

    Returns:
        Array: Observability matrix. Shape: `(n * p, n)`.
    """
    A, C = sys.A, sys.C
    n = A.shape[0]

    def step(block, _):
        return block @ A, block

    _, blocks = jax.lax.scan(step, C, None, length=n)
    return blocks.reshape(n * C.shape[0], n)


def evalfr(sys: DiscLTI | ContLTI, point: complex | Array) -> Array:
    """Evaluate the transfer matrix at a complex frequency point.

    For continuous systems this computes `G(s) = C (sI - A)^-1 B + D`. For
    discrete systems it computes `G(z) = C (zI - A)^-1 B + D`.

    Supports jit, vmap, JVP and VJP away from realization poles. The resolvent
    must be nonsingular; evaluation at a realization pole may return nonfinite
    values even when a nonminimal realization has a removable cancellation.
    The matrices and evaluation point determine the promoted complex dtype.

    Args:
        sys: Continuous or discrete LTI system.
        point: Complex evaluation point `s` or `z`. Shape: scalar.

    Returns:
        Array: Transfer matrix evaluated at `point`. Shape: `(p, m)`.
    """
    if not isinstance(sys, (ContLTI, DiscLTI)):
        raise TypeError("evalfr requires a continuous or discrete LTI model")
    point = jnp.asarray(point, dtype=jnp.result_type(sys.A, point, 1j))
    if point.ndim != 0:
        raise ValueError("point must be scalar; use vmap or freqresp for a grid")
    n = sys.A.shape[0]
    identity = jnp.eye(n, dtype=point.dtype)
    A = sys.A.astype(point.dtype)
    B = sys.B.astype(point.dtype)
    C = sys.C.astype(point.dtype)
    D = sys.D.astype(point.dtype)
    resolvent_rhs = jnp.linalg.solve(point * identity - A, B)
    return C @ resolvent_rhs + D


def freqresp(sys: DiscLTI | ContLTI, omega: Array) -> Array:
    """Evaluate the frequency response over angular frequencies.

    Continuous systems are evaluated on the imaginary axis at `s = j omega`.
    Discrete systems are evaluated on the unit circle at
    `z = exp(j omega dt)`.

    Returns the complex transfer matrix at each frequency, with jit/vmap and
    forward/reverse derivatives wherever the resolvents are nonsingular.
    An empty frequency vector returns shape (0, p, m).

    Args:
        sys: Continuous or discrete LTI system.
        omega: Angular frequencies in rad/s. Shape: `(k,)`.

    Returns:
        Array: Complex frequency response. Shape: `(k, p, m)`.
    """
    omega = jnp.asarray(omega)
    if omega.ndim != 1 or jnp.iscomplexobj(omega):
        raise ValueError("omega must be a real vector of angular frequencies")
    omega = omega.astype(jnp.result_type(sys.A.real, omega, 1.0))
    if isinstance(sys, ContLTI):
        points = 1j * omega
    else:
        points = jnp.exp(1j * omega * sys.dt)
    return jax.vmap(lambda point: evalfr(sys, point))(points)


def dcgain(sys: DiscLTI | ContLTI) -> Array:
    """Return the zero-frequency or steady-state gain of an LTI system.

    For continuous systems this evaluates `G(0)`. For discrete systems this
    evaluates `G(1)`.

    This helper assumes the corresponding resolvent is nonsingular. It is a
    useful quick check for low-frequency behavior, but it is not by itself a
    stability certificate.

    Args:
        sys: Continuous or discrete LTI system.

    Returns:
        Array: DC gain matrix. Shape: `(p, m)`.
    """
    point = 0.0 if isinstance(sys, ContLTI) else 1.0
    return evalfr(sys, point)


def lyap(
    A: Array, Q: Array, *, solver: Lyapunov | None = None, throw: bool = False
) -> LyapunovResult:
    """Solve the continuous Lyapunov equation AX + XA^T + Q = 0.

    The default uses squared Smith iteration after a Cayley transform, with
    residual validation and dense fallback bounded to eight states. Select
    `Lyapunov(method="dense")` for general uniquely solvable equations,
    including unstable systems. Uniqueness requires every eigenvalue sum
    `lambda_i + lambda_j` to be nonzero. Symmetric Q gives symmetric X.

    Requires float64 mode. Supports `jit`, `vmap`, forward and reverse
    differentiation through the equation rather than the convergence loop.
    Numerical failures return diagnostics unless `throw=True`.

    Args:
        A: Real state matrix. Shape: `(n, n)`.
        Q: Real forcing matrix. Shape: `(n, n)`.
        solver: Algorithm, tolerances and iteration limit.
        throw: Raise on numerical failure, including under `jit`.

    Returns:
        LyapunovResult: Solution `X`, status, iteration count and residual norm.

    Examples:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import contrax as cx
        >>> import jax.numpy as jnp
        >>> result = cx.lyap(-jnp.eye(2), jnp.eye(2))
        >>> bool(result.successful)
        True
    """
    return lyapunov(A, Q, continuous=True, solver=solver, throw=throw)


def dlyap(
    A: Array, Q: Array, *, solver: Lyapunov | None = None, throw: bool = False
) -> LyapunovResult:
    """Solve the discrete Lyapunov equation AXA^T - X + Q = 0.

    The default uses squared Smith iteration, with residual validation and
    dense fallback bounded to eight states. Select `Lyapunov(method="dense")`
    for general uniquely solvable equations, including unstable systems.
    Uniqueness requires every eigenvalue product `lambda_i * lambda_j` to
    differ from one. Symmetric Q gives symmetric X.

    Requires float64 mode. Supports `jit`, `vmap`, forward and reverse
    differentiation implicitly through the equation. A failed iteration on a
    large system never allocates a dense Kronecker operator automatically.

    Args:
        A: Real state transition matrix. Shape: `(n, n)`.
        Q: Real forcing matrix. Shape: `(n, n)`.
        solver: Algorithm, tolerances and iteration limit.
        throw: Raise on numerical failure, including under `jit`.

    Returns:
        LyapunovResult: Solution `X`, status, iteration count and residual norm.

    Examples:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import contrax as cx
        >>> import jax.numpy as jnp
        >>> result = cx.dlyap(0.5 * jnp.eye(2), jnp.eye(2))
        >>> bool(result.successful)
        True
    """
    return lyapunov(A, Q, continuous=False, solver=solver, throw=throw)
