"""contrax.analysis — structural and transfer analysis helpers."""

import jax
import jax.numpy as jnp
from jax import Array

from contrax._precision import require_x64
from contrax.core import ContLTI, DiscLTI

__all__ = [
    "poles",
    "ctrb",
    "obsv",
    "evalfr",
    "freqresp",
    "dcgain",
    "ctrb_gramian",
    "obsv_gramian",
]


def _finite_horizon_gramian(A_left: Array, forcing: Array, horizon: float) -> Array:
    """Finite-horizon Gramian via Van Loan block-matrix exponential.

    Reference: Van Loan (1978), "Computing Integrals Involving the Matrix
    Exponential", IEEE TAC.
    """
    n = A_left.shape[0]
    block = jnp.block(
        [
            [A_left, forcing],
            [jnp.zeros_like(A_left), -A_left.T],
        ]
    )
    exp_block = jax.scipy.linalg.expm(block * horizon)
    phi11 = exp_block[:n, :n]
    phi12 = exp_block[:n, n:]
    return phi12 @ phi11.T


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
    return jnp.transpose(blocks, (1, 0, 2)).reshape(A.shape[0], -1)


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
    return blocks.reshape(-1, A.shape[0])


def evalfr(sys: DiscLTI | ContLTI, point: complex | Array) -> Array:
    """Evaluate the transfer matrix at a complex frequency point.

    For continuous systems this computes `G(s) = C (sI - A)^-1 B + D`. For
    discrete systems it computes `G(z) = C (zI - A)^-1 B + D`.

    This is the first transfer-evaluation helper in Contrax rather than a full
    transfer-function algebra layer. It is intended for design-time analysis,
    spot checks, and building higher-level frequency-response tools.

    Args:
        sys: Continuous or discrete LTI system.
        point: Complex evaluation point `s` or `z`. Shape: scalar.

    Returns:
        Array: Transfer matrix evaluated at `point`. Shape: `(p, m)`.
    """
    point = jnp.asarray(point, dtype=jnp.result_type(sys.A.dtype, 1j))
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

    This is a basic state-space frequency-response helper. It returns the
    complex transfer matrix at each frequency but does not yet format Bode or
    Nyquist plots for you.

    Args:
        sys: Continuous or discrete LTI system.
        omega: Angular frequencies in rad/s. Shape: `(k,)`.

    Returns:
        Array: Complex frequency response. Shape: `(k, p, m)`.
    """
    omega = jnp.asarray(omega, dtype=float)
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
    point = jnp.array(0.0) if isinstance(sys, ContLTI) else jnp.array(1.0)
    return evalfr(sys, point)


def ctrb_gramian(sys: ContLTI, t: float = 10.0) -> Array:
    """Compute the finite-horizon controllability Gramian.

    Uses the Van Loan block-matrix exponential construction to evaluate the
    finite-horizon controllability Gramian in closed form for continuous-time
    linear systems.

    Args:
        sys: Continuous-time linear system.
        t: Integration horizon. Shape: scalar.

    Returns:
        Array: Finite-horizon controllability Gramian. Shape: `(n, n)`.
    """
    require_x64("ctrb_gramian")
    A, B = sys.A, sys.B
    return _finite_horizon_gramian(A, B @ B.T, t)


def obsv_gramian(sys: ContLTI, t: float = 10.0) -> Array:
    """Compute the finite-horizon observability Gramian.

    Uses the same Van Loan block-matrix exponential construction as
    `ctrb_gramian()`, applied to the dual system.

    Args:
        sys: Continuous-time linear system.
        t: Integration horizon. Shape: scalar.

    Returns:
        Array: Finite-horizon observability Gramian. Shape: `(n, n)`.
    """
    require_x64("obsv_gramian")
    return _finite_horizon_gramian(sys.A.T, sys.C.T @ sys.C, t)
