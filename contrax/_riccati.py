"""contrax._riccati — DARE and CARE solvers with implicit derivatives."""

import functools

import jax
import jax.numpy as jnp
from jax import Array

from contrax._care import care  # noqa: F401
from contrax._lyapunov import solve_lyapunov
from contrax._precision import require_x64
from contrax._riccati_common import (
    _dare_residual,
    _lqr_gain,
    _result,
    _symmetrize,
    _validate_matrices,
)
from contrax.solvers import DARE, Lyapunov
from contrax.types import LQRResult

# ── DARE ──────────────────────────────────────────────────────────────────


def _dare_value_iteration(A: Array, B: Array, Q: Array, R: Array, n_iter: int) -> Array:
    """Reference Riccati fixed-point iteration."""

    def step(S, _):
        K = _lqr_gain(A, B, S, R)
        S_new = _symmetrize(Q + A.T @ S @ A - A.T @ S @ B @ K)
        return S_new, None

    S0 = _symmetrize(Q) + jnp.eye(Q.shape[0], dtype=Q.dtype) * 1e-6
    S, _ = jax.lax.scan(step, S0, None, length=n_iter)
    return _symmetrize(S)


def _dare_structured_doubling(
    A: Array, B: Array, Q: Array, R: Array, n_iter: int
) -> Array:
    """Solve DARE with a JAX-native structured doubling iteration.

    Reference: Chiang, Fan, and Lin (2010), "Structured Doubling Algorithm for
    Discrete-Time Algebraic Riccati Equations".
    """
    Q = _symmetrize(Q)
    R = _symmetrize(R)
    eye = jnp.eye(A.shape[0], dtype=A.dtype)
    G = B @ jnp.linalg.solve(R, B.T)
    H = Q

    def step(carry, _):
        A_k, G_k, H_k = carry
        left = eye + G_k @ H_k
        right = eye + H_k @ G_k

        A_next = A_k @ jnp.linalg.solve(left, A_k)
        right_rhs = jnp.concatenate([A_k.T, H_k @ A_k], axis=1)
        right_solve = jnp.linalg.solve(right, right_rhs)
        solve_A_T = right_solve[:, : A_k.shape[0]]
        solve_HA = right_solve[:, A_k.shape[0] :]
        G_next = G_k + A_k @ G_k @ solve_A_T
        H_next = H_k + A_k.T @ solve_HA
        return (A_next, _symmetrize(G_next), _symmetrize(H_next)), None

    (_, _, S), _ = jax.lax.scan(step, (A, G, H), None, length=n_iter)
    return _symmetrize(S)


def _dare_structured_doubling_until_converged(
    A: Array, B: Array, Q: Array, R: Array, max_iter: int, tol: float, atol: float = 0.0
) -> tuple[Array, Array]:
    """Structured doubling with data-dependent stopping.

    This is the forward convergence loop. The public solve differentiates the
    Riccati equation through a custom JVP instead of unrolling this loop.
    """
    Q = _symmetrize(Q)
    R = _symmetrize(R)
    eye = jnp.eye(A.shape[0], dtype=A.dtype)
    G = B @ jnp.linalg.solve(R, B.T)
    H = Q
    initial_delta = jnp.asarray(jnp.inf, dtype=Q.dtype)

    def cond(carry):
        iteration, _, _, _, delta = carry
        return (iteration < max_iter) & (delta > 1.0)

    def step(carry):
        iteration, A_k, G_k, H_k, _ = carry
        left = eye + G_k @ H_k
        right = eye + H_k @ G_k

        A_next = A_k @ jnp.linalg.solve(left, A_k)
        right_rhs = jnp.concatenate([A_k.T, H_k @ A_k], axis=1)
        right_solve = jnp.linalg.solve(right, right_rhs)
        solve_A_T = right_solve[:, : A_k.shape[0]]
        solve_HA = right_solve[:, A_k.shape[0] :]
        G_next = G_k + A_k @ G_k @ solve_A_T
        H_next = H_k + A_k.T @ solve_HA
        G_next = _symmetrize(G_next)
        H_next = _symmetrize(H_next)
        delta = jnp.max(jnp.abs(H_next - H_k)) / (
            atol + tol * jnp.maximum(jnp.max(jnp.abs(H_next)), 1.0)
        )
        return iteration + 1, A_next, G_next, H_next, delta

    iteration, _, _, S, _ = jax.lax.while_loop(
        cond,
        step,
        (0, A, G, H, initial_delta),
    )
    return _symmetrize(S), iteration


def _solve_adjoint_discrete_lyapunov(A_cl: Array, G: Array) -> Array:
    """Solve the Riccati adjoint with bounded automatic dense fallback."""
    X, (_, status) = solve_lyapunov(
        A_cl, _symmetrize(G), continuous=False, solver=Lyapunov()
    )
    return jnp.where(status == 0, _symmetrize(X), jnp.full_like(X, jnp.nan))


def _solve_adjoint_continuous_lyapunov(A_cl: Array, G: Array) -> Array:
    """Solve A_cl X + X A_cl.T = G through the shared equation kernel."""
    X, (_, status) = solve_lyapunov(
        A_cl, -_symmetrize(G), continuous=True, solver=Lyapunov()
    )
    return jnp.where(status == 0, _symmetrize(X), jnp.full_like(X, jnp.nan))


# Reference: implicit differentiation of the Riccati residual (DiLQR, 2025).
# The JVP solves its linearized Lyapunov equation; transposition gives the VJP.
@functools.partial(jax.custom_jvp, nondiff_argnums=(4, 5, 6))
def _dare_solve_with_info(A, B, Q, R, max_iter, rtol, atol):
    return _dare_structured_doubling_until_converged(A, B, Q, R, max_iter, rtol, atol)


@_dare_solve_with_info.defjvp
def _dare_jvp(max_iter, rtol, atol, primals, tangents):
    A, B, Q, R = primals
    S, steps = _dare_solve_with_info(*primals, max_iter, rtol, atol)
    K = _lqr_gain(A, B, S, R)
    A_cl = A - B @ K
    _, forcing = jax.jvp(
        lambda a, b, q, r: _dare_residual(S, a, b, q, r), primals, tangents
    )
    dS = solve_lyapunov(
        A_cl.T, -forcing, continuous=False, solver=Lyapunov(), return_info=False
    )
    dS = _symmetrize(dS)
    return (S, steps), (dS, jnp.zeros(steps.shape, dtype=jax.dtypes.float0))


def _dare_structured_doubling_solve(A, B, Q, R, max_iter, tol):
    return _dare_solve_with_info(A, B, Q, R, max_iter, tol, 0.0)[0]


def dare(
    A: Array,
    B: Array,
    Q: Array,
    R: Array,
    *,
    solver: DARE | None = None,
    throw: bool = False,
) -> LQRResult:
    """Solve the discrete algebraic Riccati equation and design a stabilizing gain.

    Uses structured doubling, scale-aware stopping and residual validation.
    Requires real matrices, symmetric positive-semidefinite Q, positive-definite
    R and float64 mode. Stabilizability and detectability are needed for the
    stabilizing solution. Both JVP and VJP differentiate the equation implicitly
    through a residual-validated Lyapunov solve, without unrolling iterations.
    Supports `jit` and `vmap`; numerical failures have the same status in both.

    Args:
        A: State transition matrix. Shape: `(n, n)`.
        B: Input matrix. Shape: `(n, m)`.
        Q: State cost matrix. Shape: `(n, n)`.
        R: Input cost matrix. Shape: `(m, m)`.
        solver: Structured doubling tolerances and iteration limit.
        throw: Raise on failed validation instead of returning failure diagnostics.

    Returns:
        LQRResult: Gain K, Riccati solution S, poles, status and diagnostics.

    Examples:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import contrax as cx
        >>> import jax.numpy as jnp
        >>> result = cx.dare(jnp.eye(2), jnp.eye(2), jnp.eye(2), jnp.eye(2))
        >>> bool(result.successful)
        True
    """
    require_x64("dare")
    solver = DARE() if solver is None else solver
    if not isinstance(solver, DARE):
        raise TypeError("dare solver must be contrax.solvers.DARE")
    A, B, Q, R = _validate_matrices(A, B, Q, R)
    S, steps = _dare_solve_with_info(
        A, B, Q, R, solver.max_steps, solver.rtol, solver.atol
    )
    return _result(A, B, Q, R, S, steps, solver, continuous=False, throw=throw)
