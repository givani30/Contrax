"""Hamiltonian CARE solve, Newton refinement and implicit derivatives."""

import functools

import jax
import jax.numpy as jnp
from jax import Array

from contrax._lyapunov import _solve as _lyapunov_forward
from contrax._lyapunov import solve_lyapunov
from contrax._precision import require_x64
from contrax._riccati_common import (
    _care_residual,
    _result,
    _symmetrize,
    _validate_matrices,
)
from contrax.solvers import CARE, Lyapunov
from contrax.types import LQRResult

# ── CARE via Hamiltonian eigendecomposition ───────────────────────────────


def _care_hamiltonian_solve(A: Array, B: Array, Q: Array, R: Array) -> Array:
    """Solve CARE from the stable invariant subspace of the Hamiltonian.

    Reference: Laub (1979), "A Schur Method for Solving Algebraic Riccati
    Equations". The Hamiltonian eigensolve supplies a basis for its stable
    invariant subspace.
    """
    Q = _symmetrize(Q)
    R = _symmetrize(R)
    R_inv_BT = jnp.linalg.solve(R, B.T)
    hamiltonian = jnp.block(
        [
            [A, -(B @ R_inv_BT)],
            [-Q, -A.T],
        ]
    )
    eigenvalues, eigenvectors = jnp.linalg.eig(hamiltonian)
    n = A.shape[0]
    stable_idx = jnp.argsort(jnp.real(eigenvalues))[:n]
    stable_vectors = eigenvectors[:, stable_idx]
    X1 = stable_vectors[:n, :]
    X2 = stable_vectors[n:, :]
    S = jnp.real(jnp.linalg.solve(X1.T, X2.T).T)
    return _symmetrize(S)


# Reference: Kleinman (1968), "On an iterative technique for Riccati equation
# computations", IEEE TAC. Newton corrections solve the residual Lyapunov equation.
def _care_refined(A, B, Q, R, refinement_steps):
    S = _care_hamiltonian_solve(A, B, Q, R)

    def step(carry, _):
        S, accepted = carry
        K = jnp.linalg.solve(R, B.T @ S)
        residual = _care_residual(S, A, B, Q, R)
        # Refinement is inside the Riccati custom-JVP boundary and only needs
        # the forward equation kernel, without a nested implicit AD primitive.
        correction, (_, status) = _lyapunov_forward(
            (A - B @ K).T, residual, True, Lyapunov()
        )
        candidate = _symmetrize(S + correction)
        candidate_K = jnp.linalg.solve(R, B.T @ candidate)
        stable = jnp.all(jnp.real(jnp.linalg.eigvals(A - B @ candidate_K)) < 0)
        improves = jnp.linalg.norm(
            _care_residual(candidate, A, B, Q, R)
        ) < jnp.linalg.norm(residual)
        accept = (status == 0) & stable & improves & jnp.all(jnp.isfinite(candidate))
        return (jnp.where(accept, candidate, S), accepted + accept.astype(int)), None

    (S, accepted), _ = jax.lax.scan(
        step, (S, jnp.asarray(0)), None, length=refinement_steps
    )
    return S, accepted


@functools.partial(jax.custom_jvp, nondiff_argnums=(4,))
def _care_with_info(A, B, Q, R, refinement_steps):
    return _care_refined(A, B, Q, R, refinement_steps)


@_care_with_info.defjvp
def _care_jvp(refinement_steps, primals, tangents):
    A, B, Q, R = primals
    S, steps = _care_with_info(*primals, refinement_steps)
    K = jnp.linalg.solve(R, B.T @ S)
    _, forcing = jax.jvp(
        lambda a, b, q, r: _care_residual(S, a, b, q, r), primals, tangents
    )
    dS = solve_lyapunov(
        (A - B @ K).T, forcing, continuous=True, solver=Lyapunov(), return_info=False
    )
    dS = _symmetrize(dS)
    return (S, steps), (dS, jnp.zeros(steps.shape, dtype=jax.dtypes.float0))


def _care_solve(A, B, Q, R):
    return _care_with_info(A, B, Q, R, 3)[0]


def care(
    A: Array,
    B: Array,
    Q: Array,
    R: Array,
    *,
    solver: CARE | None = None,
    throw: bool = False,
) -> LQRResult:
    """Solve the continuous algebraic Riccati equation and design a stabilizing gain.

    Solves $A^T S + S A - S B R^{-1} B^T S + Q = 0$ and computes
    $K = R^{-1} B^T S$.

    The baseline solver starts from a Hamiltonian stable-subspace candidate and
    applies residual-decreasing Newton corrections. Poor separation of the
    stable subspace can make the candidate inaccurate; residual and closed-loop
    stability checks report such cases as failures. The stabilizing solution
    requires a stabilizable pair (A, B) and detectable pair (Q**1/2, A).

    Uses real matrices with symmetric positive-semidefinite Q and
    positive-definite R, and requires float64 precision. JVPs and VJPs use
    implicit differentiation of the Riccati equation.

    Args:
        A: State matrix. Shape: `(n, n)`.
        B: Input matrix. Shape: `(n, m)`.
        Q: State cost matrix. Shape: `(n, n)`.
        R: Input cost matrix. Shape: `(m, m)`.
        solver: Hamiltonian candidate refinement and residual tolerances.
        throw: Raise on failed validation instead of returning failure diagnostics.

    Returns:
        LQRResult: Gain K, Riccati solution S, poles, status and diagnostics.

    Examples:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import contrax as cx
        >>> import jax.numpy as jnp
        >>> result = cx.care(-jnp.eye(2), jnp.eye(2), jnp.eye(2), jnp.eye(2))
        >>> bool(result.successful)
        True
    """
    require_x64("care")
    solver = CARE() if solver is None else solver
    if not isinstance(solver, CARE):
        raise TypeError("care solver must be contrax.solvers.CARE")
    A, B, Q, R = _validate_matrices(A, B, Q, R)
    S, steps = _care_with_info(A, B, Q, R, solver.refinement_steps)
    return _result(A, B, Q, R, S, steps, solver, continuous=True, throw=throw)
