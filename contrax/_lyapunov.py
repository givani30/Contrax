"""Shared Lyapunov kernels and implicit differentiation."""

import equinox as eqx
import jax
import jax.numpy as jnp

from contrax._precision import require_x64
from contrax.solvers import Lyapunov, SolverStatus
from contrax.types import LyapunovResult


def _operator(A, X, continuous):
    return -A @ X - X @ A.T if continuous else X - A @ X @ A.T


def _residual(A, Q, X, continuous):
    residual = jnp.linalg.norm(_operator(A, X, continuous) - Q)
    # Bound each term separately so cancellation does not define the scale.
    if continuous:
        scale = jnp.linalg.norm(A @ X) + jnp.linalg.norm(X @ A.T)
    else:
        scale = jnp.linalg.norm(X) + jnp.linalg.norm(A @ X @ A.T)
    scale = scale + jnp.linalg.norm(Q)
    return residual, scale


def _dense(A, Q, continuous):
    n = A.shape[0]
    eye = jnp.eye(n, dtype=A.dtype)
    matrix = (
        -jnp.kron(A, eye) - jnp.kron(eye, A)
        if continuous
        else jnp.eye(n * n, dtype=A.dtype) - jnp.kron(A, A)
    )
    return jnp.linalg.solve(matrix, Q.reshape(-1)).reshape(Q.shape)


def _doubling(A, Q, continuous, solver):
    # Reference: Smith (1968), "Matrix equation XA + BX = C", SIAM J. Appl.
    # Math. Squared Smith iteration sums 2**k terms in k steps. The Cayley
    # transform maps a Hurwitz continuous equation to a Schur-stable one.
    if continuous:
        shift = jnp.maximum(jnp.linalg.norm(A, ord=jnp.inf), 1)
        eye = jnp.eye(A.shape[0], dtype=A.dtype)
        left = shift * eye - A
        transition = jnp.linalg.solve(left, shift * eye + A)
        rhs = jnp.linalg.solve(left, Q)
        X = 2 * shift * jnp.linalg.solve(left, rhs.T).T
    else:
        transition, X = A, Q

    def condition(carry):
        steps, _, X = carry
        residual, scale = _residual(A, Q, X, continuous)
        return (
            (steps < solver.max_steps)
            & jnp.all(jnp.isfinite(X))
            & (residual > solver.atol + solver.rtol * scale)
        )

    def step(carry):
        steps, transition, X = carry
        return steps + 1, transition @ transition, X + transition @ X @ transition.T

    steps, _, X = jax.lax.while_loop(condition, step, (0, transition, X))
    return X, steps


def _solve(A, Q, continuous, solver):
    if solver.method == "dense":
        X, steps = _dense(A, Q, continuous), jnp.asarray(0)
    else:
        eigenvalues = jnp.linalg.eigvals(A)
        stable = jnp.all(
            jnp.real(eigenvalues) < 0 if continuous else jnp.abs(eigenvalues) < 1
        )
        X, steps = jax.lax.cond(
            stable,
            lambda: _doubling(A, Q, continuous, solver),
            lambda: (jnp.full_like(Q, jnp.nan), jnp.asarray(0)),
        )
        residual, scale = _residual(A, Q, X, continuous)
        good = (
            jnp.isfinite(residual)
            & jnp.isfinite(scale)
            & (residual <= solver.atol + solver.rtol * scale)
        )
        if solver.method == "auto" and A.shape[0] <= solver.dense_limit:
            X = jax.lax.cond(good, lambda: X, lambda: _dense(A, Q, continuous))
    residual, scale = _residual(A, Q, X, continuous)
    finite = jnp.all(jnp.isfinite(X)) & jnp.isfinite(residual) & jnp.isfinite(scale)
    success = finite & (residual <= solver.atol + solver.rtol * scale)
    status = jnp.where(
        success,
        int(SolverStatus.successful),
        jnp.where(
            finite & (steps >= solver.max_steps) & (solver.method != "dense"),
            int(SolverStatus.max_steps_reached),
            int(SolverStatus.numerical_failure),
        ),
    )
    if solver.method == "doubling" or (
        solver.method == "auto" and A.shape[0] > solver.dense_limit
    ):
        status = jnp.where(stable, status, int(SolverStatus.invalid_problem))
    return X, (steps, status)


def solve_lyapunov(A, Q, *, continuous, solver):
    """Solve the full matrix operator with implicit JVP and transposed solve."""
    # custom_linear_solve differentiates the equation, never its stopping loop.
    return jax.lax.custom_linear_solve(
        lambda X: _operator(A, X, continuous),
        Q,
        solve=lambda _, rhs: _solve(A, rhs, continuous, solver),
        transpose_solve=lambda _, rhs: _solve(A.T, rhs, continuous, solver),
        has_aux=True,
    )


def lyapunov(A, Q, *, continuous, solver=None, throw=False):
    require_x64("lyap" if continuous else "dlyap")
    solver = Lyapunov() if solver is None else solver
    if not isinstance(solver, Lyapunov):
        raise TypeError("solver must be contrax.solvers.Lyapunov")
    A, Q = jnp.asarray(A), jnp.asarray(Q)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or Q.shape != A.shape:
        raise ValueError("A and Q must have identical square matrix shapes")
    if jnp.iscomplexobj(A) or jnp.iscomplexobj(Q):
        raise ValueError("Lyapunov solvers require real matrices")
    dtype = jnp.result_type(A, Q, 1.0)
    A, Q = A.astype(dtype), Q.astype(dtype)
    if A.shape[0] == 0:
        return LyapunovResult(Q, jnp.asarray(0), jnp.asarray(0), jnp.asarray(0.0))
    X, (steps, status) = solve_lyapunov(A, Q, continuous=continuous, solver=solver)
    residual, _ = _residual(A, Q, X, continuous)
    result = LyapunovResult(X=X, status=status, num_steps=steps, residual_norm=residual)
    return (
        eqx.error_if(
            result, status != 0, "Lyapunov solve failed; inspect status/residual"
        )
        if throw
        else result
    )
