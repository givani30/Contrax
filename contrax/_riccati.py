"""contrax._riccati — DARE and CARE solvers with implicit derivatives."""

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from contrax._lyapunov import solve_lyapunov
from contrax._precision import require_x64
from contrax.solvers import CARE, DARE, Lyapunov, SolverStatus
from contrax.types import LQRResult


def _symmetrize(X: Array) -> Array:
    return (X + X.T) / 2


# ── DARE ──────────────────────────────────────────────────────────────────


def _lqr_gain(A: Array, B: Array, S: Array, R: Array) -> Array:
    BtS = B.T @ S
    return jnp.linalg.solve(R + BtS @ B, BtS @ A)


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

    Reverse-mode gradients through `lax.while_loop` require a custom VJP, so
    this is a forward-solver candidate rather than the public gradient path.
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


def _dare_residual(S: Array, A: Array, B: Array, Q: Array, R: Array) -> Array:
    K = _lqr_gain(A, B, S, R)
    return S - Q - A.T @ S @ A + A.T @ S @ B @ K


def _care_residual(S: Array, A: Array, B: Array, Q: Array, R: Array) -> Array:
    R_inv_BT = jnp.linalg.solve(R, B.T)
    return _symmetrize(Q + A.T @ S + S @ A - S @ B @ R_inv_BT @ S)


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
    dS, _ = solve_lyapunov(A_cl.T, -forcing, continuous=False, solver=Lyapunov())
    dS = _symmetrize(dS)
    return (S, steps), (dS, jnp.zeros(steps.shape, dtype=jax.dtypes.float0))


def _dare_structured_doubling_solve(A, B, Q, R, max_iter, tol):
    return _dare_solve_with_info(A, B, Q, R, max_iter, tol, 0.0)[0]


def _validate_matrices(A, B, Q, R):
    A, B, Q, R = (jnp.asarray(x) for x in (A, B, Q, R))
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("A must be a nonempty square matrix")
    n = A.shape[0]
    if B.ndim != 2 or B.shape[0] != n or B.shape[1] == 0:
        raise ValueError("B must have shape (n, m), with m > 0")
    if Q.shape != (n, n) or R.shape != (B.shape[1], B.shape[1]):
        raise ValueError("Q and R must have shapes (n, n) and (m, m)")
    if any(jnp.iscomplexobj(x) for x in (A, B, Q, R)):
        raise ValueError("Riccati solvers require real matrices")
    dtype = jnp.result_type(A, B, Q, R, 1.0)
    return tuple(x.astype(dtype) for x in (A, B, Q, R))


def _result(A, B, Q, R, S, steps, solver, *, continuous, throw):
    K = jnp.linalg.solve(R, B.T @ S) if continuous else _lqr_gain(A, B, S, R)
    poles = jnp.linalg.eigvals(A - B @ K)
    if continuous:
        terms = (Q, A.T @ S, S @ A, -S @ B @ K)
    else:
        terms = (S, -Q, -A.T @ S @ A, A.T @ S @ B @ K)
    residual_norm = jnp.max(jnp.abs(sum(terms)))
    scale = sum(jnp.max(jnp.abs(term)) for term in terms)
    symmetric = (
        jnp.max(jnp.abs(Q - Q.T)) <= solver.atol + solver.rtol * jnp.max(jnp.abs(Q))
    ) & (jnp.max(jnp.abs(R - R.T)) <= solver.atol + solver.rtol * jnp.max(jnp.abs(R)))
    valid = (
        symmetric
        & (jnp.min(jnp.linalg.eigvalsh(_symmetrize(Q))) >= -solver.atol)
        & (jnp.min(jnp.linalg.eigvalsh(_symmetrize(R))) > 0)
    )
    stable = jnp.all(jnp.real(poles) < 0 if continuous else jnp.abs(poles) < 1)
    finite = jnp.all(jnp.isfinite(S)) & jnp.all(jnp.isfinite(K)) & jnp.isfinite(scale)
    good = finite & stable & (residual_norm <= solver.atol + solver.rtol * scale)
    exhausted = False if continuous else steps >= solver.max_steps
    status = jnp.where(
        ~valid,
        int(SolverStatus.invalid_problem),
        jnp.where(
            good,
            int(SolverStatus.successful),
            jnp.where(
                finite & exhausted,
                int(SolverStatus.max_steps_reached),
                int(SolverStatus.numerical_failure),
            ),
        ),
    )
    if throw:
        S = eqx.error_if(
            S, status != 0, "Riccati solve failed; inspect status/residual"
        )
    return LQRResult(
        K=K,
        S=S,
        poles=poles,
        residual_norm=residual_norm,
        status=status,
        num_steps=steps,
    )


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
        >>> import contrax as cx
        >>> import jax.numpy as jnp
        >>> result = cx.dare(jnp.eye(2), jnp.eye(2), jnp.eye(2), jnp.eye(2))
        >>> result.successful
        Array(True, dtype=bool)
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


# ── CARE via Hamiltonian eigendecomposition ───────────────────────────────


def _care_hamiltonian_solve(A: Array, B: Array, Q: Array, R: Array) -> Array:
    """Solve CARE from the stable invariant subspace of the Hamiltonian.

    Reference: Laub (1979), "A Schur Method for Solving Algebraic Riccati
    Equations". We follow the same stable-subspace construction but use
    `jnp.linalg.eig` rather than Schur so the path stays JAX-native.
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
        correction, (_, status) = solve_lyapunov(
            (A - B @ K).T, residual, continuous=True, solver=Lyapunov()
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
    dS, _ = solve_lyapunov((A - B @ K).T, forcing, continuous=True, solver=Lyapunov())
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

    A Hamiltonian stable-subspace candidate is refined by residual-decreasing,
    stabilizing Newton corrections. The eigenvector solve is sensitive to poor
    stable-subspace separation; an inaccurate candidate is reported as failure.
    Requires float64 mode, real matrices, symmetric positive-semidefinite Q and
    positive-definite R. Stabilizability and detectability are needed for the
    stabilizing solution. No global precision settings are changed.

    Supports `jit`, `vmap`, JVP and VJP. Derivatives solve the implicit equation,
    without differentiating through eigenvectors or refinement iterations.
    Residual and closed-loop stability validation run in JAX in every mode.

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
        >>> import contrax as cx
        >>> import jax.numpy as jnp
        >>> result = cx.care(-jnp.eye(2), jnp.eye(2), jnp.eye(2), jnp.eye(2))
        >>> result.successful
        Array(True, dtype=bool)
    """
    require_x64("care")
    solver = CARE() if solver is None else solver
    if not isinstance(solver, CARE):
        raise TypeError("care solver must be contrax.solvers.CARE")
    A, B, Q, R = _validate_matrices(A, B, Q, R)
    S, steps = _care_with_info(A, B, Q, R, solver.refinement_steps)
    return _result(A, B, Q, R, S, steps, solver, continuous=True, throw=throw)
