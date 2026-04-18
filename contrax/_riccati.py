"""contrax._riccati — DARE and CARE solvers with custom VJPs."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax import core as jax_core

from contrax._precision import require_x64
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
    A: Array, B: Array, Q: Array, R: Array, max_iter: int, tol: float
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
        return (iteration < max_iter) & (delta > tol)

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
        delta = jnp.max(jnp.abs(H_next - H_k))
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
    """Solve X - A_cl X A_cl.T = G for dense small systems."""
    n = A_cl.shape[0]
    basis = jnp.eye(n * n, dtype=A_cl.dtype).reshape(n * n, n, n)

    def op(X):
        return X - A_cl @ X @ A_cl.T

    matrix = jax.vmap(lambda X: op(X).reshape(-1))(basis).T
    X = jnp.linalg.solve(matrix, G.reshape(-1)).reshape(n, n)
    return _symmetrize(X)


def _solve_adjoint_continuous_lyapunov(A_cl: Array, G: Array) -> Array:
    """Solve A_cl X + X A_cl.T = G for dense small systems.

    This builds the dense Kronecker-form linear system explicitly, so the
    forward cost scales like O(n^6) in the naive dense solve. That is fine for
    the current small-system adjoint use case, but it is not intended as a
    large-state hot path.
    """
    n = A_cl.shape[0]
    basis = jnp.eye(n * n, dtype=A_cl.dtype).reshape(n * n, n, n)

    def op(X):
        return A_cl @ X + X @ A_cl.T

    matrix = jax.vmap(lambda X: op(X).reshape(-1))(basis).T
    X = jnp.linalg.solve(matrix, G.reshape(-1)).reshape(n, n)
    return _symmetrize(X)


# ── DARE custom VJP ────────────────────────────────────────────────────────
# Reference: Implicit Function Theorem approach from DiLQR (ICML 2025) and
# trajax (Google, github.com/google/trajax).


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _dare_structured_doubling_solve(
    A: Array, B: Array, Q: Array, R: Array, max_iter: int, tol: float
) -> Array:
    S, _ = _dare_structured_doubling_until_converged(A, B, Q, R, max_iter, tol)
    return S


def _dare_structured_doubling_solve_fwd(
    A: Array, B: Array, Q: Array, R: Array, max_iter: int, tol: float
):
    S = _dare_structured_doubling_solve(A, B, Q, R, max_iter, tol)
    return S, (S, A, B, Q, R)


def _dare_structured_doubling_solve_bwd(max_iter: int, tol: float, res, g: Array):
    del max_iter, tol
    S, A, B, Q, R = res
    K = _lqr_gain(A, B, S, R)
    A_cl = A - B @ K
    adjoint = _solve_adjoint_discrete_lyapunov(A_cl, g)

    def residual_params(A, B, Q, R):
        return _dare_residual(S, A, B, Q, R)

    _, pullback = jax.vjp(residual_params, A, B, Q, R)
    dA, dB, dQ, dR = pullback(adjoint)
    return (-dA, -dB, -dQ, -dR)


_dare_structured_doubling_solve.defvjp(
    _dare_structured_doubling_solve_fwd,
    _dare_structured_doubling_solve_bwd,
)


def dare(
    A: Array,
    B: Array,
    Q: Array,
    R: Array,
    max_iter: int = 64,
    tol: float = 1e-10,
) -> LQRResult:
    """Solve the discrete algebraic Riccati equation.

    Computes the stabilizing Riccati solution and corresponding infinite-horizon
    discrete LQR gain for the matrix tuple `(A, B, Q, R)`.

    This is currently the strongest Riccati solver path in Contrax. The
    forward solve uses structured doubling with tolerance-based stopping, and
    gradients are provided through a custom VJP built from implicit
    differentiation of the Riccati residual.

    Args:
        A: Discrete-time state transition matrix. Shape: `(n, n)`.
        B: Discrete-time input matrix. Shape: `(n, m)`.
        Q: State cost matrix. Shape: `(n, n)`.
        R: Input cost matrix. Shape: `(m, m)`.
        max_iter: Maximum structured-doubling iterations.
        tol: Forward stopping tolerance on Riccati iterate changes.

    Returns:
        [LQRResult][contrax.types.LQRResult]: A bundle containing the optimal
            feedback gain `K`, Riccati solution `S`, and closed-loop poles
            `poles`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> A = jnp.array([[1.0, 0.05], [0.0, 1.0]])
        >>> B = jnp.array([[0.0], [0.05]])
        >>> result = cx.dare(A, B, jnp.eye(2), jnp.array([[1.0]]))
    """
    require_x64("dare")
    S = _dare_structured_doubling_solve(A, B, Q, R, max_iter, tol)
    K = _lqr_gain(A, B, S, _symmetrize(R))
    A_cl = A - B @ K
    poles = jnp.linalg.eigvals(A_cl)
    residual_norm = jnp.max(jnp.abs(_dare_residual(S, A, B, Q, R)))
    return LQRResult(K=K, S=S, poles=poles, residual_norm=residual_norm)


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


# ── CARE custom VJP ────────────────────────────────────────────────────────
# Reference: Implicit Function Theorem approach from DiLQR (ICML 2025) and
# trajax (Google, github.com/google/trajax).


@jax.custom_vjp
def _care_solve(A: Array, B: Array, Q: Array, R: Array) -> Array:
    return _care_hamiltonian_solve(A, B, Q, R)


def _care_solve_fwd(A: Array, B: Array, Q: Array, R: Array):
    S = _care_solve(A, B, Q, R)
    return S, (S, A, B, Q, R)


def _care_solve_bwd(res, g: Array):
    S, A, B, Q, R = res
    K = jnp.linalg.solve(R, B.T @ S)
    A_cl = A - B @ K
    adjoint = _solve_adjoint_continuous_lyapunov(A_cl, g)

    def residual_params(A, B, Q, R):
        return _care_residual(S, A, B, Q, R)

    _, pullback = jax.vjp(residual_params, A, B, Q, R)
    dA, dB, dQ, dR = pullback(adjoint)
    return (-dA, -dB, -dQ, -dR)


_care_solve.defvjp(_care_solve_fwd, _care_solve_bwd)


def _is_tracing(*arrays: Array) -> bool:
    return any(isinstance(arr, jax_core.Tracer) for arr in arrays)


def _validate_care_solution(
    A: Array,
    B: Array,
    Q: Array,
    R: Array,
    S: Array,
    poles: Array,
) -> None:
    """Run eager-only CARE diagnostics on concrete arrays."""
    R_inv_BT = jnp.linalg.solve(_symmetrize(R), B.T)
    hamiltonian = jnp.block(
        [
            [A, -(B @ R_inv_BT)],
            [-_symmetrize(Q), -A.T],
        ]
    )
    eigenvalues = np.linalg.eigvals(np.asarray(hamiltonian))
    stable_count = int(np.sum(np.real(eigenvalues) < 0.0))
    n = A.shape[0]
    if stable_count != n:
        raise ValueError(
            "care() could not isolate a stabilizing Hamiltonian subspace: "
            f"expected {n} eigenvalues with negative real part, got "
            f"{stable_count}. This usually indicates an ill-conditioned or "
            "non-stabilizing CARE instance."
        )

    residual_norm = float(jnp.max(jnp.abs(_care_residual(S, A, B, Q, R))))
    if residual_norm > 1e-7:
        raise ValueError(
            "care() produced a Riccati matrix with an unexpectedly large "
            f"residual ({residual_norm:.3e})."
        )

    if not np.all(np.isfinite(np.asarray(S))):
        raise ValueError("care() produced a non-finite Riccati matrix.")

    if not np.all(np.real(np.asarray(poles)) < 0.0):
        raise ValueError("care() did not produce a stabilizing closed-loop system.")


def care(A: Array, B: Array, Q: Array, R: Array) -> LQRResult:
    """Solve the continuous algebraic Riccati equation.

    Computes the stabilizing Riccati solution and corresponding continuous-time
    LQR gain for the matrix tuple `(A, B, Q, R)`.

    The forward solve uses a Hamiltonian stable-subspace construction with
    `jnp.linalg.eig`, and gradients are provided through an
    implicit-differentiation custom VJP. This is now a real baseline
    continuous-time solver, but it is not yet as hardened as `dare()`.

    Args:
        A: Continuous-time state matrix. Shape: `(n, n)`.
        B: Continuous-time input matrix. Shape: `(n, m)`.
        Q: State cost matrix. Shape: `(n, n)`.
        R: Input cost matrix. Shape: `(m, m)`.

    Returns:
        [LQRResult][contrax.types.LQRResult]: A bundle containing the optimal
            feedback gain `K`, Riccati solution `S`, and closed-loop poles
            `poles`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        >>> B = jnp.array([[0.0], [1.0]])
        >>> result = cx.care(A, B, jnp.eye(2), jnp.array([[1.0]]))
    """
    require_x64("care")
    S = _care_solve(A, B, Q, R)
    K = jnp.linalg.solve(_symmetrize(R), B.T @ S)
    A_cl = A - B @ K
    poles = jnp.linalg.eigvals(A_cl)
    residual_norm = jnp.max(jnp.abs(_care_residual(S, A, B, Q, R)))
    if not _is_tracing(A, B, Q, R, S, poles):
        _validate_care_solution(A, B, Q, R, S, poles)
    return LQRResult(K=K, S=S, poles=poles, residual_norm=residual_norm)
