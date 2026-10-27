"""Rank-dependent orthogonal reduction of the Rosenbrock system pencil."""

import math

import jax.numpy as jnp
from jax import Array

from contrax._precision import require_x64
from contrax.core import ContLTI, DiscLTI


def _svd(matrix, threshold):
    U, values, Vh = jnp.linalg.svd(matrix, full_matrices=True)
    rank = int(jnp.sum(values > threshold))
    return U, values, Vh, rank


def _uncontrollable_pencil(A, B, threshold):
    """Remove the right singular part of [A-lambda I, B] orthogonally."""
    n = A.shape[0]
    reachable = jnp.empty((n, 0), dtype=A.dtype)
    complement = jnp.eye(n, dtype=A.dtype)
    for _ in range(n + 1):
        columns = jnp.concatenate((reachable, B, A @ reachable), axis=1)
        U, _, _, rank = _svd(columns, threshold)
        complement = U[:, rank:]
        if rank == reachable.shape[1] or rank == n:
            break
        reachable = U[:, :rank]
    return complement.T @ A @ complement


def zeros(
    sys: ContLTI | DiscLTI, *, rtol: float | None = None, atol: float = 0.0
) -> Array:
    """Return finite invariant zeros of a real continuous or discrete LTI system.

    Zeros are rank drops of the Rosenbrock pencil relative to its normal rank.
    Supports square and rectangular systems, arbitrary feedthrough rank and
    nonminimal realizations. Invariant zeros coincide with transmission zeros
    for minimal realizations; nonminimal models may include decoupling zeros
    absent from a minimal transfer realization. Infinite zeros are omitted.

    Orthogonal staircase reduction removes output constraints and infinite
    structure. A controllability staircase removes remaining right singular
    structure; the finite reduced pencil is solved with JAX eigensolvers.
    This uses rank-dependent shapes and host decisions: it is design-time
    analysis, with no `jit`, `vmap` or differentiation contract. Requires x64.
    Rank decisions are sensitive to scaling; set tolerances for nearly rank-
    deficient realizations and compare transfer behavior under rescaling.

    Args:
        sys: Real continuous or discrete LTI realization.
        rtol: Relative numerical rank tolerance. Defaults to ten times machine
            epsilon times the larger Rosenbrock pencil dimension.
        atol: Absolute rank threshold. Nonnegative.

    Returns:
        Array: Finite invariant zeros including multiplicities. Complex shape `(k,)`.

    Raises:
        ValueError: For complex/nonfinite matrices or invalid rank tolerances.

    Examples:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import contrax as cx
        >>> import jax.numpy as jnp
        >>> sys = cx.ss(jnp.array([[-2.]]), jnp.ones((1, 1)),
        ...             jnp.array([[3.]]), jnp.array([[2.]]))
        >>> cx.zeros(sys)
        Array([-3.5+0.j], dtype=complex128)
    """
    require_x64("zeros")
    A, B, C, D = sys.A, sys.B, sys.C, sys.D
    if any(
        jnp.iscomplexobj(x) or not bool(jnp.all(jnp.isfinite(x))) for x in (A, B, C, D)
    ):
        raise ValueError("zeros requires finite real matrices")
    n, m, p = A.shape[0], B.shape[1], C.shape[0]
    rtol = (
        float(jnp.finfo(A.dtype).eps) * max(n + m, n + p, 1) * 10
        if rtol is None
        else rtol
    )
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0:
        raise ValueError("rank tolerances must be finite and nonnegative")
    compound = jnp.block([[A, B], [C, D]])
    scale = float(jnp.linalg.norm(compound))
    threshold = atol + rtol * scale

    # Reference: Emami-Naeini & Van Dooren (1982), "Computation of Zeros of
    # Linear Multivariable Systems", Automatica 18, 415-430; SLICOT AB08ND.
    # SVD supplies rank-revealing orthogonal coordinates. Eliminate the
    # nonsingular D block, then restrict x to ker(C_remaining). The discarded
    # state rows become the next algebraic output constraints. Each such step
    # removes at least one state (infinite structure), preserving finite rank
    # drops. Once constraints vanish, remove the right singular staircase.
    for _ in range(n + 1):
        if A.shape[0] == 0:
            return jnp.empty((0,), dtype=jnp.complex128)
        U, values, Vh, rank = _svd(D, threshold)
        B = B @ Vh.T
        C = U.T @ C
        if rank:
            A = A - (B[:, :rank] / values[:rank]) @ C[:rank]
        B = B[:, rank:]
        C = C[rank:]
        # Use an absolute floor from the original pencil to distinguish exact
        # structural zeros from roundoff created by orthogonal transformations.
        threshold = max(threshold, atol + rtol * float(jnp.linalg.norm(A)))
        _, _, Vh, rank_c = _svd(C, threshold)
        if rank_c == 0:
            finite = _uncontrollable_pencil(A, B, threshold)
            return jnp.linalg.eigvals(finite).astype(jnp.complex128)
        constrained, free = Vh[:rank_c].T, Vh[rank_c:].T
        A_free, B_free = free.T @ A @ free, free.T @ B
        C, D = constrained.T @ A @ free, constrained.T @ B
        A, B = A_free, B_free
    raise RuntimeError("invariant-zero staircase did not terminate")
