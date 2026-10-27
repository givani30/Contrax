"""Validated design-time pole placement for real continuous/discrete LTI models."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from contrax._pole_algorithms import (
    _ctrb_matrix,
    _order_complex_poles,
    _place_ackermann,
    _place_robust,
)
from contrax._precision import require_x64
from contrax.core import ContLTI, DiscLTI
from contrax.solvers import PolePlacement, SolverStatus
from contrax.types import PlaceResult


def _assignment_error(actual, desired):
    """Bottleneck bipartite matching; preserve multiplicities and conjugate pairs.

    The gain/eigenvalue calculations remain native JAX. This combinatorial
    design-time validation uses augmenting paths on the host, avoiding sorted
    eigenvalue comparisons that can mismatch clustered conjugate pairs.
    """
    distances = np.asarray(jnp.abs(actual[:, None] - desired[None, :]))
    if not np.all(np.isfinite(distances)):
        return float("inf")
    n = len(distances)

    def feasible(radius):
        row_match, col_match = [-1] * n, [-1] * n
        for root in range(n):
            queue, parent, visited = [root], [-1] * n, [False] * n
            found = False
            for row in queue:
                for col in np.flatnonzero(distances[row] <= radius):
                    if visited[col]:
                        continue
                    visited[col], parent[col] = True, row
                    if col_match[col] == -1:
                        while col != -1:
                            row = parent[col]
                            old = row_match[row]
                            row_match[row], col_match[col] = col, row
                            col = old
                        found = True
                        break
                    queue.append(col_match[col])
                if found:
                    break
            if not found:
                return False
        return True

    radii = np.unique(distances)
    lo, hi = 0, len(radii) - 1
    while lo < hi:
        middle = (lo + hi) // 2
        if feasible(radii[middle]):
            hi = middle
        else:
            lo = middle + 1
    return float(radii[lo])


def _assess(A, B, K, desired, solver):
    closed = A - B @ K
    if not bool(jnp.all(jnp.isfinite(K)) & jnp.all(jnp.isfinite(closed))):
        return (
            jnp.full(A.shape[0], jnp.nan, dtype=jnp.complex128),
            float("inf"),
            float("inf"),
            False,
        )
    actual, vectors = jnp.linalg.eig(closed)
    error = _assignment_error(actual, desired)
    condition = float(jnp.linalg.cond(vectors))
    tolerance = solver.assignment_atol + solver.assignment_rtol * float(
        jnp.max(jnp.abs(desired))
    )
    valid = (
        bool(jnp.all(jnp.isfinite(actual)))
        and error <= tolerance
        and condition <= solver.condition_limit
    )
    return actual, error, condition, valid


def place(
    sys: DiscLTI | ContLTI,
    poles: ArrayLike,
    *,
    solver: PolePlacement | None = None,
    throw: bool = False,
) -> PlaceResult:
    """Assign real/conjugate-paired poles and validate the achieved feedback.

    Uses native JAX KNV0 (real poles) or YT (real or complex pairs) linear
    algebra with design-time rank decisions and convergence control. Input
    subspaces use SVD so dependent and redundant input columns are supported.
    Small SISO failures may try bounded Ackermann fallback; every candidate
    must meet assignment and conditioning tolerances.

    Requires float64 mode; calculations promote real model matrices to float64.
    Call outside jit, vmap and differentiation. Structural decisions and
    host-side matching do not have a traced or differentiable contract. Returned
    diagnostics use the same JAX status codes as other numerical solvers.

    Args:
        sys: Continuous or discrete real LTI model, with n>0 and m>0.
        poles: Finite requested poles, shape (n,), real or conjugate-paired.
        solver: PolePlacement method, stopping/acceptance tolerances and limits.
            Multiplicity beyond rank(B) requires bounded SISO fallback; robust
            KNV0/YT updates do not support that configuration.
        throw: Raise on failed assignment, convergence or conditioning checks.

    Returns:
        PlaceResult: K, achieved poles, status, matching error, conditioning,
        iterations and selected method. Unpacks as K, poles. A failed result
        may contain a finite candidate for inspection; check successful.

    Examples:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import contrax as cx
        >>> sys = cx.ss([[0., 1.], [0., 0.]], [[0.], [1.]], [[1., 0.]], [[0.]])
        >>> result = cx.place(sys, [-1., -2.])
        >>> bool(result.successful)
        True
    """
    require_x64("place")
    solver = PolePlacement() if solver is None else solver
    if not isinstance(solver, PolePlacement):
        raise TypeError("place solver must be contrax.solvers.PolePlacement")
    if not isinstance(sys, (ContLTI, DiscLTI)):
        raise TypeError("place requires a continuous or discrete LTI model")
    if jnp.iscomplexobj(sys.A) or jnp.iscomplexobj(sys.B):
        raise ValueError("place requires real model matrices")
    try:
        np.asarray(sys.A)
        np.asarray(sys.B)
        requested = np.asarray(poles, dtype=complex)
    except jax.errors.TracerArrayConversionError as error:
        raise TypeError(
            "place is design-time; call it outside JAX transforms"
        ) from error
    A, B = jnp.asarray(sys.A, dtype=jnp.float64), jnp.asarray(sys.B, dtype=jnp.float64)
    if A.ndim != 2 or A.shape[0] == 0 or A.shape[1] != A.shape[0]:
        raise ValueError("A must be a nonempty square matrix")
    n = A.shape[0]
    if B.ndim != 2 or B.shape[0] != n or B.shape[1] == 0:
        raise ValueError("B must have shape (n, m), m>0")
    if requested.ndim != 1 or requested.shape[0] != n:
        raise ValueError("place requires exactly n requested poles")
    desired = _order_complex_poles(requested)
    complex_poles = bool(jnp.any(jnp.imag(desired) != 0))
    method = (
        ("YT" if complex_poles else "KNV0")
        if solver.method == "auto"
        else solver.method
    )
    if method == "KNV0" and complex_poles:
        raise ValueError("KNV0 does not support complex poles; select YT or auto")
    cc = float("inf")

    def result(
        K, actual, error, ec, status, steps=0, improvement=float("inf"), fallback=False
    ):
        answer = PlaceResult(
            K=K,
            poles=actual,
            status=jnp.asarray(status, dtype=jnp.int32),
            assignment_error=jnp.asarray(error),
            controllability_condition=jnp.asarray(cc),
            eigenvector_condition=jnp.asarray(ec),
            num_steps=jnp.asarray(steps, dtype=jnp.int32),
            relative_improvement=jnp.asarray(improvement),
            used_fallback=jnp.asarray(fallback),
            method="ackermann" if fallback else method,
        )
        if throw and status != SolverStatus.successful:
            raise RuntimeError(
                "Pole assignment failed; inspect assignment_error and conditioning"
            )
        return answer

    def failed(status):
        return result(
            jnp.full((B.shape[1], n), jnp.nan),
            jnp.full(n, jnp.nan, dtype=jnp.complex128),
            float("inf"),
            float("inf"),
            status,
        )

    if not bool(jnp.all(jnp.isfinite(A)) & jnp.all(jnp.isfinite(B))):
        return failed(SolverStatus.invalid_problem)
    controllability = _ctrb_matrix(A, B)
    if not bool(jnp.all(jnp.isfinite(controllability))):
        return failed(SolverStatus.numerical_failure)
    singular = jnp.linalg.svd(controllability, compute_uv=False)
    if not bool(jnp.all(jnp.isfinite(singular))):
        return failed(SolverStatus.numerical_failure)
    cc = float(jnp.where(singular[-1] > 0, singular[0] / singular[-1], jnp.inf))
    rank = int(
        jnp.sum(
            singular > singular[0] * max(controllability.shape) * jnp.finfo(A.dtype).eps
        )
    )
    if rank < n:
        return failed(SolverStatus.invalid_problem)
    if cc > solver.condition_limit:
        return failed(SolverStatus.numerical_failure)
    rankB = int(jnp.linalg.matrix_rank(B))
    repeated = (
        max(int(np.sum(np.asarray(desired) == pole)) for pole in np.asarray(desired))
        > rankB
    )
    allow_fallback = B.shape[1] == 1 and n <= solver.fallback_limit
    if repeated and not allow_fallback:
        raise ValueError("Requested multiplicity exceeds rank(B) for KNV0/YT")

    fallback = repeated
    if fallback:
        K, steps, improvement = _place_ackermann(A, B, desired), 0, 0.0
    else:
        K, steps, improvement = _place_robust(
            A, B, desired, method=method, rtol=solver.rtol, maxiter=solver.max_steps
        )
    actual, error, ec, good = _assess(A, B, K, desired, solver)
    if not good and allow_fallback and not fallback:
        alternate = _place_ackermann(A, B, desired)
        ap, ae, ac, accepted = _assess(A, B, alternate, desired, solver)
        if accepted or (
            not bool(jnp.all(jnp.isfinite(K)))
            and bool(jnp.all(jnp.isfinite(alternate)))
        ):
            K, actual, error, ec, good = alternate, ap, ae, ac, accepted
            steps, improvement, fallback = 0, 0.0, True
    status = (
        (
            SolverStatus.successful
            if improvement < solver.rtol
            else SolverStatus.max_steps_reached
        )
        if good
        else SolverStatus.numerical_failure
    )
    return result(K, actual, error, ec, status, steps, improvement, fallback)
