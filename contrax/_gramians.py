"""Finite and infinite horizon continuous/discrete LTI Gramians."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from contrax._lyapunov import lyapunov
from contrax._precision import require_x64
from contrax._quadratic import _compose_gramian
from contrax.core import ContLTI, DiscLTI
from contrax.solvers import Lyapunov, SolverStatus
from contrax.types import LyapunovResult


def _finite_continuous(A, forcing, horizon):
    # Reference: Van Loan (1978), "Computing Integrals Involving the Matrix
    # Exponential", IEEE TAC.
    n = A.shape[0]
    if n == 0:
        return A, forcing
    block = jnp.block([[A, forcing], [jnp.zeros_like(A), -A.T]])
    exponential = jax.scipy.linalg.expm(block * horizon)
    phi = exponential[:n, :n]
    W = exponential[:n, n:] @ phi.T
    return phi, (W + W.T) / 2


def _gramian(sys, *, observable, t, num_steps, solver, throw):
    require_x64("obsv_gramian" if observable else "ctrb_gramian")
    if not isinstance(sys, (ContLTI, DiscLTI)):
        raise TypeError("Gramians require a continuous or discrete LTI system")
    if any(jnp.iscomplexobj(matrix) for matrix in (sys.A, sys.B, sys.C, sys.D)):
        raise ValueError("Gramian routines require real LTI matrices")
    continuous = isinstance(sys, ContLTI)
    if (continuous and num_steps is not None) or (not continuous and t is not None):
        raise ValueError(
            "Use t= for continuous horizons and num_steps= for discrete horizons"
        )
    A = sys.A.T if observable else sys.A
    forcing = sys.C.T @ sys.C if observable else sys.B @ sys.B.T
    infinite = t is None and num_steps is None
    if infinite:
        solver = Lyapunov(method="doubling") if solver is None else solver
        result = lyapunov(A, forcing, continuous=continuous, solver=solver, throw=False)
        values = jnp.linalg.eigvals(A)
        stable = jnp.all(jnp.real(values) < 0 if continuous else jnp.abs(values) < 1)
        status = jnp.where(stable, result.status, int(SolverStatus.invalid_problem))
        result = eqx.tree_at(lambda r: r.status, result, status)
    else:
        if solver is not None:
            raise ValueError(
                "solver= selects an infinite-horizon Lyapunov algorithm only"
            )
        if continuous:
            horizon = jnp.asarray(t, dtype=A.dtype)
            if horizon.ndim != 0:
                raise ValueError("t must be a scalar time horizon")
            phi, W = _finite_continuous(A, forcing, horizon)
            valid = jnp.isfinite(horizon) & (horizon >= 0)
            terms = (A @ W, W @ A.T, forcing, -phi @ forcing @ phi.T)
            steps = jnp.asarray(0)
        else:
            if not isinstance(num_steps, int) or isinstance(num_steps, bool):
                raise TypeError("num_steps must be a static Python integer")
            if num_steps < 0:
                raise ValueError("num_steps must be nonnegative")
            phi_T, W = _compose_gramian(A.T, forcing, num_steps)
            phi = phi_T.T
            valid = jnp.asarray(True)
            terms = (W, -A @ W @ A.T, -forcing, phi @ forcing @ phi.T)
            steps = jnp.asarray(num_steps)
        residual = jnp.linalg.norm(sum(terms))
        scale = sum(jnp.linalg.norm(term) for term in terms)
        tolerance = 500 * jnp.finfo(A.dtype).eps * max(A.shape[0], 1)
        good = (
            jnp.isfinite(residual)
            & jnp.isfinite(scale)
            & (residual <= tolerance * (1 + scale))
        )
        status = jnp.where(
            ~valid,
            int(SolverStatus.invalid_problem),
            jnp.where(
                good, int(SolverStatus.successful), int(SolverStatus.numerical_failure)
            ),
        )
        result = LyapunovResult(W, status, steps, residual)
    if throw:
        result = eqx.error_if(
            result,
            result.status != 0,
            "Gramian computation failed; inspect status/residual",
        )
    return result


def ctrb_gramian(
    sys: ContLTI | DiscLTI,
    *,
    t: float | Array | None = None,
    num_steps: int | None = None,
    solver: Lyapunov | None = None,
    throw: bool = False,
) -> LyapunovResult:
    """Compute a continuous or discrete controllability Gramian.

    With no horizon, solves the infinite-horizon equation for a strictly stable
    system. Specify t for a continuous finite integral or num_steps for a
    discrete finite sum. Finite horizons also support marginal/unstable systems.
    Requires float64 mode; supports `jit`, `vmap`, JVP and VJP. Discrete horizon
    counts are static Python integers. No sample-time factor scales step sums.

    Args:
        sys: Continuous or discrete LTI realization.
        t: Nonnegative continuous-time integration horizon.
        num_steps: Nonnegative number of discrete Gramian terms.
        solver: Infinite-horizon Lyapunov configuration; defaults to stable doubling.
        throw: Raise on numerical failure or a nonstable infinite-horizon model.

    Returns:
        LyapunovResult: Gramian X, validated status and equation residual.
            Finite discrete num_steps reports the horizon term count.

    Examples:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import contrax as cx
        >>> sys = cx.dss([[0.8]], [[1.0]], [[1.0]], [[0.0]], dt=1)
        >>> result = ctrb_gramian(sys, num_steps=20)
        >>> Wc = result.X
    """
    return _gramian(
        sys, observable=False, t=t, num_steps=num_steps, solver=solver, throw=throw
    )


def obsv_gramian(
    sys: ContLTI | DiscLTI,
    *,
    t: float | Array | None = None,
    num_steps: int | None = None,
    solver: Lyapunov | None = None,
    throw: bool = False,
) -> LyapunovResult:
    """Compute a continuous or discrete observability Gramian.

    The dual of ctrb_gramian, with forcing C.T C and transposed state dynamics.
    With no horizon, requires strict stability. Specify t for a finite continuous
    integral or num_steps for a finite discrete sum; zero horizons are valid.
    Requires float64 mode and supports `jit`, `vmap`, JVP and VJP. The discrete
    term count is static; its weights are per-step, with no sample-time factor.

    Args:
        sys: Continuous or discrete LTI realization.
        t: Nonnegative continuous-time integration horizon.
        num_steps: Nonnegative number of discrete Gramian terms.
        solver: Infinite-horizon Lyapunov configuration; defaults to stable doubling.
        throw: Raise on numerical failure or a nonstable infinite-horizon model.

    Returns:
        LyapunovResult: Gramian X, validated status and equation residual.
            Finite discrete num_steps reports the horizon term count.

    Examples:
        >>> import jax
        >>> jax.config.update("jax_enable_x64", True)
        >>> import contrax as cx
        >>> sys = cx.ss([[-1.0]], [[1.0]], [[1.0]], [[0.0]])
        >>> result = obsv_gramian(sys, t=2.0)
        >>> Wo = result.X
    """
    return _gramian(
        sys, observable=True, t=t, num_steps=num_steps, solver=solver, throw=throw
    )
