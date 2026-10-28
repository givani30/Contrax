"""Equation invariants and transform tests for shared Lyapunov solvers.

Scalar oracles follow x=q/(1-a*a) and x=-q/(2*a). Matrix checks
compare the independent dense operator with squared Smith iteration.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import contrax as cx
from contrax.solvers import Lyapunov, SolverStatus


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("symmetric", [False, True])
def test_nonnormal_equation_matches_dense_and_transforms(continuous, symmetric):
    A = jnp.array([[-0.4, 4.0], [0.0, -0.2]])
    if not continuous:
        A = A + jnp.eye(2)
    Q = jnp.array([[2.0, 0.3], [0.3, 1.0]])
    if not symmetric:
        Q = Q.at[0, 1].set(-0.7)
    solve = cx.lyap if continuous else cx.dlyap
    direct = solve(A, Q, solver=Lyapunov(method="dense"))
    result = jax.jit(lambda a, q: solve(a, q, solver=Lyapunov(method="doubling")))(A, Q)
    assert result.successful
    assert result.num_steps > 0
    np.testing.assert_allclose(result.X, direct.X, rtol=1e-9, atol=1e-9)
    if symmetric:
        np.testing.assert_allclose(result.X, result.X.T, rtol=1e-12, atol=1e-12)

    def loss(a, q):
        return jnp.sum(solve(a, q).X ** 2)

    direction = jnp.array([[0.1, 0.2], [0.0, -0.1]])
    _, tangent = jax.jvp(loss, (A, Q), (direction, jnp.zeros_like(Q)))
    reverse = jnp.sum(jax.grad(loss)(A, Q) * direction)
    eps = 1e-5
    finite_difference = (
        loss(A + eps * direction, Q) - loss(A - eps * direction, Q)
    ) / (2 * eps)
    np.testing.assert_allclose(tangent, reverse, rtol=1e-9)
    np.testing.assert_allclose(tangent, finite_difference, rtol=1e-7)
    batched = jax.jit(jax.vmap(lambda q: solve(A, q).X))(jnp.stack([Q, 2 * Q]))
    np.testing.assert_allclose(batched[1], 2 * batched[0], rtol=1e-10)


@pytest.mark.parametrize(
    "solve,a,expected", [(cx.lyap, 2.0, -0.25), (cx.dlyap, 2.0, -1 / 3)]
)
def test_general_nonstable_equations_remain_supported(solve, a, expected):
    for method in ("auto", "dense"):
        result = solve(
            jnp.array([[a]]), jnp.ones((1, 1)), solver=Lyapunov(method=method)
        )
        assert result.successful
        np.testing.assert_allclose(result.X[0, 0], expected, atol=1e-12)


def test_large_failed_iteration_never_traces_dense_fallback():
    A, Q = 0.9 * jnp.eye(12), jnp.eye(12)
    solver = Lyapunov(max_steps=0, dense_limit=8)
    result = jax.jit(lambda a: cx.dlyap(a, Q, solver=solver))(A)
    assert not result.successful
    assert result.status == SolverStatus.max_steps_reached
    # A dense fallback would introduce an n**2 by n**2 operator even in a
    # non-taken lax.cond branch. Inspect the entire traced solve, including aux.
    program = str(jax.make_jaxpr(lambda a: cx.dlyap(a, Q, solver=solver))(A))
    assert "144,144" not in program


def test_singular_equation_reports_failure_eager_and_compiled():
    A, Q = jnp.ones((1, 1)), jnp.ones((1, 1))
    for solve in (cx.dlyap, jax.jit(cx.dlyap)):
        result = solve(A, Q)
        assert not result.successful
        assert result.status == SolverStatus.numerical_failure
    with pytest.raises(Exception, match="Lyapunov solve failed"):
        jax.jit(lambda a: cx.dlyap(a, Q, throw=True))(A).X.block_until_ready()


@pytest.mark.parametrize("solve", [cx.lyap, cx.dlyap])
def test_empty_shape_and_configuration_contract(solve):
    result = solve(jnp.empty((0, 0)), jnp.empty((0, 0)))
    assert result.X.shape == (0, 0)
    assert result.successful
    with pytest.raises(ValueError, match="square"):
        solve(jnp.ones((2, 3)), jnp.eye(2))
    with pytest.raises(TypeError, match="solver"):
        solve(jnp.eye(2), jnp.eye(2), solver="dense")


def test_float64_mode_is_required_without_global_mutation():
    with jax.enable_x64(False):
        with pytest.raises(RuntimeError, match="requires jax_enable_x64"):
            cx.dlyap(jnp.eye(2), jnp.eye(2))


def test_riccati_adjoint_large_nonnormal_equation():
    from contrax._lyapunov import _doubling, _residual
    from contrax._riccati import _solve_adjoint_discrete_lyapunov

    n = 32
    transition = jnp.diag(jnp.linspace(0.965, 0.997, n)) + 0.45 * jnp.diag(
        jnp.ones(n - 1), 1
    )
    B = jnp.zeros((n, 2)).at[-1, 0].set(1).at[n // 2, 1].set(1)
    design = cx.dare(transition, B, 0.2 * jnp.eye(n), jnp.eye(2))
    A = design.closed_loop.T
    raw, _ = _doubling(A, jnp.eye(n), False, Lyapunov())
    result = jax.jit(cx.dlyap)(A, jnp.eye(n))
    assert result.successful
    assert result.num_steps <= Lyapunov().max_steps
    assert result.residual_norm < _residual(A, jnp.eye(n), raw, False)[0] * 1e-3
    X = jax.jit(_solve_adjoint_discrete_lyapunov)(A, jnp.eye(n))
    # Symmetrization and separate compiled solves round differently in this
    # strongly nonnormal realization; assess the matrix-scale error.
    assert jnp.linalg.norm(X - result.X) <= 1e-8 * jnp.linalg.norm(result.X)
    limited = cx.dlyap(A, jnp.eye(n), solver=Lyapunov(max_steps=16))
    assert limited.num_steps <= 16
    assert not limited.successful
