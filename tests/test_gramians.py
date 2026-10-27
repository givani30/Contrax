"""Continuous and discrete Gramian identities and horizon conventions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import contrax as cx


@pytest.mark.parametrize("observable", [False, True])
@pytest.mark.parametrize("steps", [0, 1, 3, 32])
def test_finite_discrete_matches_explicit_sum_for_nonnormal_marginal_system(
    observable, steps
):
    A = jnp.array([[1.0, 0.2], [0.0, 1.0]])
    B, C = jnp.array([[0.1], [1.0]]), jnp.array([[1.0, 0.3]])
    sys = cx.dss(A, B, C, jnp.zeros((1, 1)), 0.1)
    solve = cx.obsv_gramian if observable else cx.ctrb_gramian
    result = jax.jit(lambda s: solve(s, num_steps=steps))(sys)
    expected = jnp.zeros_like(A)
    power = jnp.eye(2)
    for _ in range(steps):
        expected = expected + (
            power.T @ C.T @ C @ power if observable else power @ B @ B.T @ power.T
        )
        power = A @ power
    assert result.successful
    np.testing.assert_allclose(result.X, expected, atol=1e-10)


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("observable", [False, True])
def test_infinite_gramian_uses_stable_lyapunov_equation_and_has_implicit_derivatives(
    continuous, observable
):
    def solve(a):
        A = (
            jnp.array([[-jnp.exp(a)]])
            if continuous
            else jnp.array([[0.5 * jnp.exp(a)]])
        )
        sys = (
            cx.ss(A, [[2.0]], [[3.0]], [[0.0]])
            if continuous
            else cx.dss(A, [[2.0]], [[3.0]], [[0.0]], 0.1)
        )
        return (cx.obsv_gramian if observable else cx.ctrb_gramian)(sys)

    result = jax.jit(solve)(jnp.array(0.0))
    forcing = 9.0 if observable else 4.0
    expected = forcing / 2 if continuous else forcing / (1 - 0.25)
    assert result.successful
    np.testing.assert_allclose(result.X[0, 0], expected, atol=1e-10)

    def value(a):
        return solve(a).X[0, 0]

    _, forward = jax.jvp(value, (jnp.array(0.0),), (jnp.array(1.0),))
    np.testing.assert_allclose(forward, jax.grad(value)(jnp.array(0.0)), atol=1e-10)
    batched = jax.jit(jax.vmap(solve))(jnp.array([0.0, -0.2]))
    assert jnp.all(batched.successful)


def test_horizon_arguments_and_nonstable_infinite_status():
    c = cx.ss([[-1.0]], [[1.0]], [[1.0]], [[0.0]])
    d = cx.dss([[1.0]], [[1.0]], [[1.0]], [[0.0]], 1)
    assert cx.ctrb_gramian(c, t=0).successful
    assert cx.ctrb_gramian(c, t=0).X[0, 0] == 0
    assert cx.ctrb_gramian(c, t=-1).status == cx.solvers.SolverStatus.invalid_problem
    assert not cx.ctrb_gramian(d).successful
    assert not jax.jit(cx.obsv_gramian)(d).successful
    for sys, kwargs in ((c, {"num_steps": 2}), (d, {"t": 1.0})):
        with pytest.raises(ValueError, match="Use t="):
            cx.ctrb_gramian(sys, **kwargs)
    with pytest.raises(Exception, match="Gramian computation failed"):
        cx.ctrb_gramian(d, throw=True)
