"""Riccati residual, status and implicit transform contracts.

Scalar reference: DARE with a=b=q=r=1 has S=(1+sqrt(5))/2.
CARE with a=b=q=r=1 has S=1+sqrt(2).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import contrax as cx
from contrax.solvers import CARE, DARE, SolverStatus


@pytest.mark.parametrize("continuous", [False, True])
def test_scalar_oracle_and_implicit_forward_reverse_derivatives(continuous):
    solve = cx.care if continuous else cx.dare
    A = B = Q = R = jnp.ones((1, 1))
    expected = 1 + jnp.sqrt(2) if continuous else (1 + jnp.sqrt(5)) / 2
    result = jax.jit(solve)(A, B, Q, R)
    assert result.successful
    np.testing.assert_allclose(result.S[0, 0], expected, atol=1e-12)

    def loss(q):
        return solve(A, B, q[None, None], R).S[0, 0]

    # DARE: S(q)=(q+sqrt(q*q+4*q))/2 at a=b=r=1.
    derivative = 1 / (2 * jnp.sqrt(2)) if continuous else (1 + 3 / jnp.sqrt(5)) / 2
    _, tangent = jax.jit(lambda q: jax.jvp(loss, (q,), (jnp.ones_like(q),)))(
        jnp.array(1.0)
    )
    reverse = jax.jit(jax.grad(loss))(jnp.array(1.0))
    np.testing.assert_allclose(tangent, reverse, atol=1e-10)
    np.testing.assert_allclose(tangent, derivative, atol=1e-10)
    batched = jax.jit(jax.vmap(lambda q: solve(A, B, q[None, None], R)))(
        jnp.array([1.0, 2.0])
    )
    assert jnp.all(batched.successful)


@pytest.mark.parametrize("solve", [cx.dare, cx.care])
def test_invalid_cost_and_unstabilizable_problem_have_compiled_status(solve):
    one = jnp.ones((1, 1))
    for operation in (solve, jax.jit(solve)):
        invalid = operation(one, one, one, -one)
        assert invalid.status == SolverStatus.invalid_problem
        assert not invalid.successful
        stable = -one if solve is cx.care else one * 0.5
        tiny_invalid = operation(stable, one, -1e-13 * one, one)
        assert tiny_invalid.status == SolverStatus.invalid_problem
        failed = operation(one, jnp.zeros_like(one), one, one)
        assert not failed.successful
    with pytest.raises(Exception, match="Riccati solve failed"):
        jax.jit(lambda q: solve(one, one, q, -one, throw=True))(
            one
        ).S.block_until_ready()


def test_dare_iteration_limit_and_wrapper_selection():
    sys = cx.dss(
        jnp.ones((1, 1)), jnp.ones((1, 1)), jnp.ones((1, 1)), jnp.zeros((1, 1)), dt=1
    )
    solver = DARE(max_steps=0)
    result = cx.lqr(sys, jnp.eye(1), jnp.eye(1), solver=solver)
    assert result.status == SolverStatus.max_steps_reached
    assert result.num_steps == 0
    assert not cx.kalman_gain(sys, jnp.eye(1), jnp.eye(1), solver=solver).successful
    assert not cx.lqi(sys, jnp.eye(2), jnp.eye(1), solver=solver).successful
    with pytest.raises(TypeError, match="DARE"):
        cx.lqr(sys, jnp.eye(1), jnp.eye(1), solver=CARE())


@pytest.mark.parametrize("solve", [cx.dare, cx.care])
def test_shapes_are_immediate_errors(solve):
    with pytest.raises(ValueError, match="Q and R"):
        solve(jnp.eye(2), jnp.ones((2, 1)), jnp.eye(3), jnp.eye(1))
