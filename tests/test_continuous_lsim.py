"""Exact sampled continuous responses with explicit input-hold conventions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import contrax as cx


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("method", ["zoh", "foh"])
def test_marginal_integrator_nonuniform_grid_and_feedthrough(dtype, method):
    one, zero = jnp.ones((1, 1), dtype=dtype), jnp.zeros((1, 1), dtype=dtype)
    sys = cx.ss(zero, one, one, 2 * one)
    ts = jnp.array([0.0, 0.2, 0.7, 1.0], dtype=dtype)
    us = ts[:, None]
    result = jax.jit(lambda u: cx.lsim(sys, u, ts=ts, input_interpolation=method))(us)
    expected = (
        ts**2 / 2
        if method == "foh"
        else jnp.concatenate(
            [jnp.zeros(1, dtype=dtype), jnp.cumsum(jnp.diff(ts) * ts[:-1])]
        )
    )
    assert result.successful
    assert result.xs.dtype == dtype
    assert result.num_steps == ts.shape[0] - 1
    np.testing.assert_allclose(result.xs[:, 0], expected, atol=1e-6)
    np.testing.assert_allclose(result.ys[:, 0], expected + 2 * ts, atol=1e-6)

    # x'=u gives an independent oracle for every interval derivative, including
    # when the primal reuses a uniform-grid transition. Inputs stay fixed.
    def terminal_output(intervals):
        grid = jnp.concatenate([jnp.zeros(1, dtype=dtype), jnp.cumsum(intervals)])
        return cx.lsim(sys, us, ts=grid, input_interpolation=method).ys[-1, 0]

    intervals = jnp.full(3, 0.2, dtype=dtype)
    direction = jnp.array([-0.7, 0.2, 0.4], dtype=dtype)
    weights = (us[:-1, 0] + us[1:, 0]) / 2 if method == "foh" else us[:-1, 0]
    _, tangent = jax.jvp(terminal_output, (intervals,), (direction,))
    np.testing.assert_allclose(tangent, weights @ direction, atol=2e-6)
    np.testing.assert_allclose(jax.grad(terminal_output)(intervals), weights, atol=2e-6)


def test_foh_stable_scalar_analytic_response_and_derivatives():
    ts = jnp.linspace(0, 2, 9)

    def value(a):
        sys = cx.ss(-a[None, None], jnp.eye(1), jnp.eye(1), jnp.zeros((1, 1)))
        return cx.lsim(sys, ts[:, None], ts=ts, input_interpolation="foh").xs[-1, 0]

    def reference(a):
        return ts[-1] / a - (1 - jnp.exp(-a * ts[-1])) / a**2

    a = jnp.array(1.2)
    np.testing.assert_allclose(jax.jit(value)(a), reference(a), atol=1e-12)
    np.testing.assert_allclose(jax.grad(value)(a), jax.grad(reference)(a), atol=1e-10)
    _, forward = jax.jvp(value, (a,), (jnp.ones_like(a),))
    np.testing.assert_allclose(forward, jax.grad(reference)(a), atol=1e-10)
    batched = jax.jit(jax.vmap(value))(jnp.array([0.5, 1.2, 2.0]))
    np.testing.assert_allclose(
        batched, jax.vmap(reference)(jnp.array([0.5, 1.2, 2.0])), atol=1e-12
    )


def test_continuous_zoh_matches_discrete_indexing_and_initial_time():
    sys = cx.ss([[-0.7, 0.2], [0.0, -0.4]], [[1.0], [0.3]], [[1.0, 0.5]], [[0.2]])
    ts, us, x0 = (
        jnp.arange(5) * 0.2 + 3,
        jnp.arange(5)[:, None] * 0.1,
        jnp.array([0.4, -0.2]),
    )
    actual = cx.lsim(sys, us, x0, ts=ts)
    discrete = cx.lsim(cx.c2d(sys, 0.2), us[:-1], x0)
    np.testing.assert_allclose(actual.xs, discrete.xs, atol=1e-12)
    np.testing.assert_allclose(actual.ys[:-1], discrete.ys, atol=1e-12)
    np.testing.assert_allclose(actual.ts, ts)


def test_shortest_empty_grid_and_compiled_invalid_grid_status():
    sys = cx.ss([[-1.0]], [[1.0]], [[1.0]], [[2.0]])
    for N in (0, 1):
        result = cx.lsim(
            sys, jnp.ones((N, 1)), jnp.ones(1), ts=jnp.arange(N, dtype=float)
        )
        assert result.xs.shape == result.ys.shape == (N, 1)
        assert result.successful
    result = jax.jit(lambda t: cx.lsim(sys, jnp.ones((2, 1)), ts=t))(
        jnp.array([1.0, 0.0])
    )
    assert result.status == cx.solvers.SolverStatus.invalid_problem
    with pytest.raises(ValueError, match="explicit ts"):
        cx.lsim(sys, jnp.ones((2, 1)))
    with pytest.raises(ValueError, match="real"):
        cx.lsim(sys, jnp.ones((2, 1)), ts=jnp.array([0j, 1j]))
    for m, hold in ((0, "foh"), (1, "zoh")):
        static = cx.ss(
            jnp.empty((0, 0)), jnp.empty((0, m)), jnp.empty((1, 0)), jnp.ones((1, m))
        )
        response = cx.lsim(
            static, jnp.ones((2, m)), ts=jnp.array([0.0, 0.5]), input_interpolation=hold
        )
        assert response.successful and response.xs.shape == (2, 0)
        np.testing.assert_allclose(response.ys, m * jnp.ones((2, 1)))
