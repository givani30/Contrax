"""Explicit caching and whitened conditioning against covariance filtering."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import contrax as cx


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_information_filter_values_gradients_masks_and_conditioning(dtype):
    A = jnp.array([[0.9, 0.2], [0.0, 0.8]], dtype=dtype)
    B = jnp.array([[0.1], [0.2]], dtype=dtype)
    C = jnp.sin(jnp.arange(16, dtype=dtype).reshape(8, 2) + 1)
    model = cx.dss(A, B, C, jnp.ones((8, 1), dtype=dtype) * 0.2, 1)
    ys = jnp.sin(jnp.arange(32, dtype=dtype).reshape(4, 8) * 0.1).at[1].set(jnp.nan)
    us = jnp.arange(4, dtype=dtype)[:, None] * 0.1
    flags = jnp.array([True, False, True, True])
    noise = jnp.linspace(0.1, 0.3, 8, dtype=dtype)
    solver = cx.solvers.InformationKalman()

    def solve(theta, reference=False):
        Q, R = jnp.exp(theta[0]) * jnp.eye(2, dtype=dtype), jnp.exp(theta[1]) * noise
        return cx.kalman(
            model,
            ys,
            Q=Q,
            R=jnp.diag(R) if reference else R,
            us=us,
            has_measurement=flags,
            solver=None if reference else solver,
        )

    theta = jnp.array([-3.0, -1.0], dtype=dtype)
    actual, expected = jax.jit(solve)(theta), solve(theta, True)
    assert actual.successful and actual.xs.dtype == dtype
    np.testing.assert_allclose(actual.xs, expected.xs, rtol=3e-5, atol=1e-6)
    np.testing.assert_allclose(actual.Ps, expected.Ps, rtol=3e-5, atol=1e-6)

    def loss(t, reference=False):
        result = solve(t, reference)
        return jnp.sum(result.xs**2) + jnp.sum(result.Ps)

    reference_gradient = jax.grad(lambda t: loss(t, True))(theta)
    np.testing.assert_allclose(
        jax.jit(jax.grad(loss))(theta), reference_gradient, rtol=3e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        jax.jacfwd(loss)(theta), reference_gradient, rtol=3e-5, atol=1e-6
    )
    empty = cx.kalman(model, ys[:0], Q=jnp.eye(2, dtype=dtype), R=noise, solver=solver)
    assert empty.successful and empty.xs.shape == (0, 2)
    if dtype == jnp.float64:
        parameters = jnp.stack([theta, theta + 0.1])
        batched = jax.jit(jax.vmap(solve))(parameters)
        assert jnp.all(batched.successful)
        np.testing.assert_allclose(batched.xs[1], solve(parameters[1]).xs, atol=1e-10)

    # Eight identical sensors: exact conditional mean=8/(8+r), P=r/(8+r).
    # In FP32, r=1e-8 is lost when added to the dense rank-one innovation
    # covariance; state whitening preserves that observation precision.
    one = jnp.ones((1, 1), dtype=dtype)
    repeated = cx.dss(
        one * 0.9, one, jnp.ones((8, 1), dtype=dtype), jnp.zeros((8, 1), dtype=dtype), 1
    )
    tiny = jnp.full(8, 1e-8, dtype=dtype)
    observed = jnp.ones((1, 8), dtype=dtype)
    robust = cx.kalman(repeated, observed, Q=one * 0.01, R=tiny, solver=solver)
    assert robust.successful
    np.testing.assert_allclose(robust.xs, [[8 / (8 + 1e-8)]], atol=1e-6)
    np.testing.assert_allclose(robust.Ps, [[[1e-8 / (8 + 1e-8)]]], rtol=1e-5)
    if dtype == jnp.float32:
        dense = cx.kalman(repeated, observed, Q=one * 0.01, R=jnp.diag(tiny))
        assert dense.status == cx.solvers.SolverStatus.numerical_failure


def test_explicit_steady_design_matches_recursive_values_gradients_and_online_steps():
    ys, us = jnp.array([[0.2], [0.5], [0.4], [0.3]]), jnp.arange(4)[:, None] * 0.1
    x0 = jnp.ones(1) * 0.3

    def problem(theta):
        a, q, r = jnp.exp(theta)
        model = cx.dss(a[None, None], [[0.2]], [[1.0]], [[0.1]], 1)
        Q, R = q * jnp.eye(1), r * jnp.eye(1)
        return model, Q, R, cx.kalman_gain(model, Q, R)

    def solve(theta, reference=False):
        model, Q, R, design = problem(theta)
        return cx.kalman(
            model,
            ys,
            Q=Q,
            R=R,
            x0=x0,
            us=us,
            P0=design.P if reference else None,
            solver=None if reference else cx.solvers.SteadyStateKalman(design),
        )

    theta = jnp.log(jnp.array([0.8, 0.01, 0.1]))
    actual, expected = jax.jit(solve)(theta), solve(theta, True)
    assert actual.successful
    np.testing.assert_allclose(actual.xs, expected.xs, atol=1e-10)
    np.testing.assert_allclose(actual.Ps, expected.Ps, atol=1e-10)

    def loss(t, reference=False):
        result = solve(t, reference)
        return jnp.sum(result.xs**2) + jnp.sum(result.Ps)

    expected_gradient = jax.grad(lambda t: loss(t, True))(theta)
    np.testing.assert_allclose(
        jax.jit(jax.grad(loss))(theta), expected_gradient, atol=1e-8
    )
    np.testing.assert_allclose(jax.jacfwd(loss)(theta), expected_gradient, atol=1e-8)
    batched = jax.jit(jax.vmap(solve))(jnp.stack([theta, theta + 0.05]))
    assert jnp.all(batched.successful)
    model, Q, R, design = problem(theta)
    solver = cx.solvers.SteadyStateKalman(design)
    empty = cx.kalman(model, ys[:0], Q=Q, R=R, solver=solver)
    assert empty.successful and empty.Ps.shape == (0, 1, 1)
    online = cx.kalman_update(model, x0, design.P, ys[0], R=R, u=us[0], solver=solver)
    for k in range(1, 4):
        online = cx.kalman_step(
            model,
            online.x,
            online.P,
            ys[k],
            Q=Q,
            R=R,
            u=us[k - 1],
            u_measurement=us[k],
            solver=solver,
        )
        assert online.successful
    np.testing.assert_allclose(online.x, actual.final_state, atol=1e-10)
    for options in (
        {"P0": jnp.eye(1)},
        {"has_measurement": jnp.array([True, False, True, True])},
        {"R": 2 * R},
    ):
        kwargs = dict(Q=Q, R=R, solver=solver)
        kwargs.update(options)
        failed = cx.kalman(model, ys, **kwargs)
        assert failed.status == cx.solvers.SolverStatus.invalid_problem
    with pytest.raises(Exception, match="Kalman filter failed"):
        jax.jit(
            lambda: cx.kalman(model, ys, Q=Q, R=2 * R, solver=solver, throw=True).xs
        )().block_until_ready()
