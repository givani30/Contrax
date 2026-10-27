"""Binary quadratic composition checked against explicit state/input sums."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import contrax as cx


@pytest.mark.parametrize("steps", [0, 1, 2, 3, 17, 64])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_binary_cost_and_gradients_match_explicit_trajectory(steps, dtype):
    A = jnp.array([[1.0, 0.2], [0.0, 1.1]], dtype=dtype)
    B, C, D = (
        jnp.array([[0.0], [0.2]], dtype=dtype),
        jnp.eye(2, dtype=dtype),
        jnp.zeros((2, 1), dtype=dtype),
    )
    sys = cx.dss(A, B, C, D, dt=0.2)
    K = jnp.array([[0.3, 1.0]], dtype=dtype)
    Q, R, terminal = (
        jnp.eye(2, dtype=dtype),
        0.2 * jnp.eye(1, dtype=dtype),
        0.7 * jnp.eye(2, dtype=dtype),
    )
    x0 = jnp.array([1.0, -0.4], dtype=dtype)

    def binary(k, x):
        return cx.quadratic_rollout_cost(
            sys, k, Q, R, x, num_steps=steps, Q_terminal=terminal
        )

    def explicit(k, x):
        def step(x, _):
            u = -k @ x
            return A @ x + B @ u, x @ Q @ x + u @ R @ u

        final, terms = jax.lax.scan(step, x, None, length=steps)
        return jnp.sum(terms) + final @ terminal @ final

    actual = jax.jit(binary)(K, x0)
    assert actual.ndim == 0 and actual.dtype == dtype
    tolerance = 3e-5 if dtype == jnp.float32 else 1e-10
    np.testing.assert_allclose(actual, explicit(K, x0), rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(
        jax.grad(binary)(K, x0),
        jax.grad(explicit)(K, x0),
        rtol=tolerance,
        atol=tolerance,
    )
    _, forward = jax.jvp(binary, (K, x0), (jnp.ones_like(K), jnp.zeros_like(x0)))
    np.testing.assert_allclose(
        forward, jnp.sum(jax.grad(explicit)(K, x0)), rtol=tolerance, atol=tolerance
    )
    batch = jax.jit(jax.vmap(lambda x: binary(K, x)))(jnp.stack([x0, 2 * x0]))
    np.testing.assert_allclose(batch[1], 4 * batch[0], rtol=tolerance)


def test_zero_horizon_and_invalid_configuration():
    sys = cx.dss(jnp.eye(1), jnp.eye(1), jnp.eye(1), jnp.zeros((1, 1)), 1)
    args = (sys, jnp.zeros((1, 1)), jnp.eye(1), jnp.eye(1), jnp.ones(1))
    assert cx.quadratic_rollout_cost(*args, num_steps=0) == 0
    with pytest.raises(ValueError, match="nonnegative"):
        cx.quadratic_rollout_cost(*args, num_steps=-1)
