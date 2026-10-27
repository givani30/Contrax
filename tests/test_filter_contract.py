"""Covariance form, control indexing and missing-data gradient contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import contrax as cx


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("method", ["joseph", "compact"])
def test_batch_matches_online_with_controls_feedthrough_and_missing_data(dtype, method):
    A = jnp.array([[0.9, 0.2], [0.0, 0.8]], dtype=dtype)
    B, C, D = (
        jnp.ones((2, 1), dtype=dtype),
        jnp.array([[1.0, 0.5]], dtype=dtype),
        jnp.ones((1, 1), dtype=dtype),
    )
    sys = cx.dss(A, B, C, D, dt=0.1)
    Q, R = 0.01 * jnp.eye(2, dtype=dtype), 0.1 * jnp.eye(1, dtype=dtype)
    ys = jnp.array([[1.0], [jnp.nan], [0.5]], dtype=dtype)
    us = jnp.array([[0.1], [0.2], [0.3]], dtype=dtype)
    flags = jnp.array([True, False, True])
    x0, P0 = jnp.ones(2, dtype=dtype), 2 * jnp.eye(2, dtype=dtype)
    solver = cx.solvers.Kalman(covariance_update=method)
    batch = jax.jit(
        lambda q: cx.kalman(
            sys, q, R, ys, x0, P0, us=us, has_measurement=flags, solver=solver
        )
    )(Q)
    x, P, innovation = cx.kalman_update(sys, x0, P0, ys[0], R, us[0], solver=solver)
    expected = [(x, P, innovation)]
    for k in range(1, 3):
        # The transition uses u[k-1]; the measurement uses u[k]. The step
        # convenience function shares one u, so separate helpers express this.
        x, P = cx.kalman_predict(sys, x, P, Q, us[k - 1])
        x, P, innovation = cx.kalman_update(
            sys, x, P, ys[k], R, us[k], has_measurement=flags[k], solver=solver
        )
        expected.append((x, P, innovation))
    for actual, reference in zip(
        (batch.x_hat, batch.P, batch.innovations), zip(*expected)
    ):
        np.testing.assert_allclose(actual, jnp.stack(reference), atol=1e-6, rtol=1e-6)
        assert actual.dtype == dtype
    assert jnp.all(jnp.linalg.eigvalsh(batch.P) >= 0)
    assert jnp.all(batch.innovations[1] == 0)


@pytest.mark.parametrize("extended", [False, True])
def test_missing_nan_measurements_have_finite_noise_gradients_under_jit_and_vmap(
    extended,
):
    sys = cx.dss(
        jnp.array([[0.9]]), jnp.ones((1, 1)), jnp.ones((1, 1)), jnp.zeros((1, 1)), dt=1
    )
    ys, us = jnp.array([[1.0], [jnp.nan], [0.4]]), jnp.ones((3, 1))
    flags = jnp.array([True, False, True])

    def loss(log_noise):
        Q, R = jnp.exp(log_noise[0]) * jnp.eye(1), jnp.exp(log_noise[1]) * jnp.eye(1)
        if extended:
            model = cx.nonlinear_system(
                lambda t, x, u: sys.A @ x + sys.B @ u,
                output=lambda t, x, u: sys.C @ x + sys.D @ u,
                dt=1.0,
            )
            result = cx.ekf(
                model,
                Q,
                R,
                ys,
                us,
                jnp.zeros(1),
                jnp.eye(1),
                has_measurement=flags,
                solver=cx.solvers.EKF(num_iter=3),
            )
        else:
            result = cx.kalman(sys, Q, R, ys, us=us, has_measurement=flags)
        return jnp.sum(result.x_hat**2) + jnp.sum(result.P)

    parameters = jnp.array([-3.0, -2.0])
    gradient = jax.jit(jax.grad(loss))(parameters)
    assert jnp.all(jnp.isfinite(gradient))
    batched = jax.jit(jax.vmap(jax.grad(loss)))(
        jnp.stack([parameters, parameters + 0.1])
    )
    assert jnp.all(jnp.isfinite(batched))
    eps = 1e-5
    for k in range(2):
        delta = jnp.eye(2)[k] * eps
        reference = (loss(parameters + delta) - loss(parameters - delta)) / (2 * eps)
        np.testing.assert_allclose(gradient[k], reference, rtol=1e-6, atol=1e-7)


def test_missing_observation_does_not_evaluate_invalid_measurement_model():
    """sqrt at a negative predicted mean would poison h and its Jacobian."""

    def loss(x):
        mean, covariance, innovation = cx.ekf_update(
            lambda x: jnp.sqrt(x),
            x,
            jnp.eye(1),
            jnp.array([jnp.nan]),
            jnp.eye(1),
            has_measurement=False,
        )
        return jnp.sum(mean + covariance + innovation)

    assert jnp.isfinite(jax.jit(loss)(jnp.array([-1.0])))
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(jnp.array([-1.0])), [1.0])


def test_batch_iterated_ekf_matches_online_update_and_empty_horizon():
    def f(t, x, u):
        return 0.9 * x + u

    def h(x):
        return x**2

    Q, R, x0, P0 = jnp.eye(1) * 0.01, jnp.eye(1) * 0.1, jnp.ones(1), jnp.eye(1)
    solver = cx.solvers.EKF(num_iter=4)
    batch = cx.ekf(
        f,
        Q,
        R,
        jnp.array([[2.0]]),
        jnp.zeros((1, 1)),
        x0,
        P0,
        observation=h,
        solver=solver,
    )
    online = cx.ekf_update(h, x0, P0, jnp.array([2.0]), R, solver=solver)
    np.testing.assert_allclose(batch.x_hat[0], online[0], atol=1e-12)
    np.testing.assert_allclose(batch.P[0], online[1], atol=1e-12)
    empty = cx.ekf(
        f,
        Q,
        R,
        jnp.empty((0, 1)),
        jnp.empty((0, 1)),
        x0,
        P0,
        observation=h,
        solver=solver,
    )
    assert empty.x_hat.shape == (0, 1)
