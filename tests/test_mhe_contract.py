"""MHE model, horizon, failure and implicit gradient contracts.

The independent linear-Gaussian oracle solves the trajectory's normal equations:
H = arrival precision + G.T G/q + I/r, where Gx gives x[k+1]-a*x[k].
"""

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest

import contrax as cx


def _linear_problem(parameters):
    a, q, r = jnp.exp(parameters)
    zero, eye = (
        jnp.zeros((1, 1), dtype=parameters.dtype),
        jnp.eye(1, dtype=parameters.dtype),
    )
    sys = cx.dss(a[None, None], zero, eye, zero, 1)
    return sys, q * eye, r * eye


def test_model_and_noise_implicit_gradients_match_normal_equations():
    N = 4
    ys = jnp.array([[0.2], [0.5], [0.3], [0.4]])
    parameters = jnp.log(jnp.array([0.8, 0.1, 0.2]))

    def solved(theta):
        sys, Q, R = _linear_problem(theta)
        result = cx.mhe(
            sys,
            jnp.zeros((N, 1)),
            jnp.zeros((N - 1, 1)),
            ys,
            jnp.zeros(1),
            jnp.eye(1),
            Q,
            R,
            solver=optx.LBFGS(rtol=1e-10, atol=1e-10),
        )
        return result.x_hat[0]

    def reference(theta):
        a, q, r = jnp.exp(theta)
        eye = jnp.eye(N)
        G = eye[1:] - a * eye[:-1]
        precision = G.T @ G / q + eye / r
        precision = precision.at[0, 0].add(1)
        return jnp.linalg.solve(precision, ys[:, 0] / r)[-1]

    np.testing.assert_allclose(
        jax.jit(solved)(parameters), reference(parameters), atol=1e-7
    )
    actual = jax.jit(jax.grad(solved))(parameters)
    np.testing.assert_allclose(
        actual, jax.grad(reference)(parameters), atol=1e-6, rtol=1e-5
    )
    _, forward = jax.jvp(solved, (parameters,), (jnp.ones_like(parameters),))
    np.testing.assert_allclose(forward, jnp.sum(actual), atol=1e-6)


def test_callable_parameters_are_optimizer_arguments():
    def transition(x, u, parameters):
        return parameters * x

    def output(x, parameters):
        return x

    def loss(a):
        return cx.mhe(
            transition,
            jnp.zeros((3, 1)),
            jnp.zeros((2, 1)),
            jnp.array([[0.2], [0.3], [0.4]]),
            jnp.zeros(1),
            jnp.eye(1),
            0.1 * jnp.eye(1),
            0.2 * jnp.eye(1),
            observation=output,
            params=a,
        ).x_hat[0]

    actual = jax.jit(jax.grad(loss))(jnp.array(0.8))
    eps = 1e-4
    expected = (loss(0.8 + eps) - loss(0.8 - eps)) / (2 * eps)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)


def test_one_state_window_and_nonconvergence_diagnostics():
    sys, Q, R = _linear_problem(jnp.log(jnp.array([0.8, 0.1, 0.2])))
    args = (
        sys,
        jnp.zeros((1, 1)),
        jnp.empty((0, 1)),
        jnp.ones((1, 1)),
        jnp.zeros(1),
        jnp.eye(1),
        Q,
        R,
    )
    converged = cx.mhe(*args)
    assert converged.successful
    np.testing.assert_allclose(converged.x_hat, [1 / 1.2], atol=1e-5)
    limited = jax.jit(lambda: cx.mhe(*args, max_steps=0))()
    assert limited.status == cx.solvers.SolverStatus.max_steps_reached
    assert not limited.successful
    assert jnp.isfinite(limited.final_cost)
    assert limited.final_cost <= limited.initial_cost
    assert limited.num_steps == 0
    assert limited.gradient_norm > 0
    with pytest.raises(Exception, match="MHE solve failed"):
        cx.mhe(*args, max_steps=0, throw=True)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_mhe_preserves_precision_and_batches(dtype):
    sys, Q, R = _linear_problem(jnp.log(jnp.array([0.8, 0.1, 0.2], dtype=dtype)))

    def solve(y):
        return cx.mhe(
            sys,
            jnp.zeros((1, 1), dtype=dtype),
            jnp.empty((0, 1), dtype=dtype),
            y[None, None],
            jnp.zeros(1, dtype=dtype),
            jnp.eye(1, dtype=dtype),
            Q,
            R,
            solver=optx.BFGS(rtol=1e-6, atol=1e-6),
        )

    result = jax.jit(jax.vmap(solve))(jnp.array([1.0, 2.0], dtype=dtype))
    assert result.xs.dtype == dtype
    assert jnp.all(result.successful)
    np.testing.assert_allclose(result.x_hat[:, 0], [1 / 1.2, 2 / 1.2], atol=1e-5)


def test_default_optimizer_float32_context_and_limited_accepted_iterate():
    with jax.enable_x64(False):
        eye = jnp.eye(1, dtype=jnp.float32)
        sys = cx.dss(0.8 * eye, eye, eye, 0 * eye, 1)
        args = (
            sys,
            jnp.zeros((4, 1), dtype=jnp.float32),
            jnp.zeros((3, 1), dtype=jnp.float32),
            jnp.array([[0.2], [0.5], [0.3], [0.4]], dtype=jnp.float32),
            jnp.zeros(1, dtype=jnp.float32),
            eye,
            0.1 * eye,
            0.2 * eye,
        )
        result = cx.mhe(*args)
        assert result.xs.dtype == jnp.float32
        assert jnp.isfinite(result.final_cost)
        assert not jax.config.jax_enable_x64
        limited = cx.mhe(*args, max_steps=10)
        assert not limited.successful
        assert limited.status == cx.solvers.SolverStatus.max_steps_reached
        assert jnp.all(jnp.isfinite(limited.xs))
        assert limited.final_cost < limited.initial_cost


def test_system_object_observations_include_terminal_feedthrough_and_time():
    sys = cx.nonlinear_system(
        lambda t, x, u: 0.8 * x + u, output=lambda t, x, u: x + 2 * u + t, dt=0.5
    )
    xs = jnp.array([[1.0], [1.8]])
    us = jnp.ones((1, 1))
    measurement_us = jnp.array([[1.0], [2.0]])
    ys = jnp.array([[3.0], [6.3]])
    cost = jax.jit(
        lambda x: cx.mhe_objective(
            sys,
            x,
            us,
            ys,
            xs[0],
            jnp.eye(1),
            jnp.eye(1),
            jnp.eye(1),
            measurement_us=measurement_us,
        )
    )(xs)
    np.testing.assert_allclose(cost, 0, atol=1e-12)
    with pytest.raises(ValueError, match="N-1"):
        cx.mhe_objective(
            sys, xs, jnp.ones((2, 1)), ys, xs[0], jnp.eye(1), jnp.eye(1), jnp.eye(1)
        )
