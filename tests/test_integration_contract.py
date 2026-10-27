"""Integration controls, numerical status and sampled-model derivatives."""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import contrax as cx


@pytest.mark.parametrize("solver", [diffrax.Tsit5(), diffrax.Kvaerno3()])
def test_sampled_explicit_and_implicit_model_jacobians_and_reverse_losses(solver):
    model = cx.nonlinear_system(lambda t, x, u: -2 * x + u)
    sampled = cx.sample_system(
        model,
        0.1,
        solver=solver,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
        max_steps=128,
    )
    x, u = jnp.ones(1), jnp.array([0.4])
    result = jax.jit(lambda t, state, control: sampled.step(t, state, control))(
        0.0, x, u
    )
    expected = 0.2 + 0.8 * jnp.exp(-0.2)
    assert result.successful and result.num_steps > 0
    np.testing.assert_allclose(result.x[0], expected, atol=1e-7)
    jacobian = jax.jit(jax.jacfwd(sampled.dynamics, argnums=1))(0.0, x, u)
    np.testing.assert_allclose(jacobian[0, 0], jnp.exp(-0.2), atol=1e-7)
    gradient = jax.jit(
        jax.grad(lambda state: jnp.sum(sampled.dynamics(0.0, state, u) ** 2))
    )(x)
    np.testing.assert_allclose(gradient[0], 2 * expected * jnp.exp(-0.2), atol=1e-6)


@pytest.mark.parametrize("solver", [diffrax.Tsit5(), diffrax.Kvaerno3()])
def test_continuous_simulation_status_and_parameter_gradient(solver):
    def solve(a):
        sys = cx.ss(-a[None, None], jnp.eye(1), jnp.eye(1), jnp.zeros((1, 1)))
        return cx.simulate(
            sys,
            jnp.ones(1),
            lambda t, x: jnp.zeros(1),
            duration=0.2,
            dt=0.1,
            solver=solver,
            stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
            max_steps=128,
        )

    result = jax.jit(solve)(jnp.array(2.0))
    assert result.successful and result.num_steps > 0
    np.testing.assert_allclose(result.xs[-1, 0], jnp.exp(-0.4), atol=1e-7)
    gradient = jax.jit(jax.grad(lambda a: solve(a).xs[-1, 0]))(jnp.array(2.0))
    np.testing.assert_allclose(gradient, -0.2 * jnp.exp(-0.4), atol=1e-7)


def test_simulation_and_sampled_limits_report_status_and_checked_failures():
    sys = cx.ss([[-2.0]], [[1.0]], [[1.0]], [[0.0]])

    def simulate(throw=False):
        return cx.simulate(
            sys,
            jnp.ones(1),
            lambda t, x: jnp.zeros(1),
            duration=1,
            dt=0.1,
            dt0=1e-3,
            max_steps=1,
            throw=throw,
        )

    assert jax.jit(simulate)().status == cx.solvers.SolverStatus.max_steps_reached
    with pytest.raises(Exception, match="Continuous simulation failed"):
        jax.jit(lambda: simulate(True).ys)().block_until_ready()
    model = cx.nonlinear_system(lambda t, x, u: -2 * x + u)
    sampled = cx.sample_system(model, 1.0, dt0=1e-3, max_steps=1)
    result = jax.jit(lambda t, state, control: sampled.step(t, state, control))(
        0.0, jnp.ones(1), jnp.zeros(1)
    )
    assert result.status == cx.solvers.SolverStatus.max_steps_reached
    assert not result.successful
    trajectory = cx.simulate(
        sampled, jnp.ones(1), lambda t, x: jnp.zeros(1), num_steps=3
    )
    assert trajectory.status == cx.solvers.SolverStatus.max_steps_reached
    assert trajectory.num_steps == result.num_steps
    assert jnp.all(jnp.isnan(sampled.dynamics(0.0, jnp.ones(1), jnp.zeros(1))))
    checked = cx.sample_system(model, 1.0, dt0=1e-3, max_steps=1, throw=True)
    with pytest.raises(Exception, match="Sample integration failed"):
        jax.jit(checked.dynamics)(0.0, jnp.ones(1), jnp.zeros(1)).block_until_ready()


def test_zero_horizons_and_response_wrappers_keep_integration_controls():
    sys = cx.ss([[-2.0]], [[1.0]], [[1.0]], [[0.0]])
    zero = cx.simulate(sys, jnp.ones(1), lambda t, x: jnp.zeros(1), duration=0)
    assert zero.successful and zero.num_steps == 0
    np.testing.assert_allclose(zero.xs, [[1.0]])
    for response in (cx.step_response, cx.impulse_response):
        result = response(sys, duration=1.0, dt=0.1, dt0=1e-3, max_steps=1)
        assert result.status == cx.solvers.SolverStatus.max_steps_reached
    d = cx.dss([[0.8]], [[1.0]], [[1.0]], [[0.0]], 1)
    result = cx.impulse_response(d, num_steps=0)
    assert result.successful and result.xs.shape == (1, 1) and result.ys.shape == (0, 1)
    with pytest.raises(ValueError, match="continuous-only"):
        cx.simulate(
            d,
            jnp.ones(1),
            lambda t, x: jnp.zeros(1),
            num_steps=1,
            solver=diffrax.Tsit5(),
        )


def test_checked_design_failure_cannot_be_pruned_when_only_gain_is_returned():
    one = jnp.ones((1, 1))
    with pytest.raises(Exception, match="Riccati solve failed"):
        jax.jit(
            lambda: cx.dare(one, one, one, -one, throw=True).K
        )().block_until_ready()


def test_float32_sampled_estimation_preserves_precision_with_x64_enabled():
    model = cx.nonlinear_system(
        lambda t, x, u: -x + u + 0.01 * t, output=lambda t, x, u: x + 0.1 * t
    )
    sampled = cx.sample_system(model, 0.1, method="rk4")
    eye = jnp.eye(1, dtype=jnp.float32)
    result = jax.jit(
        lambda: cx.ekf(
            sampled,
            0.01 * eye,
            0.1 * eye,
            jnp.zeros((3, 1), dtype=jnp.float32),
            jnp.ones((3, 1), dtype=jnp.float32),
            jnp.zeros(1, dtype=jnp.float32),
            eye,
        )
    )()
    assert result.x_hat.dtype == result.P.dtype == jnp.float32


def test_continuous_vector_fields_require_sampling_before_estimation():
    model = cx.nonlinear_system(lambda t, x, u: -x + u)
    x, P, u = jnp.ones(1), jnp.eye(1), jnp.zeros(1)
    with pytest.raises(ValueError, match="discrete model"):
        cx.ekf_predict(model, x, P, u, P)
    with pytest.raises(ValueError, match="discrete model"):
        cx.ukf(model, P, P, jnp.ones((2, 1)), jnp.zeros((2, 1)), x, P)
    with pytest.raises(ValueError, match="discrete model"):
        cx.mhe_objective(
            model, jnp.ones((2, 1)), jnp.zeros((1, 1)), jnp.ones((2, 1)), x, P, P, P
        )
