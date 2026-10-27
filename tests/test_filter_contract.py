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
            sys, ys, Q=q, R=R, x0=x0, P0=P0, us=us, has_measurement=flags, solver=solver
        )
    )(Q)
    x, P, innovation = cx.kalman_update(sys, x0, P0, ys[0], R=R, u=us[0], solver=solver)
    expected = [(x, P, innovation)]
    for k in range(1, 3):
        # The transition uses u[k-1]; the measurement uses u[k]. The step
        # convenience function shares one u, so separate helpers express this.
        x, P = cx.kalman_predict(sys, x, P, Q=Q, u=us[k - 1])
        x, P, innovation = cx.kalman_update(
            sys, x, P, ys[k], R=R, u=us[k], has_measurement=flags[k], solver=solver
        )
        expected.append((x, P, innovation))
    for actual, reference in zip(
        (batch.xs, batch.Ps, batch.innovations), zip(*expected)
    ):
        np.testing.assert_allclose(actual, jnp.stack(reference), atol=1e-6, rtol=1e-6)
        assert actual.dtype == dtype
    assert jnp.all(jnp.linalg.eigvalsh(batch.Ps) >= 0)
    assert jnp.all(batch.innovations[1] == 0)
    assert batch.successful


@pytest.mark.parametrize("kind", ["kalman", "ekf", "ukf"])
def test_missing_nan_measurements_have_finite_noise_gradients_under_jit_and_vmap(
    kind,
):
    sys = cx.dss(
        jnp.array([[0.9]]), jnp.ones((1, 1)), jnp.ones((1, 1)), jnp.zeros((1, 1)), dt=1
    )
    ys, us = jnp.array([[1.0], [jnp.nan], [0.4]]), jnp.ones((3, 1))
    flags = jnp.array([True, False, True])

    def loss(log_noise):
        Q, R = jnp.exp(log_noise[0]) * jnp.eye(1), jnp.exp(log_noise[1]) * jnp.eye(1)
        if kind == "ekf":
            model = cx.nonlinear_system(
                lambda t, x, u: sys.A @ x + sys.B @ u,
                output=lambda t, x, u: sys.C @ x + sys.D @ u,
                dt=1.0,
            )
            result = cx.ekf(
                model,
                ys,
                Q=Q,
                R=R,
                us=us,
                x0=jnp.zeros(1),
                P0=jnp.eye(1),
                has_measurement=flags,
                solver=cx.solvers.EKF(num_iter=3),
            )
        else:
            result = getattr(cx, kind)(sys, ys, Q=Q, R=R, us=us, has_measurement=flags)
        return jnp.sum(result.xs**2) + jnp.sum(result.Ps)

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
            R=jnp.eye(1),
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
        jnp.array([[2.0]]),
        Q=Q,
        R=R,
        us=jnp.zeros((1, 1)),
        x0=x0,
        P0=P0,
        observation=h,
        solver=solver,
    )
    online = cx.ekf_update(h, x0, P0, jnp.array([2.0]), R=R, solver=solver)
    np.testing.assert_allclose(batch.xs[0], online.x, atol=1e-12)
    np.testing.assert_allclose(batch.Ps[0], online.P, atol=1e-12)
    assert online.successful
    empty = cx.ekf(
        f,
        jnp.empty((0, 1)),
        Q=Q,
        R=R,
        us=jnp.empty((0, 1)),
        x0=x0,
        P0=P0,
        observation=h,
        solver=solver,
    )
    assert empty.xs.shape == (0, 1)
    np.testing.assert_allclose(batch.final_state, batch.xs[-1])
    with pytest.raises(ValueError, match="empty trajectory"):
        _ = empty.final_state
    linear = cx.dss([[0.9]], [[1.0]], [[1.0]], [[0.0]], 1.0)
    for filter_fn in (cx.kalman, cx.ukf):
        empty = filter_fn(linear, jnp.empty((0, 1)), Q=Q, R=R)
        assert empty.xs.shape == (0, 1)
        with pytest.raises(ValueError, match="empty trajectory"):
            _ = empty.final_state
    empty_innovations, empty_likelihood = cx.ukf_diagnostics(empty)
    assert not empty_innovations.nonfinite and not empty_likelihood.nonfinite
    assert empty_likelihood.total_log_likelihood == 0
    empty_smoother = cx.uks(empty)
    assert empty_smoother.successful
    assert not cx.smoother_diagnostics(empty_smoother, empty).nonfinite


def test_sampled_foh_batch_filters_match_exact_linear_model_and_default_controls():
    """x'=u has exact FOH increment dt*(u_left+u_right)/2."""
    model = cx.sample_system(
        cx.nonlinear_system(
            lambda t, x, u: u,
            output=lambda t, x, u: x + 2 * u,
            input_dim=1,
        ),
        0.2,
        method="rk4",
        input_interpolation="foh",
    )
    exact = cx.dss([[1.0]], [[0.1, 0.1]], [[1.0]], [[2.0, 0.0]], 0.2)
    ys = jnp.array([[2.0], [4.2], [6.5]])
    us = jnp.array([[[1.0], [2.0]], [[2.0], [3.0]], [[3.0], [4.0]]])
    Q, R = jnp.eye(1) * 0.01, jnp.eye(1) * 0.1
    expected = cx.kalman(exact, ys, Q=Q, R=R, us=us[..., 0])
    for filter_fn in (cx.ekf, cx.ukf):
        actual = jax.jit(lambda: filter_fn(model, ys, Q=Q, R=R, us=us))()
        np.testing.assert_allclose(actual.xs, expected.xs, atol=1e-10)
        np.testing.assert_allclose(actual.Ps, expected.Ps, atol=1e-10)
        omitted = filter_fn(model, ys, Q=Q, R=R)
        explicit = filter_fn(model, ys, Q=Q, R=R, us=jnp.zeros_like(us))
        np.testing.assert_allclose(omitted.xs, explicit.xs, atol=1e-12)


@pytest.mark.parametrize("filter_fn", [cx.kalman, cx.ekf, cx.ukf])
def test_filter_status_distinguishes_invalid_noise_and_nonfinite_observations(
    filter_fn,
):
    model = cx.dss([[0.9]], [[1.0]], [[1.0]], [[0.0]], 1.0)
    eye = jnp.eye(1)
    results = jax.jit(
        jax.vmap(lambda q: filter_fn(model, jnp.ones((2, 1)), Q=q * eye, R=eye))
    )(jnp.array([0.1, -1.0]))
    np.testing.assert_array_equal(
        results.status, [0, cx.solvers.SolverStatus.invalid_problem]
    )
    assert results.final_state.shape == (2, 1)
    nonfinite = filter_fn(model, jnp.full((1, 1), jnp.nan), Q=eye, R=eye)
    assert nonfinite.status == cx.solvers.SolverStatus.numerical_failure
    if filter_fn is not cx.ukf:
        update_fn = cx.kalman_update if filter_fn is cx.kalman else cx.ekf_update
        point = update_fn(model, jnp.zeros(1), eye, jnp.ones(1), R=-2 * eye)
        assert point.status == cx.solvers.SolverStatus.invalid_problem
        assert point.x.shape == (1,) and point.P.shape == (1, 1)
    with pytest.raises(Exception, match="filter failed"):
        jax.jit(
            lambda: (
                filter_fn(
                    model,
                    jnp.ones((1, 1)),
                    Q=-eye,
                    R=eye,
                    throw=True,
                ).xs
            )
        )().block_until_ready()
