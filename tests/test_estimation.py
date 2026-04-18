"""Tests for contrax.estimation — Kalman, EKF, UKF, and RTS smoother."""

import jax
import jax.numpy as jnp

import contrax as cx


def _scalar_sys():
    return cx.dss(
        A=jnp.array([[0.8]]),
        B=jnp.zeros((1, 1)),
        C=jnp.array([[1.0]]),
        D=jnp.zeros((1, 1)),
        dt=1.0,
    )


def test_kalman_shapes():
    sys = _scalar_sys()
    ys = jnp.ones((20, 1))
    result = cx.kalman(
        sys,
        Q_noise=jnp.array([[1e-3]]),
        R_noise=jnp.array([[1e-2]]),
        ys=ys,
    )

    assert result.x_hat.shape == (20, 1)
    assert result.P.shape == (20, 1, 1)
    assert result.innovations.shape == (20, 1)


def test_kalman_converges_to_constant_measurement():
    """For y_k=1 and A=C=1-ish stable scalar model, estimate should approach y."""
    sys = cx.dss(
        A=jnp.array([[1.0]]),
        B=jnp.zeros((1, 1)),
        C=jnp.array([[1.0]]),
        D=jnp.zeros((1, 1)),
        dt=1.0,
    )
    ys = jnp.ones((80, 1))
    result = cx.kalman(
        sys,
        Q_noise=jnp.array([[1e-4]]),
        R_noise=jnp.array([[1e-3]]),
        ys=ys,
        x0=jnp.array([0.0]),
        P0=jnp.array([[1.0]]),
    )

    assert abs(float(result.x_hat[-1, 0]) - 1.0) < 1e-3
    assert abs(float(result.innovations[-1, 0])) < 1e-3


def test_kalman_covariance_remains_symmetric_positive():
    sys = _scalar_sys()
    ys = jnp.linspace(0.0, 1.0, 30)[:, None]
    result = cx.kalman(
        sys,
        Q_noise=jnp.array([[1e-3]]),
        R_noise=jnp.array([[1e-2]]),
        ys=ys,
    )

    assert jnp.allclose(result.P, jnp.swapaxes(result.P, -1, -2), atol=1e-12)
    assert jnp.all(result.P[:, 0, 0] >= 0.0)


def test_kalman_step_matches_batch_filter():
    """kalman_step (predict-from-posterior-at-k-1) reproduces the batch filter.

    The batch kalman() treats (x0, P0) as the prior on x_0 and updates first.
    kalman_step takes the posterior at k-1 and does predict-then-update to
    return the posterior at k.  Together:
      - step 0: kalman_update(x0, P0, y[0])
      - steps 1+: kalman_step(posterior_{k-1}, y[k])
    reproduce the full batch output.
    """
    sys = _scalar_sys()
    ys = jnp.linspace(0.0, 1.0, 20)[:, None]
    Q = jnp.array([[1e-3]])
    R = jnp.array([[1e-2]])
    x0 = jnp.array([0.0])
    P0 = jnp.array([[1.0]])
    batch = cx.kalman(sys, Q, R, ys, x0=x0, P0=P0)

    # Step 0: update x0 directly (no predict — x0 is prior on x_0)
    x_hat_0, P_hat_0, innov_0 = cx.kalman_update(sys, R, ys[0], x0, P0)

    # Steps 1+: kalman_step from the previous posterior
    def one_step(carry, y):
        x, P = carry
        x_new, P_new, innov = cx.kalman_step(sys, Q, R, y, x, P)
        return (x_new, P_new), (x_new, P_new, innov)

    _, (x_hats_rest, Ps_rest, innovs_rest) = jax.lax.scan(
        one_step, (x_hat_0, P_hat_0), ys[1:]
    )
    x_hats = jnp.concatenate([x_hat_0[None], x_hats_rest], axis=0)
    Ps = jnp.concatenate([P_hat_0[None], Ps_rest], axis=0)
    innovations = jnp.concatenate([innov_0[None], innovs_rest], axis=0)

    assert jnp.allclose(x_hats, batch.x_hat, atol=1e-12)
    assert jnp.allclose(Ps, batch.P, atol=1e-12)
    assert jnp.allclose(innovations, batch.innovations, atol=1e-12)


def test_kalman_step_can_skip_missing_measurement_under_jit():
    sys = _scalar_sys()
    Q = jnp.array([[1e-3]])
    R = jnp.array([[1e-2]])
    y = jnp.array([10.0])
    x = jnp.array([1.0])
    P = jnp.array([[0.5]])

    def step(has_measurement):
        return cx.kalman_step(sys, Q, R, y, x, P, has_measurement=has_measurement)

    x_skip, P_skip, innov_skip = jax.jit(step)(jnp.array(False))
    x_pred, P_pred = cx.kalman_predict(sys, Q, x, P)

    assert jnp.allclose(x_skip, x_pred, atol=1e-12)
    assert jnp.allclose(P_skip, P_pred, atol=1e-12)
    assert jnp.allclose(innov_skip, jnp.zeros_like(y), atol=1e-12)

    x_update, _, innov_update = jax.jit(step)(jnp.array(True))
    assert not jnp.allclose(x_update, x_pred)
    assert not jnp.allclose(innov_update, jnp.zeros_like(y))


def test_kalman_gain_matches_dare_dual_and_covariance_fixed_point():
    sys = cx.dss(
        A=jnp.array([[0.9, 0.1], [0.0, 0.8]]),
        B=jnp.zeros((2, 1)),
        C=jnp.array([[1.0, 0.0]]),
        D=jnp.zeros((1, 1)),
        dt=1.0,
    )
    Q = jnp.diag(jnp.array([1e-3, 2e-3]))
    R = jnp.array([[1e-2]])

    design = cx.kalman_gain(sys, Q, R)
    P_expected = cx.dare(sys.A.T, sys.C.T, Q, R).S
    S = sys.C @ P_expected @ sys.C.T + R
    K_expected = jnp.linalg.solve(S.T, (P_expected @ sys.C.T).T).T

    assert design.K.shape == (2, 1)
    assert design.P.shape == (2, 2)
    assert design.poles.shape == (2,)
    assert jnp.allclose(design.P, P_expected, atol=1e-10)
    assert jnp.allclose(design.K, K_expected, atol=1e-10)

    P_filtered = (jnp.eye(2) - design.K @ sys.C) @ design.P
    P_next = sys.A @ P_filtered @ sys.A.T + Q
    assert jnp.allclose(P_next, design.P, atol=1e-9)
    assert jnp.all(jnp.abs(design.poles) < 1.0)


def test_kalman_grad_through_noise_parameters_finite():
    sys = _scalar_sys()
    ys = jnp.sin(jnp.linspace(0.0, 1.0, 25))[:, None]

    def loss(log_q, log_r):
        Q = jnp.exp(log_q)[None, None]
        R = jnp.exp(log_r)[None, None]
        result = cx.kalman(sys, Q, R, ys)
        return jnp.sum(result.innovations**2) + jnp.sum(result.P)

    dlog_q, dlog_r = jax.grad(loss, argnums=(0, 1))(jnp.array(-5.0), jnp.array(-3.0))

    assert jnp.isfinite(dlog_q)
    assert jnp.isfinite(dlog_r)


def test_ekf_matches_kalman_for_linear_system():
    sys = _scalar_sys()
    ys = jnp.linspace(0.0, 1.0, 20)[:, None]
    us = jnp.zeros((20, 1))
    Q = jnp.array([[1e-3]])
    R = jnp.array([[1e-2]])
    x0 = jnp.array([0.0])
    P0 = jnp.array([[1.0]])

    kf = cx.kalman(sys, Q, R, ys, x0=x0, P0=P0)

    def f(x, u):
        return sys.A @ x + sys.B @ u

    def h(x):
        return sys.C @ x

    ekf = cx.ekf(f, Q, R, ys, us, x0, P0, observation=h)

    assert jnp.allclose(ekf.x_hat, kf.x_hat, atol=1e-12)
    assert jnp.allclose(ekf.P, kf.P, atol=1e-12)
    assert jnp.allclose(ekf.innovations, kf.innovations, atol=1e-12)


def test_ekf_accepts_nonlinear_system_object():
    scalar_sys = _scalar_sys()
    sys = cx.nonlinear_system(
        lambda t, x, u: scalar_sys.A @ x + scalar_sys.B @ u,
        observation=lambda t, x, u: scalar_sys.C @ x,
        dt=1.0,
        state_dim=1,
        input_dim=1,
        output_dim=1,
    )
    ys = jnp.linspace(0.0, 1.0, 20)[:, None]
    us = jnp.zeros((20, 1))
    Q = jnp.array([[1e-3]])
    R = jnp.array([[1e-2]])
    x0 = jnp.array([0.0])
    P0 = jnp.array([[1.0]])

    result = cx.ekf(sys, Q, R, ys, us, x0, P0)
    reference = cx.kalman(scalar_sys, Q, R, ys, x0=x0, P0=P0)

    assert jnp.allclose(result.x_hat, reference.x_hat, atol=1e-12)
    assert jnp.allclose(result.P, reference.P, atol=1e-12)
    assert jnp.allclose(result.innovations, reference.innovations, atol=1e-12)


def test_ekf_step_matches_batch_filter():
    """ekf_step reproduces the batch EKF.

    The batch ekf() uses (x0, P0) as the prior on x_0.  ekf_step does
    predict-then-update from the posterior at k-1.  Together:
      - step 0: ekf_update(x0, P0, y[0], u[0])
      - steps 1+: ekf_step(posterior_{k-1}, y[k], u[k])
    reproduce the full batch output.
    """
    sys = _scalar_sys()
    ys = jnp.linspace(0.0, 1.0, 20)[:, None]
    us = jnp.zeros((20, 1))
    Q = jnp.array([[1e-3]])
    R = jnp.array([[1e-2]])
    x0 = jnp.array([0.0])
    P0 = jnp.array([[1.0]])

    nonlinear = cx.nonlinear_system(
        lambda t, x, u: sys.A @ x + sys.B @ u,
        observation=lambda t, x, u: sys.C @ x,
        dt=1.0,
        state_dim=1,
        input_dim=1,
        output_dim=1,
    )

    batch = cx.ekf(nonlinear, Q, R, ys, us, x0, P0)

    # Step 0: update x0 directly (x0 is prior on x_0)
    def f(x, u):
        return sys.A @ x + sys.B @ u

    def h(x):
        return sys.C @ x

    x_hat_0, P_hat_0, innov_0 = cx.ekf_update(h, R, ys[0], x0, P0)

    # Steps 1+: ekf_step from the previous posterior
    def one_step(carry, inputs):
        x, P = carry
        y, u = inputs
        x_new, P_new, innov = cx.ekf_step(
            f,
            Q,
            R,
            y,
            u,
            x,
            P,
            observation=h,
        )
        return (x_new, P_new), (x_new, P_new, innov)

    _, (x_hats_rest, Ps_rest, innovs_rest) = jax.lax.scan(
        one_step, (x_hat_0, P_hat_0), (ys[1:], us[1:])
    )
    x_hats = jnp.concatenate([x_hat_0[None], x_hats_rest], axis=0)
    Ps = jnp.concatenate([P_hat_0[None], Ps_rest], axis=0)
    innovations = jnp.concatenate([innov_0[None], innovs_rest], axis=0)

    assert jnp.allclose(x_hats, batch.x_hat, atol=1e-12)
    assert jnp.allclose(Ps, batch.P, atol=1e-12)
    assert jnp.allclose(innovations, batch.innovations, atol=1e-12)


def test_ekf_step_can_skip_missing_measurement_under_jit():
    Q = jnp.array([[1e-3]])
    R = jnp.array([[1e-2]])
    y = jnp.array([10.0])
    u = jnp.array([0.0])
    x = jnp.array([1.0])
    P = jnp.array([[0.5]])

    def f(x, u):
        return jnp.array([0.8 * x[0] + u[0]])

    def h(x):
        return jnp.array([x[0] ** 2])

    def step(has_measurement):
        return cx.ekf_step(
            f,
            Q,
            R,
            y,
            u,
            x,
            P,
            observation=h,
            has_measurement=has_measurement,
        )

    x_skip, P_skip, innov_skip = jax.jit(step)(jnp.array(False))
    x_pred, P_pred = cx.ekf_predict(f, Q, x, P, u)

    assert jnp.allclose(x_skip, x_pred, atol=1e-12)
    assert jnp.allclose(P_skip, P_pred, atol=1e-12)
    assert jnp.allclose(innov_skip, jnp.zeros_like(y), atol=1e-12)


def test_ekf_update_supports_iterated_update():
    R = jnp.array([[1e-2]])
    y = jnp.array([4.0])
    x_pred = jnp.array([1.0])
    P_pred = jnp.array([[0.5]])

    def h(x):
        return jnp.array([x[0] ** 2])

    x_one, P_one, _ = cx.ekf_update(h, R, y, x_pred, P_pred, num_iter=1)
    x_iter, P_iter, innov_iter = cx.ekf_update(h, R, y, x_pred, P_pred, num_iter=3)

    assert x_one.shape == x_iter.shape == x_pred.shape
    assert P_one.shape == P_iter.shape == P_pred.shape
    assert innov_iter.shape == y.shape
    assert jnp.all(jnp.isfinite(x_iter))
    assert jnp.all(jnp.isfinite(P_iter))


def test_mhe_objective_zero_for_exact_nonlinear_window():
    def f(x, u, params):
        return jnp.array([params["a"] * x[0] + u[0] ** 2])

    def h(x, params):
        return jnp.array([params["c"] * x[0]])

    us = jnp.array([[0.0], [2.0], [1.0]])
    xs = cx.rollout(f, jnp.array([1.0]), us, {"a": 0.5, "c": 2.0})
    ys = jax.vmap(lambda x: h(x, {"a": 0.5, "c": 2.0}))(xs)

    cost = cx.mhe_objective(
        f,
        h,
        xs,
        us,
        ys,
        x_prior=xs[0],
        P_prior=jnp.eye(1),
        Q_noise=jnp.eye(1),
        R_noise=jnp.eye(1),
        params={"a": 0.5, "c": 2.0},
    )

    assert jnp.allclose(cost, 0.0, atol=1e-12)


def test_mhe_objective_matches_manual_linear_gaussian_cost():
    sys = _scalar_sys()
    xs = jnp.array([[0.0], [0.7], [0.9]])
    us = jnp.zeros((2, 1))
    ys = jnp.array([[0.1], [0.5], [1.2]])
    x_prior = jnp.array([0.2])
    P = jnp.array([[0.5]])
    Q = jnp.array([[0.25]])
    R = jnp.array([[2.0]])

    def f(x, u):
        return sys.A @ x + sys.B @ u

    def h(x):
        return sys.C @ x

    cost = cx.mhe_objective(f, h, xs, us, ys, x_prior, P, Q, R)
    arrival = (xs[0, 0] - x_prior[0]) ** 2 / P[0, 0]
    process = jnp.sum((xs[1:, 0] - 0.8 * xs[:-1, 0]) ** 2 / Q[0, 0])
    measurement = jnp.sum((ys[:, 0] - xs[:, 0]) ** 2 / R[0, 0])

    assert jnp.allclose(cost, arrival + process + measurement)


def test_mhe_objective_jit_grad_and_extra_cost():
    def f(x, u, params):
        return params["a"] * x + u

    def h(x, params):
        return x

    def extra_cost(xs, us, ys, params):
        del us, ys
        return params["soft"] * jnp.sum(jnp.maximum(xs - 2.0, 0.0) ** 2)

    us = jnp.ones((4, 1)) * 0.5
    xs = cx.rollout(f, jnp.array([1.0]), us, {"a": 0.8, "soft": 0.1})
    ys = xs + 0.1

    def objective(candidate_xs):
        return cx.mhe_objective(
            f,
            h,
            candidate_xs,
            us,
            ys,
            x_prior=jnp.array([1.0]),
            P_prior=jnp.eye(1),
            Q_noise=0.1 * jnp.eye(1),
            R_noise=0.2 * jnp.eye(1),
            params={"a": 0.8, "soft": 0.1},
            extra_cost=extra_cost,
        )

    cost = jax.jit(objective)(xs)
    grad = jax.grad(objective)(xs)

    assert cost.shape == ()
    assert jnp.isfinite(cost)
    assert grad.shape == xs.shape
    assert jnp.all(jnp.isfinite(grad))


def test_rts_shapes_and_terminal_state_matches_filter():
    """Pass zero Q_noise for deterministic process dynamics."""
    sys = _scalar_sys()
    Q = jnp.zeros((1, 1))
    ys = jnp.linspace(0.0, 1.0, 20)[:, None]
    kf = cx.kalman(
        sys,
        Q_noise=Q,
        R_noise=jnp.array([[1e-2]]),
        ys=ys,
    )
    smoothed = cx.rts(sys, kf, Q_noise=Q)

    assert smoothed.x_smooth.shape == kf.x_hat.shape
    assert smoothed.P_smooth.shape == kf.P.shape
    assert jnp.allclose(smoothed.x_smooth[-1], kf.x_hat[-1])
    assert jnp.allclose(smoothed.P_smooth[-1], kf.P[-1])


def test_rts_accepts_process_noise_covariance():
    """RTS should include the same process noise used by the forward filter."""
    sys = _scalar_sys()
    Q = jnp.array([[1e-3]])
    ys = jnp.linspace(0.0, 1.0, 25)[:, None]
    kf = cx.kalman(
        sys,
        Q_noise=Q,
        R_noise=jnp.array([[1e-2]]),
        ys=ys,
    )
    smoothed = cx.rts(sys, kf, Q_noise=Q)

    assert smoothed.x_smooth.shape == kf.x_hat.shape
    assert smoothed.P_smooth.shape == kf.P.shape
    assert jnp.all(jnp.isfinite(smoothed.x_smooth))
    assert jnp.all(jnp.isfinite(smoothed.P_smooth))
    assert jnp.all(smoothed.P_smooth[:, 0, 0] <= kf.P[:, 0, 0] + 1e-12)


def test_uks_matches_rts_for_linear_system():
    sys = _scalar_sys()
    nonlinear = cx.nonlinear_system(
        lambda t, x, u: sys.A @ x + sys.B @ u,
        observation=lambda t, x, u: sys.C @ x,
        dt=1.0,
        state_dim=1,
        input_dim=1,
        output_dim=1,
    )
    ys = jnp.linspace(0.0, 1.0, 25)[:, None]
    us = jnp.zeros((25, 1))
    Q = jnp.array([[1e-3]])
    R = jnp.array([[1e-2]])
    x0 = jnp.array([0.0])
    P0 = jnp.array([[1.0]])

    filtered_linear = cx.kalman(sys, Q, R, ys, x0=x0, P0=P0)
    filtered_nonlinear = cx.ukf(nonlinear, Q, R, ys, us, x0, P0)
    smoothed_linear = cx.rts(sys, filtered_linear, Q_noise=Q)
    smoothed_nonlinear = cx.uks(nonlinear, filtered_nonlinear, Q_noise=Q, us=us)

    assert jnp.allclose(
        smoothed_nonlinear.x_smooth, smoothed_linear.x_smooth, atol=1e-9
    )
    assert jnp.allclose(
        smoothed_nonlinear.P_smooth, smoothed_linear.P_smooth, atol=1e-9
    )


def test_uks_shapes_and_terminal_match_filter():
    sys = cx.nonlinear_system(
        lambda t, x, u: jnp.array([0.8 * x[0] + u[0]]),
        observation=lambda t, x, u: jnp.array([x[0] ** 2]),
        dt=1.0,
        state_dim=1,
        input_dim=1,
        output_dim=1,
    )
    ys = jnp.full((15, 1), 4.0)
    us = jnp.zeros((15, 1))
    Q = jnp.array([[1e-4]])
    R = jnp.array([[1e-3]])
    filtered = cx.ukf(
        sys, Q, R, ys, us, jnp.array([1.0]), jnp.array([[0.5]]), alpha=0.5
    )
    smoothed = cx.uks(sys, filtered, Q_noise=Q, us=us, alpha=0.5)

    assert smoothed.x_smooth.shape == filtered.x_hat.shape
    assert smoothed.P_smooth.shape == filtered.P.shape
    assert jnp.allclose(smoothed.x_smooth[-1], filtered.x_hat[-1], atol=1e-12)
    assert jnp.allclose(smoothed.P_smooth[-1], filtered.P[-1], atol=1e-12)
    assert jnp.all(jnp.isfinite(smoothed.x_smooth))
    assert jnp.all(jnp.isfinite(smoothed.P_smooth))


def test_ukf_matches_kalman_for_linear_system():
    sys = _scalar_sys()
    ys = jnp.linspace(0.0, 1.0, 20)[:, None]
    us = jnp.zeros((20, 1))
    Q = jnp.array([[1e-3]])
    R = jnp.array([[1e-2]])
    x0 = jnp.array([0.0])
    P0 = jnp.array([[1.0]])

    kf = cx.kalman(sys, Q, R, ys, x0=x0, P0=P0)

    def f(x, u):
        return sys.A @ x + sys.B @ u

    def h(x):
        return sys.C @ x

    ukf = cx.ukf(f, Q, R, ys, us, x0, P0, observation=h)

    assert isinstance(ukf, cx.UKFResult)
    assert jnp.allclose(ukf.x_hat, kf.x_hat, atol=1e-9)
    assert jnp.allclose(ukf.P, kf.P, atol=1e-9)
    assert jnp.allclose(ukf.innovations, kf.innovations, atol=1e-9)


def test_ukf_accepts_nonlinear_system_object():
    scalar_sys = _scalar_sys()
    sys = cx.nonlinear_system(
        lambda t, x, u: scalar_sys.A @ x + scalar_sys.B @ u,
        observation=lambda t, x, u: scalar_sys.C @ x,
        dt=1.0,
        state_dim=1,
        input_dim=1,
        output_dim=1,
    )
    ys = jnp.linspace(0.0, 1.0, 20)[:, None]
    us = jnp.zeros((20, 1))
    Q = jnp.array([[1e-3]])
    R = jnp.array([[1e-2]])
    x0 = jnp.array([0.0])
    P0 = jnp.array([[1.0]])

    result = cx.ukf(sys, Q, R, ys, us, x0, P0)
    reference = cx.kalman(scalar_sys, Q, R, ys, x0=x0, P0=P0)

    assert jnp.allclose(result.x_hat, reference.x_hat, atol=1e-9)
    assert jnp.allclose(result.P, reference.P, atol=1e-9)
    assert jnp.allclose(result.innovations, reference.innovations, atol=1e-9)


def test_ukf_handles_nonlinear_observation():
    sys = cx.nonlinear_system(
        lambda t, x, u: x,
        observation=lambda t, x, u: jnp.array([x[0] ** 2]),
        dt=1.0,
        state_dim=1,
        input_dim=1,
        output_dim=1,
    )

    true_x = 2.0
    ys = jnp.full((15, 1), true_x**2)
    us = jnp.zeros((15, 1))
    result = cx.ukf(
        sys,
        Q_noise=jnp.array([[1e-4]]),
        R_noise=jnp.array([[1e-3]]),
        ys=ys,
        us=us,
        x0=jnp.array([1.0]),
        P0=jnp.array([[0.5]]),
        alpha=0.5,
    )

    assert result.x_hat.shape == (15, 1)
    assert result.P.shape == (15, 1, 1)
    assert result.innovations.shape == (15, 1)
    assert jnp.all(jnp.isfinite(result.x_hat))
    assert abs(float(result.x_hat[-1, 0]) - true_x) < 5e-2


def test_ukf_exposes_prediction_and_likelihood_surface():
    sys = cx.nonlinear_system(
        lambda t, x, u: jnp.array([0.9 * x[0] + u[0]]),
        observation=lambda t, x, u: jnp.array([x[0] ** 2]),
        dt=1.0,
        state_dim=1,
        input_dim=1,
        output_dim=1,
    )
    ys = jnp.full((12, 1), 1.44)
    us = jnp.zeros((12, 1))

    result = cx.ukf(
        sys,
        Q_noise=jnp.array([[1e-3]]),
        R_noise=jnp.array([[1e-2]]),
        ys=ys,
        us=us,
        x0=jnp.array([1.0]),
        P0=jnp.array([[0.2]]),
        alpha=0.5,
    )

    assert result.predicted_measurements.shape == ys.shape
    assert result.innovation_covariances.shape == (12, 1, 1)
    assert result.log_likelihood_terms.shape == (12,)
    assert result.predicted_state_means.shape == result.x_hat.shape
    assert result.predicted_state_covariances.shape == result.P.shape
    assert result.transition_cross_covariances.shape == result.P.shape
    assert jnp.allclose(
        result.innovations,
        ys - result.predicted_measurements,
        atol=1e-12,
    )
    assert jnp.all(jnp.isfinite(result.log_likelihood_terms))
    assert jnp.all(result.innovation_covariances[:, 0, 0] > 0.0)


def test_uks_uses_stored_ukf_prediction_intermediates():
    sys = cx.nonlinear_system(
        lambda t, x, u: jnp.array([0.8 * x[0] + u[0]]),
        observation=lambda t, x, u: jnp.array([x[0]]),
        dt=1.0,
        state_dim=1,
        input_dim=1,
        output_dim=1,
    )
    ys = jnp.linspace(0.0, 1.0, 10)[:, None]
    us = jnp.zeros((10, 1))
    filtered = cx.ukf(
        sys,
        Q_noise=jnp.array([[1e-3]]),
        R_noise=jnp.array([[1e-2]]),
        ys=ys,
        us=us,
        x0=jnp.array([0.0]),
        P0=jnp.array([[1.0]]),
    )

    tampered = cx.UKFResult(
        x_hat=filtered.x_hat,
        P=filtered.P,
        innovations=filtered.innovations,
        predicted_measurements=filtered.predicted_measurements,
        innovation_covariances=filtered.innovation_covariances,
        log_likelihood_terms=filtered.log_likelihood_terms,
        predicted_state_means=filtered.predicted_state_means + 0.25,
        predicted_state_covariances=filtered.predicted_state_covariances,
        transition_cross_covariances=filtered.transition_cross_covariances,
    )

    smoothed_reference = cx.uks(sys, filtered, Q_noise=jnp.array([[1e-3]]), us=us)
    smoothed_tampered = cx.uks(sys, tampered, Q_noise=jnp.array([[1e-3]]), us=us)

    assert not jnp.allclose(
        smoothed_reference.x_smooth,
        smoothed_tampered.x_smooth,
    )


def test_sample_system_matches_known_zoh_scalar_decay():
    dt = 0.1
    sys = cx.nonlinear_system(
        lambda t, x, u: jnp.array([-x[0] + 2.0 * u[0]]),
        output=lambda t, x, u: x,
        state_dim=1,
        input_dim=1,
        output_dim=1,
    )
    sampled = cx.sample_system(sys, dt)

    x = jnp.array([1.5])
    u = jnp.array([0.25])
    x_next = sampled.dynamics(0.0, x, u)

    expected = jnp.exp(-dt) * x + 2.0 * (1.0 - jnp.exp(-dt)) * u
    assert sampled.dt.shape == ()
    assert jnp.allclose(x_next, expected, atol=1e-6, rtol=1e-6)


def test_foh_inputs_builds_endpoint_pairs():
    us = jnp.array([[0.0], [1.0], [3.0]])
    paired = cx.foh_inputs(us)

    assert paired.shape == (3, 2, 1)
    assert jnp.allclose(paired[0, :, 0], jnp.array([0.0, 1.0]))
    assert jnp.allclose(paired[1, :, 0], jnp.array([1.0, 3.0]))
    assert jnp.allclose(paired[2, :, 0], jnp.array([3.0, 3.0]))


def test_sample_system_foh_matches_integrator_average_input():
    dt = 0.2
    sys = cx.nonlinear_system(
        lambda t, x, u: jnp.array([u[0]]),
        output=lambda t, x, u: x,
        state_dim=1,
        input_dim=1,
        output_dim=1,
    )
    sampled = cx.sample_system(sys, dt, input_interpolation="foh")

    x_next = sampled.dynamics(
        0.0,
        jnp.array([1.0]),
        jnp.array([[2.0], [4.0]]),
    )

    assert jnp.allclose(x_next, jnp.array([1.0 + dt * 3.0]), atol=1e-6)


def test_ekf_accepts_sampled_continuous_system():
    sys_c = cx.nonlinear_system(
        lambda t, x, u: jnp.array([-0.4 * x[0] + u[0]]),
        output=lambda t, x, u: x,
        state_dim=1,
        input_dim=1,
        output_dim=1,
    )
    sys_d = cx.sample_system(sys_c, 0.1)
    ys = jnp.linspace(0.0, 0.5, 12)[:, None]
    us = jnp.zeros((12, 1))

    result = cx.ekf(
        sys_d,
        Q_noise=jnp.array([[1e-3]]),
        R_noise=jnp.array([[1e-2]]),
        ys=ys,
        us=us,
        x0=jnp.array([0.0]),
        P0=jnp.array([[1.0]]),
    )

    assert result.x_hat.shape == (12, 1)
    assert jnp.all(jnp.isfinite(result.x_hat))
    assert jnp.all(jnp.isfinite(result.P))


def test_ukf_accepts_sampled_phs_system():
    def H(x):
        return 0.5 * (x[0] ** 2 + x[1] ** 2)

    sys_c = cx.phs_system(H, state_dim=2, input_dim=1, output_dim=2)
    sys_d = cx.sample_system(sys_c, 0.05)
    ys = jnp.zeros((8, 2))
    us = jnp.zeros((8, 1))

    result = cx.ukf(
        sys_d,
        Q_noise=1e-4 * jnp.eye(2),
        R_noise=1e-3 * jnp.eye(2),
        ys=ys,
        us=us,
        x0=jnp.array([1.0, 0.0]),
        P0=0.1 * jnp.eye(2),
    )

    assert result.x_hat.shape == (8, 2)
    assert result.predicted_measurements.shape == (8, 2)
    assert jnp.all(jnp.isfinite(result.x_hat))


def test_innovation_diagnostics_compute_nis_and_conditioning():
    innovations = jnp.array([[1.0, 0.0], [0.0, 2.0]])
    innovation_covariances = jnp.array(
        [
            [[2.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 4.0]],
        ]
    )

    diag = cx.innovation_diagnostics(innovations, innovation_covariances)

    assert isinstance(diag, cx.InnovationDiagnostics)
    assert diag.nis.shape == (2,)
    assert jnp.allclose(diag.nis, jnp.array([0.5, 1.0]), atol=1e-12)
    assert jnp.allclose(diag.innovation_norms, jnp.array([1.0, 2.0]), atol=1e-12)
    assert float(diag.max_innovation_cov_condition_number) >= 1.0
    assert not bool(diag.nonfinite)


def test_likelihood_diagnostics_summarize_ukf_terms():
    sys = cx.nonlinear_system(
        lambda t, x, u: x,
        output=lambda t, x, u: x,
        dt=1.0,
        state_dim=1,
        input_dim=1,
        output_dim=1,
    )
    result = cx.ukf(
        sys,
        Q_noise=jnp.array([[1e-4]]),
        R_noise=jnp.array([[1e-2]]),
        ys=jnp.zeros((6, 1)),
        us=jnp.zeros((6, 1)),
        x0=jnp.array([0.0]),
        P0=jnp.array([[0.1]]),
    )

    diag = cx.likelihood_diagnostics(result.log_likelihood_terms)

    assert isinstance(diag, cx.LikelihoodDiagnostics)
    assert diag.log_likelihood_terms.shape == (6,)
    assert jnp.isfinite(diag.total_log_likelihood)
    assert jnp.isfinite(diag.mean_log_likelihood)
    assert not bool(diag.nonfinite)


def test_ukf_diagnostics_wrap_common_ukf_summary():
    sys = cx.nonlinear_system(
        lambda t, x, u: x,
        output=lambda t, x, u: x,
        dt=1.0,
        state_dim=1,
        input_dim=1,
        output_dim=1,
    )
    result = cx.ukf(
        sys,
        Q_noise=jnp.array([[1e-4]]),
        R_noise=jnp.array([[1e-2]]),
        ys=jnp.zeros((5, 1)),
        us=jnp.zeros((5, 1)),
        x0=jnp.array([0.0]),
        P0=jnp.array([[0.1]]),
    )

    innovation_diag, likelihood_diag = cx.ukf_diagnostics(result)

    assert isinstance(innovation_diag, cx.InnovationDiagnostics)
    assert isinstance(likelihood_diag, cx.LikelihoodDiagnostics)
    assert innovation_diag.nis.shape == (5,)
    assert likelihood_diag.log_likelihood_terms.shape == (5,)


def test_innovation_rms_accepts_kalman_and_ukf_results():
    kalman_result = cx.KalmanResult(
        x_hat=jnp.zeros((3, 1)),
        P=jnp.ones((3, 1, 1)),
        innovations=jnp.array([[1.0], [2.0], [2.0]]),
    )
    ukf_result = cx.UKFResult(
        x_hat=jnp.zeros((3, 1)),
        P=jnp.ones((3, 1, 1)),
        innovations=jnp.array([[1.0], [2.0], [2.0]]),
        predicted_measurements=jnp.zeros((3, 1)),
        innovation_covariances=jnp.tile(jnp.eye(1)[None], (3, 1, 1)),
        log_likelihood_terms=jnp.zeros(3),
        predicted_state_means=jnp.zeros((3, 1)),
        predicted_state_covariances=jnp.ones((3, 1, 1)),
        transition_cross_covariances=jnp.ones((3, 1, 1)),
    )

    expected = jnp.sqrt((1.0**2 + 2.0**2 + 2.0**2) / 3.0)
    assert jnp.allclose(cx.innovation_rms(kalman_result), expected)
    assert jnp.allclose(cx.innovation_rms(ukf_result), expected)


def test_mhe_shapes_and_convergence():
    """mhe() returns correct shapes and converges on a simple scalar problem."""
    sys = _scalar_sys()
    T = 8
    Q = 0.05 * jnp.eye(1)
    R = 0.2 * jnp.eye(1)
    ys = jnp.linspace(0.1, 0.5, T)[:, None]

    def f(x, u):
        return sys.A @ x

    def h(x):
        return sys.C @ x

    result = cx.mhe(
        f,
        h,
        xs_init=jnp.zeros((T, 1)),
        us=jnp.zeros((T - 1, 1)),
        ys=ys,
        x_prior=jnp.zeros(1),
        P_prior=jnp.eye(1),
        Q_noise=Q,
        R_noise=R,
    )

    assert result.xs.shape == (T, 1)
    assert result.x_hat.shape == (1,)
    assert result.cost.shape == ()
    assert jnp.isfinite(result.cost)
    assert bool(result.converged)


def test_mhe_matches_rts_smoother_linear_gaussian():
    """Full-window linear-Gaussian MHE MAP trajectory equals RTS smoother.

    For a linear model with Gaussian noise, the MHE objective is the negative
    log joint posterior of the state trajectory.  Its MAP equals the posterior
    mean, which is exactly what the RTS smoother returns.

    Both kalman()/rts() and mhe() use (x0, P0) as the prior directly on x_0,
    so the objectives are identical when they share the same x0, P0, Q, and R.

    # Octave: pkg load control; (analytical — no direct Octave equivalent)
    """
    sys = _scalar_sys()  # A=0.8, C=1.0
    T = 12
    Q = 0.05 * jnp.eye(1)
    R = 0.2 * jnp.eye(1)
    ys = jnp.array(
        [
            [0.10],
            [0.28],
            [0.45],
            [0.58],
            [0.65],
            [0.68],
            [0.63],
            [0.57],
            [0.50],
            [0.44],
            [0.39],
            [0.35],
        ]
    )

    # Kalman and MHE share the same prior on x_0
    x0 = jnp.zeros(1)
    P0 = jnp.eye(1)

    filtered = cx.kalman(sys, Q_noise=Q, R_noise=R, ys=ys, x0=x0, P0=P0)
    smoothed = cx.rts(sys, filtered, Q_noise=Q)

    def f(x, u):
        return sys.A @ x

    def h(x):
        return sys.C @ x

    result = cx.mhe(
        f,
        h,
        xs_init=filtered.x_hat,
        us=jnp.zeros((T - 1, 1)),
        ys=ys,
        x_prior=x0,
        P_prior=P0,
        Q_noise=Q,
        R_noise=R,
    )

    assert bool(result.converged)
    assert jnp.allclose(result.xs, smoothed.x_smooth, atol=1e-4)


def test_mhe_result_is_pytree():
    """MHEResult leaves are all JAX arrays (eqx.Module pytree contract)."""
    import jax

    sys = _scalar_sys()
    T = 5
    Q = 0.1 * jnp.eye(1)
    R = 0.1 * jnp.eye(1)

    result = cx.mhe(
        lambda x, u: sys.A @ x,
        lambda x: sys.C @ x,
        xs_init=jnp.zeros((T, 1)),
        us=jnp.zeros((T - 1, 1)),
        ys=jnp.zeros((T, 1)),
        x_prior=jnp.zeros(1),
        P_prior=jnp.eye(1),
        Q_noise=Q,
        R_noise=R,
    )

    leaves = jax.tree_util.tree_leaves(result)
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)


def test_soft_quadratic_penalty_supports_scalar_and_matrix_weights():
    residuals = jnp.array([[1.0, -2.0], [0.5, 1.0]])

    scalar_cost = cx.soft_quadratic_penalty(residuals, 0.5)
    matrix_cost = cx.soft_quadratic_penalty(residuals, jnp.diag(jnp.array([2.0, 4.0])))

    assert jnp.allclose(scalar_cost, 0.5 * jnp.sum(residuals**2))
    manual_matrix = 0.0
    for row in residuals:
        manual_matrix = manual_matrix + row @ jnp.linalg.solve(
            jnp.diag(jnp.array([2.0, 4.0])),
            row,
        )
    assert jnp.allclose(matrix_cost, manual_matrix)


def test_mhe_warm_start_shifts_window_and_repeats_terminal_state():
    xs = jnp.array([[0.0], [1.0], [2.0], [3.0]])

    warm = cx.mhe_warm_start(xs)

    assert warm.shape == xs.shape
    assert jnp.allclose(warm[:, 0], jnp.array([1.0, 2.0, 3.0, 3.0]))


def test_mhe_warm_start_can_propagate_terminal_guess():
    xs = jnp.array([[0.0], [1.0], [2.0]])

    warm = cx.mhe_warm_start(
        xs,
        transition=lambda x, u: x + u,
        terminal_input=jnp.array([0.5]),
    )

    assert jnp.allclose(warm[:, 0], jnp.array([1.0, 2.0, 2.5]))


def test_mhe_accepts_soft_penalty_helper_in_extra_cost():
    sys = _scalar_sys()
    ys = jnp.array([[0.0], [0.2], [0.4], [0.3]])

    def f(x, u):
        return sys.A @ x

    def h(x):
        return sys.C @ x

    def extra_cost(xs, us, ys):
        del us, ys
        violation = jnp.maximum(xs - 0.25, 0.0)
        return cx.soft_quadratic_penalty(violation, 2.0)

    result = cx.mhe(
        f,
        h,
        xs_init=jnp.zeros((4, 1)),
        us=jnp.zeros((3, 1)),
        ys=ys,
        x_prior=jnp.zeros(1),
        P_prior=jnp.eye(1),
        Q_noise=0.05 * jnp.eye(1),
        R_noise=0.2 * jnp.eye(1),
        extra_cost=extra_cost,
    )

    assert jnp.isfinite(result.cost)


def test_positive_parameterizations_are_positive_and_jittable():
    raw = jnp.array([-2.0, 0.0, 1.0])

    exp_vals = jax.jit(cx.positive_exp)(raw)
    softplus_vals = jax.jit(cx.positive_softplus)(raw)

    assert jnp.all(exp_vals > 0.0)
    assert jnp.all(softplus_vals > 0.0)


def test_spd_from_cholesky_raw_returns_symmetric_positive_definite_matrix():
    raw = jnp.array([[0.0, 3.0], [-1.0, 0.5]])

    spd = cx.spd_from_cholesky_raw(raw, diagonal="softplus", min_diagonal=1e-4)

    assert jnp.allclose(spd, spd.T, atol=1e-12)
    assert jnp.min(jnp.linalg.eigvalsh(spd)) > 0.0


def test_diagonal_spd_builds_positive_diagonal_matrix():
    spd = cx.diagonal_spd(jnp.array([-2.0, 0.0, 1.0]), parameterization="exp")

    assert spd.shape == (3, 3)
    assert jnp.allclose(spd, jnp.diag(jnp.diag(spd)))
    assert jnp.all(jnp.diag(spd) > 0.0)


def test_lower_triangular_overrides_diagonal():
    raw = jnp.array([[1.0, 2.0], [3.0, 4.0]])

    L = cx.lower_triangular(raw, diagonal=jnp.array([10.0, 20.0]))

    assert jnp.allclose(L, jnp.array([[10.0, 0.0], [3.0, 20.0]]))


def test_ukf_grad_through_noise_parameters_finite():
    ys = jnp.linspace(0.5, 1.5, 25)[:, None]
    us = jnp.zeros((25, 1))

    sys = cx.nonlinear_system(
        lambda t, x, u: jnp.array([0.9 * x[0] + 0.1]),
        observation=lambda t, x, u: jnp.array([x[0] ** 2]),
        dt=1.0,
        state_dim=1,
        input_dim=1,
        output_dim=1,
    )

    def loss(log_q, log_r):
        Q = jnp.exp(log_q)[None, None]
        R = jnp.exp(log_r)[None, None]
        result = cx.ukf(
            sys,
            Q,
            R,
            ys,
            us,
            x0=jnp.array([1.0]),
            P0=jnp.array([[0.5]]),
            alpha=0.5,
        )
        return jnp.sum(result.innovations**2) + jnp.sum(result.P)

    dlog_q, dlog_r = jax.grad(loss, argnums=(0, 1))(jnp.array(-5.0), jnp.array(-3.0))

    assert jnp.isfinite(dlog_q)
    assert jnp.isfinite(dlog_r)
