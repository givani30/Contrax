"""Tests for contrax.estimation — Kalman, EKF, UKF, and RTS smoother."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import contrax as cx


def _scalar_sys():
    return cx.dss(
        A=jnp.array([[0.8]]),
        B=jnp.zeros((1, 1)),
        C=jnp.array([[1.0]]),
        D=jnp.zeros((1, 1)),
        dt=1.0,
    )


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
        ys,
        x0=jnp.array([0.0]),
        P0=jnp.array([[1.0]]),
        Q=jnp.array([[0.0001]]),
        R=jnp.array([[0.001]]),
    )

    assert abs(float(result.xs[-1, 0]) - 1.0) < 1e-3
    assert abs(float(result.innovations[-1, 0])) < 1e-3


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
    batch = cx.kalman(sys, ys, Q=Q, R=R, x0=x0, P0=P0)

    # Step 0: update x0 directly (no predict — x0 is prior on x_0)
    x_hat_0, P_hat_0, innov_0 = cx.kalman_update(sys, x0, P0, ys[0], R=R)

    # Steps 1+: kalman_step from the previous posterior
    def one_step(carry, y):
        x, P = carry
        x_new, P_new, innov = cx.kalman_step(sys, x, P, y, Q=Q, R=R)
        return (x_new, P_new), (x_new, P_new, innov)

    _, (x_hats_rest, Ps_rest, innovs_rest) = jax.lax.scan(
        one_step, (x_hat_0, P_hat_0), ys[1:]
    )
    x_hats = jnp.concatenate([x_hat_0[None], x_hats_rest], axis=0)
    Ps = jnp.concatenate([P_hat_0[None], Ps_rest], axis=0)
    innovations = jnp.concatenate([innov_0[None], innovs_rest], axis=0)

    assert jnp.allclose(x_hats, batch.xs, atol=1e-12)
    assert jnp.allclose(Ps, batch.Ps, atol=1e-12)
    assert jnp.allclose(innovations, batch.innovations, atol=1e-12)


def test_kalman_step_can_skip_missing_measurement_under_jit():
    sys = _scalar_sys()
    Q = jnp.array([[1e-3]])
    R = jnp.array([[1e-2]])
    y = jnp.array([10.0])
    x = jnp.array([1.0])
    P = jnp.array([[0.5]])

    def step(has_measurement):
        return cx.kalman_step(sys, x, P, y, Q=Q, R=R, has_measurement=has_measurement)

    x_skip, P_skip, innov_skip = jax.jit(step)(jnp.array(False))
    x_pred, P_pred = cx.kalman_predict(sys, x, P, Q=Q)

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


def test_ekf_matches_kalman_for_linear_system():
    sys = _scalar_sys()
    ys = jnp.linspace(0.0, 1.0, 20)[:, None]
    us = jnp.zeros((20, 1))
    Q = jnp.array([[1e-3]])
    R = jnp.array([[1e-2]])
    x0 = jnp.array([0.0])
    P0 = jnp.array([[1.0]])

    kf = cx.kalman(sys, ys, Q=Q, R=R, x0=x0, P0=P0)

    def f(t, x, u):
        return sys.A @ x + sys.B @ u

    def h(x):
        return sys.C @ x

    ekf = cx.ekf(f, ys, Q=Q, R=R, us=us, x0=x0, P0=P0, observation=h)

    assert jnp.allclose(ekf.xs, kf.xs, atol=1e-12)
    assert jnp.allclose(ekf.Ps, kf.Ps, atol=1e-12)
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

    result = cx.ekf(sys, ys, Q=Q, R=R, us=us, x0=x0, P0=P0)
    reference = cx.kalman(scalar_sys, ys, Q=Q, R=R, x0=x0, P0=P0)

    assert jnp.allclose(result.xs, reference.xs, atol=1e-12)
    assert jnp.allclose(result.Ps, reference.Ps, atol=1e-12)
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

    batch = cx.ekf(nonlinear, ys, Q=Q, R=R, us=us, x0=x0, P0=P0)

    # Step 0: update x0 directly (x0 is prior on x_0)
    def f(t, x, u):
        return sys.A @ x + sys.B @ u

    def h(x):
        return sys.C @ x

    x_hat_0, P_hat_0, innov_0 = cx.ekf_update(h, x0, P0, ys[0], R=R)

    # Steps 1+: ekf_step from the previous posterior
    def one_step(carry, inputs):
        x, P = carry
        y, u = inputs
        x_new, P_new, innov = cx.ekf_step(f, x, P, y, u=u, Q=Q, R=R, observation=h)
        return (x_new, P_new), (x_new, P_new, innov)

    _, (x_hats_rest, Ps_rest, innovs_rest) = jax.lax.scan(
        one_step, (x_hat_0, P_hat_0), (ys[1:], us[1:])
    )
    x_hats = jnp.concatenate([x_hat_0[None], x_hats_rest], axis=0)
    Ps = jnp.concatenate([P_hat_0[None], Ps_rest], axis=0)
    innovations = jnp.concatenate([innov_0[None], innovs_rest], axis=0)

    assert jnp.allclose(x_hats, batch.xs, atol=1e-12)
    assert jnp.allclose(Ps, batch.Ps, atol=1e-12)
    assert jnp.allclose(innovations, batch.innovations, atol=1e-12)


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
        xs,
        us,
        ys,
        x_prior=xs[0],
        P_prior=jnp.eye(1),
        Q_noise=jnp.eye(1),
        R_noise=jnp.eye(1),
        params={"a": 0.5, "c": 2.0},
        observation=h,
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

    cost = cx.mhe_objective(f, xs, us, ys, x_prior, P, Q, R, observation=h)
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
            candidate_xs,
            us,
            ys,
            x_prior=jnp.array([1.0]),
            P_prior=jnp.eye(1),
            Q_noise=0.1 * jnp.eye(1),
            R_noise=0.2 * jnp.eye(1),
            params={"a": 0.8, "soft": 0.1},
            extra_cost=extra_cost,
            observation=h,
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
    kf = cx.kalman(sys, ys, Q=Q, R=jnp.array([[0.01]]))
    smoothed = cx.rts(sys, kf, Q=Q)

    assert smoothed.xs.shape == kf.xs.shape
    assert smoothed.Ps.shape == kf.Ps.shape
    assert jnp.allclose(smoothed.xs[-1], kf.xs[-1])
    assert jnp.allclose(smoothed.Ps[-1], kf.Ps[-1])


def test_rts_validation_preserves_singular_psd_and_failure_status():
    """The PD fast check must retain the spectral PSD and failure contracts."""
    from contrax._filter_checks import covariance_valid

    matrices = jnp.array(
        [
            [[1.0, 0.2], [0.2, 2.0]],
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, -1e-3]],
            [[1.0, 0.1], [0.0, 1.0]],
            [[jnp.nan, 0.0], [0.0, 1.0]],
        ]
    )
    spectral = jax.jit(covariance_valid)(matrices)
    fast = jax.jit(lambda p: covariance_valid(p, cholesky=True))(matrices)
    assert jnp.array_equal(fast, spectral)
    assert jnp.array_equal(fast, jnp.array([True, True, False, False, False]))
    sys = _scalar_sys()
    filtered = cx.kalman(sys, jnp.zeros((3, 1)), Q=jnp.zeros((1, 1)), R=jnp.eye(1))
    invalid = eqx.tree_at(lambda r: r.Ps, filtered, -jnp.ones_like(filtered.Ps))
    smooth = jax.jit(lambda r: cx.rts(sys, r, Q=jnp.eye(1)))
    assert smooth(filtered).successful
    assert not smooth(invalid).successful
    failed = eqx.tree_at(lambda r: r.solver_status, filtered, jnp.asarray(2))
    assert smooth(failed).status == 2
    singular = cx.types.RTSResult(xs=jnp.zeros((3, 1)), Ps=jnp.zeros((3, 1, 1)))
    assert jax.jit(lambda r: r.successful)(singular)


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

    filtered_linear = cx.kalman(sys, ys, Q=Q, R=R, x0=x0, P0=P0)
    filtered_nonlinear = cx.ukf(nonlinear, ys, Q=Q, R=R, us=us, x0=x0, P0=P0)
    smoothed_linear = cx.rts(sys, filtered_linear, Q=Q)
    smoothed_nonlinear = cx.uks(filtered_nonlinear)

    assert jnp.allclose(smoothed_nonlinear.xs, smoothed_linear.xs, atol=1e-9)
    assert jnp.allclose(smoothed_nonlinear.Ps, smoothed_linear.Ps, atol=1e-9)


def test_ukf_matches_kalman_for_linear_system():
    sys = _scalar_sys()
    ys = jnp.linspace(0.0, 1.0, 20)[:, None]
    us = jnp.zeros((20, 1))
    Q = jnp.array([[1e-3]])
    R = jnp.array([[1e-2]])
    x0 = jnp.array([0.0])
    P0 = jnp.array([[1.0]])

    kf = cx.kalman(sys, ys, Q=Q, R=R, x0=x0, P0=P0)

    def f(t, x, u):
        return sys.A @ x + sys.B @ u

    def h(x):
        return sys.C @ x

    ukf = cx.ukf(f, ys, Q=Q, R=R, us=us, x0=x0, P0=P0, observation=h)

    assert isinstance(ukf, cx.UKFResult)
    assert jnp.allclose(ukf.xs, kf.xs, atol=1e-9)
    assert jnp.allclose(ukf.Ps, kf.Ps, atol=1e-9)
    assert jnp.allclose(ukf.innovations, kf.innovations, atol=1e-9)


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
        ys,
        us=us,
        x0=jnp.array([1.0]),
        P0=jnp.array([[0.5]]),
        Q=jnp.array([[0.0001]]),
        R=jnp.array([[0.001]]),
        solver=cx.solvers.UKF(alpha=0.5),
    )

    assert result.xs.shape == (15, 1)
    assert result.Ps.shape == (15, 1, 1)
    assert result.innovations.shape == (15, 1)
    assert result.predicted_measurements.shape == ys.shape
    assert result.innovation_covariances.shape == (15, 1, 1)
    assert result.log_likelihood_terms.shape == (15,)
    assert result.predicted_state_means.shape == result.xs.shape
    assert result.predicted_state_covariances.shape == result.Ps.shape
    assert result.transition_cross_covariances.shape == result.Ps.shape
    assert jnp.allclose(result.innovations, ys - result.predicted_measurements)
    assert jnp.all(jnp.isfinite(result.log_likelihood_terms))
    assert jnp.all(result.innovation_covariances[:, 0, 0] > 0.0)
    assert jnp.all(jnp.isfinite(result.xs))
    assert abs(float(result.xs[-1, 0]) - true_x) < 5e-2


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
        ys,
        us=us,
        x0=jnp.array([0.0]),
        P0=jnp.array([[1.0]]),
        Q=jnp.array([[0.001]]),
        R=jnp.array([[0.01]]),
    )

    assert result.xs.shape == (12, 1)
    assert jnp.all(jnp.isfinite(result.xs))
    assert jnp.all(jnp.isfinite(result.Ps))


def test_ukf_accepts_sampled_phs_system():
    def H(x):
        return 0.5 * (x[0] ** 2 + x[1] ** 2)

    sys_c = cx.phs_system(H, state_dim=2, input_dim=1, output_dim=2)
    sys_d = cx.sample_system(sys_c, 0.05)
    ys = jnp.zeros((8, 2))
    us = jnp.zeros((8, 1))

    result = cx.ukf(
        sys_d,
        ys,
        us=us,
        x0=jnp.array([1.0, 0.0]),
        P0=0.1 * jnp.eye(2),
        Q=0.0001 * jnp.eye(2),
        R=0.001 * jnp.eye(2),
    )

    assert result.xs.shape == (8, 2)
    assert result.predicted_measurements.shape == (8, 2)
    assert jnp.all(jnp.isfinite(result.xs))


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
        jnp.zeros((5, 1)),
        us=jnp.zeros((5, 1)),
        x0=jnp.array([0.0]),
        P0=jnp.array([[0.1]]),
        Q=jnp.array([[0.0001]]),
        R=jnp.array([[0.01]]),
    )

    innovation_diag, likelihood_diag = cx.ukf_diagnostics(result)

    assert isinstance(innovation_diag, cx.InnovationDiagnostics)
    assert isinstance(likelihood_diag, cx.LikelihoodDiagnostics)
    assert innovation_diag.nis.shape == (5,)
    assert likelihood_diag.log_likelihood_terms.shape == (5,)
    assert jax.jit(cx.innovation_rms)(result) == 0
    masked = eqx.tree_at(
        lambda r: (r.innovations, r.has_measurement),
        result,
        (
            jnp.array([[3.0], [jnp.nan], [jnp.nan], [jnp.nan], [jnp.nan]]),
            jnp.array([True, False, False, False, False]),
        ),
    )
    assert cx.innovation_rms(masked) == 3
    summary = cx.likelihood_diagnostics(
        jnp.array([-1.0, jnp.nan]), has_measurement=jnp.array([True, False])
    )
    assert summary.total_log_likelihood == summary.mean_log_likelihood == -1
    assert not summary.nonfinite


def test_smoother_diagnostics_healthy_rts():
    """RTS smoother on a stable system should never increase uncertainty."""
    sys = _scalar_sys()
    Q = jnp.array([[1e-3]])
    R = jnp.array([[1e-2]])
    ys = jnp.linspace(0.0, 1.0, 20)[:, None]

    filtered = cx.kalman(sys, ys, Q=Q, R=R)
    smoothed = cx.rts(sys, filtered, Q=Q)
    diag = cx.smoother_diagnostics(smoothed, filtered)

    assert isinstance(diag, cx.SmootherDiagnostics)
    assert diag.covariance_reduction.shape == (20,)
    assert diag.state_corrections.shape == (20,)
    assert diag.smoothed_min_eigenvalue.shape == (20,)
    assert float(diag.min_covariance_reduction) >= -1e-10
    assert jnp.all(diag.smoothed_min_eigenvalue > 0.0)
    assert not bool(diag.nonfinite)


def test_smoother_diagnostics_healthy_uks():
    """UKS smoother should also satisfy the covariance-reduction guarantee."""
    sys = cx.nonlinear_system(
        lambda t, x, u: jnp.array([0.9 * x[0] + u[0]]),
        observation=lambda t, x, u: jnp.array([x[0]]),
        dt=1.0,
        state_dim=1,
        input_dim=1,
        output_dim=1,
    )
    Q = jnp.array([[1e-3]])
    R = jnp.array([[1e-2]])
    ys = jnp.linspace(0.0, 1.0, 15)[:, None]
    us = jnp.zeros((15, 1))

    filtered = cx.ukf(
        sys, ys, Q=Q, R=R, us=us, x0=jnp.array([0.0]), P0=jnp.array([[1.0]])
    )
    smoothed = cx.uks(filtered)
    diag = cx.smoother_diagnostics(smoothed, filtered)

    assert float(diag.min_covariance_reduction) >= -1e-9
    assert not bool(diag.nonfinite)


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

    filtered = cx.kalman(sys, ys, x0=x0, P0=P0, Q=Q, R=R)
    smoothed = cx.rts(sys, filtered, Q=Q)

    def f(x, u):
        return sys.A @ x

    def h(x):
        return sys.C @ x

    result = cx.mhe(
        f,
        observation=h,
        xs_init=filtered.xs,
        us=jnp.zeros((T - 1, 1)),
        ys=ys,
        x_prior=x0,
        P_prior=P0,
        Q_noise=Q,
        R_noise=R,
    )

    assert bool(result.solver_converged)
    assert jnp.allclose(result.xs, smoothed.xs, atol=1e-4)


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
        observation=h,
        xs_init=jnp.zeros((4, 1)),
        us=jnp.zeros((3, 1)),
        ys=ys,
        x_prior=jnp.zeros(1),
        P_prior=jnp.eye(1),
        Q_noise=0.05 * jnp.eye(1),
        R_noise=0.2 * jnp.eye(1),
        extra_cost=extra_cost,
    )

    assert jnp.isfinite(result.final_cost)


def test_positive_parameterizations_are_positive_and_jittable():
    raw = jnp.array([-2.0, 0.0, 1.0])

    exp_vals = jax.jit(cx.positive_exp)(raw)
    softplus_vals = jax.jit(cx.positive_softplus)(raw)

    assert jnp.all(exp_vals > 0.0)
    assert jnp.all(softplus_vals > 0.0)
    raw = jnp.array([[0.0, 3.0], [-1.0, 0.5]])
    spd = cx.spd_from_cholesky_raw(raw, diagonal="softplus", min_diagonal=1e-4)
    assert jnp.allclose(spd, spd.T, atol=1e-12)
    assert jnp.min(jnp.linalg.eigvalsh(spd)) > 0.0
    spd = cx.diagonal_spd(jnp.array([-2.0, 0.0, 1.0]), parameterization="exp")
    assert spd.shape == (3, 3)
    assert jnp.allclose(spd, jnp.diag(jnp.diag(spd)))
    assert jnp.all(jnp.diag(spd) > 0.0)
    L = cx.lower_triangular(
        jnp.array([[1.0, 2.0], [3.0, 4.0]]), diagonal=jnp.array([10.0, 20.0])
    )
    assert jnp.allclose(L, jnp.array([[10.0, 0.0], [3.0, 20.0]]))
    integer_spd = cx.spd_from_cholesky_raw(jnp.zeros((2, 2), dtype=int))
    expected = (jax.nn.softplus(0.0) + 1e-6) ** 2 * jnp.eye(2)
    assert jnp.allclose(integer_spd, expected)
    assert cx.positive_exp(jnp.zeros(2, dtype=jnp.float32)).dtype == jnp.float32
    assert (
        cx.lower_triangular(jnp.zeros((1, 1), dtype=int), diagonal=jnp.array([0.5]))[
            0, 0
        ]
        == 0.5
    )
    for invalid_floor in (-1.0, jnp.nan, jnp.inf):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            cx.positive_softplus(0, min_value=invalid_floor)
    with pytest.raises(ValueError, match="shape"):
        cx.diagonal_spd(jnp.zeros((2, 2)))
    with pytest.raises(ValueError, match="real"):
        cx.spd_from_cholesky_raw(jnp.eye(2, dtype=complex))
    with pytest.raises(Exception, match="finite and nonnegative"):
        jax.jit(lambda floor: cx.positive_exp(0.0, min_value=floor))(-1.0)

    def loss(value):
        return jnp.sum(cx.spd_from_cholesky_raw(value))

    assert jnp.allclose(jax.jacfwd(loss)(raw), jax.grad(loss)(raw))


def test_ukf_likelihood_and_noise_gradients_match_joint_gaussian():
    """Observed y0/y2 have covariance L diag(1,q,q) L.T + r I.

    This joint-distribution oracle uses no filtering recursion. Missing y1
    contributes neither likelihood nor a diagnostic failure.
    """
    sys = cx.dss([[0.9]], [[1.0]], [[1.0]], [[0.0]], 1.0)
    ys = jnp.array([[0.5], [jnp.nan], [1.5]])
    flags = jnp.array([True, False, True])

    def filter_result(theta):
        q, r = jnp.exp(theta)
        return cx.ukf(
            sys,
            ys,
            Q=q * jnp.eye(1),
            R=r * jnp.eye(1),
            has_measurement=flags,
        )

    def loss(theta):
        return jnp.sum(filter_result(theta).log_likelihood_terms)

    def oracle(theta):
        q, r = jnp.exp(theta)
        L = jnp.array([[1.0, 0.0, 0.0], [0.81, 0.9, 1.0]])
        covariance = (L * jnp.array([1.0, q, q])) @ L.T + r * jnp.eye(2)
        return jax.scipy.stats.multivariate_normal.logpdf(
            ys[jnp.array([0, 2]), 0], jnp.zeros(2), covariance
        )

    theta = jnp.array([-3.0, -2.0])
    assert jnp.allclose(jax.jit(loss)(theta), oracle(theta), atol=1e-10)
    assert jnp.allclose(
        jax.jit(jax.grad(loss))(theta), jax.grad(oracle)(theta), atol=1e-9
    )
    tangent = jnp.array([0.3, -0.2])
    assert jnp.allclose(
        jax.jvp(loss, (theta,), (tangent,))[1],
        jax.jvp(oracle, (theta,), (tangent,))[1],
        atol=1e-9,
    )
    result = filter_result(theta)
    innovations, likelihood = cx.ukf_diagnostics(result)
    assert not innovations.nonfinite and not likelihood.nonfinite
    assert result.log_likelihood_terms[1] == 0
    assert jnp.all(result.predicted_measurements[1] == 0)
    assert jnp.allclose(likelihood.mean_log_likelihood, oracle(theta) / 2)
