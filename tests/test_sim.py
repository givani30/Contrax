"""Tests for contrax.sim — lsim and simulate."""

import jax
import jax.numpy as jnp
import pytest

import contrax as cx

DT = 0.20039
A_CONT = jnp.array([[-2.2, -0.4, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
B_CONT = jnp.array([[0.25], [0.0], [0.0]])
C_CONT = jnp.array([[0.0, 0.0, 0.4]])
D_CONT = jnp.zeros((1, 1))


@pytest.fixture
def disc_sys():
    sys = cx.ss(A_CONT, B_CONT, C_CONT, D_CONT)
    return cx.c2d(sys, DT)


@pytest.fixture
def cont_sys():
    return cx.ss(A_CONT, B_CONT, C_CONT, D_CONT)


def test_lsim_shapes(disc_sys):
    T, n, m, p = 50, 3, 1, 1
    x0 = jnp.zeros(n)
    us = jnp.zeros((T, m))
    ts, xs, ys = cx.lsim(disc_sys, us, x0)
    assert ts.shape == (T,)
    assert xs.shape == (T + 1, n)
    assert ys.shape == (T, p)


def test_lsim_default_x0(disc_sys):
    """x0=None should default to zeros."""
    T = 10
    us = jnp.zeros((T, 1))
    ts, xs, ys = cx.lsim(disc_sys, us)
    assert jnp.allclose(xs, 0.0, atol=1e-12)


def test_lsim_time_vector(disc_sys):
    """ts must equal k * dt for k = 0, ..., T-1."""
    T = 20
    us = jnp.zeros((T, 1))
    ts, _, _ = cx.lsim(disc_sys, us, jnp.zeros(3))
    expected = jnp.arange(T, dtype=float) * DT
    assert jnp.allclose(ts, expected, atol=1e-10)


def test_lsim_zero_input_stays_at_zero(disc_sys):
    """From x0=0 with u=0, state must remain zero."""
    x0 = jnp.zeros(3)
    us = jnp.zeros((20, 1))
    _, xs, ys = cx.lsim(disc_sys, us, x0)
    assert jnp.allclose(xs, 0.0, atol=1e-12)
    assert jnp.allclose(ys, 0.0, atol=1e-12)


def test_lsim_initial_state_preserved(disc_sys):
    x0 = jnp.array([1.0, 2.0, 3.0])
    us = jnp.zeros((10, 1))
    _, xs, _ = cx.lsim(disc_sys, us, x0)
    assert jnp.allclose(xs[0], x0)


def test_lsim_one_step_manual(disc_sys):
    """First step must equal A @ x0 + B @ u0."""
    x0 = jnp.array([1.0, 0.0, 0.0])
    us = jnp.array([[0.5]])
    _, xs, _ = cx.lsim(disc_sys, us, x0)
    x1_expected = disc_sys.A @ x0 + disc_sys.B @ us[0]
    assert jnp.allclose(xs[1], x1_expected, atol=1e-12)


def test_lsim_finite(disc_sys):
    x0 = jnp.ones(3) * 0.01
    us = jax.random.normal(jax.random.PRNGKey(0), (100, 1)) * 0.1
    _, xs, ys = cx.lsim(disc_sys, us, x0)
    assert jnp.all(jnp.isfinite(xs))
    assert jnp.all(jnp.isfinite(ys))


def test_lsim_jit_compatible(disc_sys):
    """lsim should JIT-compile for fixed-shape input sequences."""
    us = jnp.zeros((15, 1))
    x0 = jnp.array([1.0, 0.0, 0.0])

    run = jax.jit(lambda sys, us, x0: cx.lsim(sys, us, x0))
    ts, xs, ys = run(disc_sys, us, x0)

    assert ts.shape == (15,)
    assert xs.shape == (16, 3)
    assert ys.shape == (15, 1)


def test_rollout_matches_manual_nonlinear_transition():
    def f(x, u, params):
        return jnp.array([params["a"] * x[0] + u[0] ** 2])

    x0 = jnp.array([1.0])
    us = jnp.array([[0.0], [2.0], [1.0]])
    xs = cx.rollout(f, x0, us, {"a": 0.5})

    expected = jnp.array([[1.0], [0.5], [4.25], [3.125]])
    assert jnp.allclose(xs, expected)


def test_rollout_without_params():
    def f(x, u):
        return x + u

    x0 = jnp.array([1.0, 2.0])
    us = jnp.array([[0.5, -0.5], [1.0, 1.0]])
    xs = cx.rollout(f, x0, us)

    expected = jnp.array([[1.0, 2.0], [1.5, 1.5], [2.5, 2.5]])
    assert jnp.allclose(xs, expected)


def test_rollout_jit_vmap_and_grad():
    def f(x, u, a):
        return a * x + u

    x0 = jnp.array([1.0])
    us = jnp.ones((4, 1)) * 0.5

    xs = jax.jit(cx.rollout, static_argnums=0)(f, x0, us, 0.8)
    assert xs.shape == (5, 1)
    assert jnp.all(jnp.isfinite(xs))

    candidate_us = jnp.stack([us, 2.0 * us, -us], axis=0)
    batched = jax.vmap(lambda u_seq: cx.rollout(f, x0, u_seq, 0.8))(candidate_us)
    assert batched.shape == (3, 5, 1)

    def terminal_loss(a):
        return cx.rollout(f, x0, us, a)[-1, 0] ** 2

    grad = jax.grad(terminal_loss)(jnp.array(0.8))
    assert jnp.isfinite(grad)


def test_simulate_lqr_closed_loop_converges(disc_sys):
    """LQR closed-loop simulation must drive state to zero."""
    Q = jnp.eye(3)
    R = jnp.array([[1.0]])
    result = cx.lqr(disc_sys, Q, R)
    K = result.K

    x0 = jnp.array([1.0, 0.0, 0.0])

    def policy(t, x):
        del t
        return -K @ x

    ts, xs, ys = cx.simulate(disc_sys, x0, policy, num_steps=200)
    assert jnp.all(jnp.isfinite(xs))
    assert jnp.linalg.norm(xs[-1]) < jnp.linalg.norm(xs[0]) * 0.01


def test_simulate_shapes(disc_sys):
    T = 30

    def policy(t, x):
        del t, x
        return jnp.zeros(1)

    ts, xs, ys = cx.simulate(disc_sys, jnp.zeros(3), policy, num_steps=T)
    assert ts.shape == (T,)
    assert xs.shape == (T + 1, 3)
    assert ys.shape == (T, 1)


def test_simulate_jit_compatible(disc_sys):
    """simulate should JIT-compile when T is fixed at trace time."""
    Q = jnp.eye(3)
    R = jnp.array([[1.0]])
    K = cx.lqr(disc_sys, Q, R).K

    def run(sys, x0, K):
        def policy(t, x):
            del t
            return -K @ x

        return cx.simulate(sys, x0, policy, num_steps=40)

    ts, xs, ys = jax.jit(run)(disc_sys, jnp.array([1.0, 0.0, 0.0]), K)

    assert ts.shape == (40,)
    assert xs.shape == (41, 3)
    assert ys.shape == (40, 1)
    assert jnp.all(jnp.isfinite(xs))


def test_simulate_continuous_shapes(cont_sys):
    ts, xs, ys = cx.simulate(
        cont_sys, jnp.zeros(3), lambda t, x: jnp.zeros(1), duration=2.0, dt=0.1
    )

    assert ts.shape == (21,)
    assert xs.shape == (21, 3)
    assert ys.shape == (21, 1)


def test_simulate_continuous_preserves_initial_state(cont_sys):
    x0 = jnp.array([1.0, -0.5, 0.25])
    _, xs, _ = cx.simulate(
        cont_sys, x0, lambda t, x: jnp.zeros(1), duration=1.0, dt=0.1
    )

    assert jnp.allclose(xs[0], x0, atol=1e-12)


def test_simulate_continuous_zero_input_matches_matrix_exponential():
    sys = cx.ss(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.zeros((1, 1)),
    )
    x0 = jnp.array([2.0])
    ts, xs, ys = cx.simulate(sys, x0, lambda t, x: jnp.zeros(1), duration=1.0, dt=0.1)
    expected = 2.0 * jnp.exp(-ts)

    assert jnp.allclose(xs[:, 0], expected, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(ys[:, 0], expected, atol=1e-5, rtol=1e-5)


def test_simulate_continuous_lqr_closed_loop_converges():
    sys = cx.ss(
        A=jnp.array([[0.0, 1.0], [0.0, 0.0]]),
        B=jnp.array([[0.0], [1.0]]),
        C=jnp.eye(2),
        D=jnp.zeros((2, 1)),
    )
    result = cx.lqr(sys, jnp.eye(2), jnp.ones((1, 1)))
    x0 = jnp.array([1.0, 0.0])

    _, xs, _ = cx.simulate(sys, x0, lambda t, x: -result.K @ x, duration=4.0, dt=0.05)

    assert jnp.all(jnp.isfinite(xs))
    assert jnp.linalg.norm(xs[-1]) < jnp.linalg.norm(xs[0]) * 0.1


def test_simulate_continuous_jit_compatible():
    sys = cx.ss(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.zeros((1, 1)),
    )

    def run(x0):
        return cx.simulate(sys, x0, lambda t, x: jnp.zeros(1), duration=1.0, dt=0.1)

    ts, xs, ys = jax.jit(run)(jnp.array([2.0]))

    assert ts.shape == (11,)
    assert xs.shape == (11, 1)
    assert ys.shape == (11, 1)
    assert jnp.all(jnp.isfinite(xs))


def test_simulate_continuous_nonlinear_system_matches_exponential_decay():
    sys = cx.nonlinear_system(
        lambda t, x, u: jnp.array([-x[0] + u[0]]),
        observation=lambda t, x, u: x,
        state_dim=1,
        input_dim=1,
        output_dim=1,
    )

    ts, xs, ys = cx.simulate(
        sys, jnp.array([2.0]), lambda t, x: jnp.zeros(1), duration=1.0, dt=0.1
    )
    expected = 2.0 * jnp.exp(-ts)

    assert jnp.allclose(xs[:, 0], expected, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(ys[:, 0], expected, atol=1e-5, rtol=1e-5)


def test_simulate_discrete_nonlinear_system_shapes():
    sys = cx.nonlinear_system(
        lambda t, x, u: jnp.array([0.8 * x[0] + u[0]]),
        observation=lambda t, x, u: x,
        dt=1.0,
        state_dim=1,
        input_dim=1,
        output_dim=1,
    )

    ts, xs, ys = cx.simulate(
        sys, jnp.array([1.0]), lambda t, x: jnp.zeros(1), num_steps=5
    )

    assert ts.shape == (5,)
    assert xs.shape == (6, 1)
    assert ys.shape == (5, 1)
    assert jnp.all(jnp.isfinite(xs))


def test_simulate_phs_system_conserves_energy_when_undamped_and_unforced():
    def H(x):
        return 0.5 * (x[0] ** 2 + x[1] ** 2)

    sys = cx.phs_system(H, state_dim=2, input_dim=1, output_dim=2)
    _, xs, _ = cx.simulate(
        sys, jnp.array([1.0, 0.0]), lambda t, x: jnp.zeros(1), duration=1.0, dt=0.01
    )
    energies = jax.vmap(H)(xs)

    assert jnp.all(jnp.isfinite(xs))
    assert jnp.max(jnp.abs(energies - energies[0])) < 5e-4


def test_schedule_phs_binds_time_varying_context_into_dynamics():
    def H(x):
        return 0.5 * (x[0] ** 2 + x[1] ** 2)

    base = cx.phs_system(H, state_dim=2, input_dim=1, output_dim=2)
    scheduled = cx.schedule_phs(
        base,
        context_fn=lambda t: t,
        R=lambda t, x, theta: jnp.array([[0.0, 0.0], [0.0, theta]], dtype=x.dtype),
    )

    x = jnp.array([0.0, 1.0])
    u = jnp.zeros(1)
    xdot_0 = scheduled.dynamics(0.0, x, u)
    xdot_2 = scheduled.dynamics(2.0, x, u)

    assert isinstance(scheduled, cx.NonlinearSystem)
    assert jnp.allclose(xdot_0, jnp.array([1.0, 0.0]), atol=1e-12)
    assert jnp.allclose(xdot_2, jnp.array([1.0, -2.0]), atol=1e-12)


def test_schedule_phs_simulate_works_with_scheduled_dissipation():
    def H(x):
        return 0.5 * (x[0] ** 2 + x[1] ** 2)

    base = cx.phs_system(H, state_dim=2, input_dim=1, output_dim=2)
    scheduled = cx.schedule_phs(
        base,
        context_fn=lambda t: 0.5 + 0.0 * jnp.asarray(t),
        R=lambda t, x, theta: jnp.array([[0.0, 0.0], [0.0, theta]], dtype=x.dtype),
    )

    _, xs, _ = cx.simulate(
        scheduled,
        jnp.array([1.0, 0.0]),
        lambda t, x: jnp.zeros(1),
        duration=1.0,
        dt=0.01,
    )
    energies = jax.vmap(H)(xs)

    assert jnp.all(jnp.isfinite(xs))
    assert energies[-1] < energies[0]


def test_partition_state_splits_structured_state_blocks():
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    q, p, z = cx.partition_state(x, (2, 1, 2))

    assert jnp.allclose(q, jnp.array([1.0, 2.0]))
    assert jnp.allclose(p, jnp.array([3.0]))
    assert jnp.allclose(z, jnp.array([4.0, 5.0]))


def test_block_observation_selects_partial_state_blocks():
    observe = cx.block_observation((2, 1, 2), (0, 2))
    y = observe(0.0, jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), jnp.zeros(1))

    assert jnp.allclose(y, jnp.array([1.0, 2.0, 4.0, 5.0]))


def test_block_matrix_builds_structured_input_map():
    G = cx.block_matrix(
        (2, 1),
        (1, 2),
        {
            (0, 0): jnp.array([[1.0], [2.0]]),
            (1, 1): jnp.array([[3.0, 4.0]]),
        },
    )

    expected = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 4.0],
        ]
    )
    assert jnp.allclose(G, expected)


def test_symmetrize_and_project_psd_clean_small_matrix_drift():
    M = jnp.array([[2.0, 3.0], [1.0, -1.0]])

    sym = cx.symmetrize_matrix(M)
    psd = cx.project_psd(M)

    assert jnp.allclose(sym, sym.T, atol=1e-12)
    assert jnp.allclose(psd, psd.T, atol=1e-12)
    assert jnp.min(jnp.linalg.eigvalsh(psd)) >= -1e-8


def test_phs_diagnostics_report_power_balance_and_structure_errors():
    def H(x):
        return 0.5 * (x[0] ** 2 + x[1] ** 2)

    sys = cx.phs_system(
        H,
        R=lambda x: jnp.array([[0.0, 0.2], [0.1, 0.5]], dtype=x.dtype),
        G=lambda x: jnp.array([[1.0], [0.0]], dtype=x.dtype),
        state_dim=2,
        input_dim=1,
        output_dim=2,
    )

    diag = cx.phs_diagnostics(sys, x=jnp.array([1.0, -2.0]), u=jnp.array([3.0]))

    assert isinstance(diag, cx.PHSStructureDiagnostics)
    assert float(diag.skew_symmetry_error) < 1e-12
    assert float(diag.dissipation_symmetry_error) > 0.0
    assert float(diag.dissipation_power) > 0.0
    assert abs(float(diag.power_balance_residual)) < 1e-10


def test_phs_diagnostics_jit_compatible():
    def H(x):
        return 0.5 * (x[0] ** 2 + x[1] ** 2)

    sys = cx.phs_system(
        H,
        R=lambda x: jnp.array([[0.2, 0.0], [0.0, 0.3]], dtype=x.dtype),
        G=lambda x: jnp.array([[1.0], [0.5]], dtype=x.dtype),
        state_dim=2,
        input_dim=1,
        output_dim=2,
    )

    run = jax.jit(lambda x, u: cx.phs_diagnostics(sys, x, u))
    diag = run(jnp.array([1.0, -2.0]), jnp.array([0.25]))

    assert isinstance(diag, cx.PHSStructureDiagnostics)
    assert jnp.isfinite(diag.storage_rate)
    assert jnp.isfinite(diag.dissipation_power)
    assert jnp.isfinite(diag.power_balance_residual)


def test_schedule_phs_respects_base_structure_map_when_unscheduled():
    def H(x):
        return 0.5 * (x[0] ** 2 + x[1] ** 2)

    base = cx.phs_system(
        H,
        J=lambda x: jnp.array([[0.0, 2.0], [-2.0, 0.0]], dtype=x.dtype),
        state_dim=2,
        input_dim=1,
        output_dim=2,
    )
    scheduled = cx.schedule_phs(base, context_fn=lambda t: t)

    xdot = scheduled.dynamics(0.0, jnp.array([1.0, 0.0]), jnp.zeros(1))

    assert jnp.allclose(xdot, jnp.array([0.0, -2.0]), atol=1e-12)


def test_schedule_phs_accepts_scheduled_structure_map():
    def H(x):
        return 0.5 * (x[0] ** 2 + x[1] ** 2)

    base = cx.phs_system(H, state_dim=2, input_dim=1, output_dim=2)
    scheduled = cx.schedule_phs(
        base,
        context_fn=lambda t: 1.0 + jnp.asarray(t),
        J=lambda t, x, theta: (
            theta * jnp.array([[0.0, 1.0], [-1.0, 0.0]], dtype=x.dtype)
        ),
    )

    xdot = scheduled.dynamics(2.0, jnp.array([1.0, 0.0]), jnp.zeros(1))

    assert jnp.allclose(xdot, jnp.array([0.0, -3.0]), atol=1e-12)


def test_block_observation_jit_compatible():
    observe = cx.block_observation((2, 1, 2), (1, 2))
    run = jax.jit(lambda x: observe(0.0, x, jnp.zeros(1)))

    y = run(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    assert jnp.allclose(y, jnp.array([3.0, 4.0, 5.0]))


def test_step_response_discrete_matches_lsim_with_unit_step(disc_sys):
    T = 12
    us = jnp.ones((T, 1))

    ts_step, _, ys_step = cx.step_response(disc_sys, num_steps=T)
    ts_lsim, _, ys_lsim = cx.lsim(disc_sys, us)

    assert jnp.allclose(ts_step, ts_lsim, atol=1e-12)
    assert jnp.allclose(ys_step, ys_lsim, atol=1e-12)


def test_impulse_response_discrete_matches_lsim_with_unit_pulse(disc_sys):
    T = 12
    us = jnp.zeros((T, 1)).at[0].set(1.0)

    ts_imp, _, ys_imp = cx.impulse_response(disc_sys, num_steps=T)
    ts_lsim, _, ys_lsim = cx.lsim(disc_sys, us)

    assert jnp.allclose(ts_imp, ts_lsim, atol=1e-12)
    assert jnp.allclose(ys_imp, ys_lsim, atol=1e-12)


def test_initial_response_discrete_matches_lsim_with_zero_input(disc_sys):
    x0 = jnp.array([1.0, -0.5, 0.25])
    T = 12
    us = jnp.zeros((T, 1))

    ts_init, _, ys_init = cx.initial_response(disc_sys, x0, num_steps=T)
    ts_lsim, _, ys_lsim = cx.lsim(disc_sys, us, x0=x0)

    assert jnp.allclose(ts_init, ts_lsim, atol=1e-12)
    assert jnp.allclose(ys_init, ys_lsim, atol=1e-12)


def test_step_response_continuous_matches_simulate_with_constant_input():
    sys = cx.ss(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[2.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.zeros((1, 1)),
    )

    ts_step, _, ys_step = cx.step_response(sys, duration=1.0, dt=0.1)
    ts_sim, _, ys_sim = cx.simulate(
        sys, jnp.zeros(1), lambda t, x: jnp.ones(1), duration=1.0, dt=0.1
    )

    assert jnp.allclose(ts_step, ts_sim, atol=1e-12)
    assert jnp.allclose(ys_step, ys_sim, atol=1e-8)


def test_impulse_response_continuous_matches_state_jump_convention():
    sys = cx.ss(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[2.0]]),
        C=jnp.array([[3.0]]),
        D=jnp.zeros((1, 1)),
    )

    ts, _, ys = cx.impulse_response(sys, duration=1.0, dt=0.1)
    expected = 6.0 * jnp.exp(-ts)

    assert jnp.allclose(ys[:, 0], expected, atol=1e-5, rtol=1e-5)


def test_initial_response_continuous_matches_zero_input_simulate():
    sys = cx.ss(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[2.0]]),
        C=jnp.array([[1.5]]),
        D=jnp.zeros((1, 1)),
    )
    x0 = jnp.array([2.0])

    ts_init, _, ys_init = cx.initial_response(sys, x0, duration=1.0, dt=0.1)
    ts_sim, _, ys_sim = cx.simulate(
        sys, x0, lambda t, x: jnp.zeros(1), duration=1.0, dt=0.1
    )

    assert jnp.allclose(ts_init, ts_sim, atol=1e-12)
    assert jnp.allclose(ys_init, ys_sim, atol=1e-8)


def test_response_functions_reject_bad_input_index():
    sys = cx.ss(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.zeros((1, 1)),
    )

    with pytest.raises(ValueError, match="input_index"):
        cx.step_response(sys, duration=1.0, input_index=1)


def test_lsim_grad_finite(disc_sys):
    """Gradients through lsim must be finite."""
    x0 = jnp.zeros(3)

    def cost(A_flat):
        dsys = cx.DiscLTI(
            A=A_flat.reshape(3, 3),
            B=disc_sys.B,
            C=disc_sys.C,
            D=disc_sys.D,
            dt=disc_sys.dt,
        )
        _, xs, _ = cx.lsim(dsys, jnp.zeros((10, 1)), x0)
        return jnp.sum(xs**2)

    g = jax.grad(cost)(disc_sys.A.ravel())
    assert jnp.all(jnp.isfinite(g))


def test_lqr_to_simulate_grad_finite(disc_sys):
    """Gradient through lqr -> closed-loop simulate must be finite."""
    x0 = jnp.array([1.0, 0.0, 0.0])

    def closed_loop_cost(Q_diag, log_R):
        Q = jnp.diag(Q_diag)
        R = jnp.exp(log_R)[None, None]
        K = cx.lqr(disc_sys, Q, R).K
        _, xs, _ = cx.simulate(disc_sys, x0, lambda t, x: -K @ x, num_steps=60)
        return jnp.sum(xs**2)

    dQ, dlog_R = jax.grad(closed_loop_cost, argnums=(0, 1))(jnp.ones(3), jnp.array(0.0))

    assert jnp.all(jnp.isfinite(dQ))
    assert jnp.isfinite(dlog_R)


def test_simulate_disc_rejects_duration_keyword(disc_sys):
    with pytest.raises(ValueError, match="num_steps"):
        cx.simulate(disc_sys, jnp.zeros(3), lambda t, x: jnp.zeros(1), duration=1.0)


def test_simulate_continuous_rejects_num_steps_keyword(cont_sys):
    with pytest.raises(ValueError, match="duration"):
        cx.simulate(cont_sys, jnp.zeros(3), lambda t, x: jnp.zeros(1), num_steps=10)


def test_vmap_linearize_c2d_lqr_over_operating_points():
    """Small gain-scheduling workflow: vmap(linearize -> c2d -> lqr)."""

    def pendulum(t, x, u):
        return jnp.array([x[1], -jnp.sin(x[0]) + u[0]])

    def sensor(x, u):
        return x

    def design(x0, u0):
        sys_c = cx.linearize_ss(pendulum, x0, u0, output=sensor)
        sys_d = cx.c2d(sys_c, dt=0.05)
        return cx.lqr(sys_d, jnp.eye(2), jnp.ones((1, 1))).K

    x0s = jnp.array([[0.0, 0.0], [0.1, 0.0], [-0.1, 0.0]])
    u0s = jnp.zeros((3, 1))
    Ks = jax.vmap(design)(x0s, u0s)

    assert Ks.shape == (3, 1, 2)
    assert jnp.all(jnp.isfinite(Ks))
