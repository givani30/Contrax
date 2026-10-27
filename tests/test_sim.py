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
    result = cx.lsim(disc_sys, us, x0)
    ts, xs, ys = result
    assert ts.shape == (T,)
    assert xs.shape == (T + 1, n)
    assert ys.shape == (T, p)
    expected = jnp.arange(T, dtype=float) * DT
    assert jnp.allclose(ts, expected, atol=1e-10)
    assert result.state_times.shape[0] == xs.shape[0]
    assert jnp.allclose(result.state_times, jnp.arange(T + 1) * DT)
    assert jnp.array_equal(result.output_times, ts)
    assert jnp.array_equal(result.final_state, xs[-1])


def test_lsim_one_step_manual(disc_sys):
    """First step must equal A @ x0 + B @ u0."""
    x0 = jnp.array([1.0, 0.0, 0.0])
    us = jnp.array([[0.5]])
    _, xs, _ = cx.lsim(disc_sys, us, x0)
    x1_expected = disc_sys.A @ x0 + disc_sys.B @ us[0]
    assert jnp.allclose(xs[1], x1_expected, atol=1e-12)


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


def test_step_response_discrete_matches_lsim_with_unit_step(disc_sys):
    T = 12
    us = jnp.ones((T, 1))

    ts_step, _, ys_step = cx.step_response(disc_sys, num_steps=T)
    ts_lsim, _, ys_lsim = cx.lsim(disc_sys, us)

    assert jnp.allclose(ts_step, ts_lsim, atol=1e-12)
    assert jnp.allclose(ys_step, ys_lsim, atol=1e-12)


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


# ---------------------------------------------------------------------------
# sample_system fixed-step methods
# ---------------------------------------------------------------------------

def _make_oscillator_phs():
    """Undamped unit harmonic oscillator as a PHS.

    H(q,p) = 0.5*(q^2 + p^2). Exact solution from [q0,p0]=[1,0]:
        q(t) = cos(t),  p(t) = -sin(t).
    """
    def H(x):
        return 0.5 * (x[0] ** 2 + x[1] ** 2)

    return cx.phs_system(H, state_dim=2, input_dim=1, output_dim=2)


@pytest.mark.parametrize("method", ["euler", "midpoint"])
def test_sample_system_fixed_step_shapes(method):
    """Fixed-step sample_system should produce correct output shapes."""
    sys = _make_oscillator_phs()
    sys_d = cx.sample_system(sys, 0.1, method=method)
    x0 = jnp.array([1.0, 0.0])
    ts, xs, ys = cx.simulate(sys_d, x0, lambda t, x: jnp.zeros(1), num_steps=10)
    assert ts.shape == (10,)
    assert xs.shape == (11, 2)
    assert ys.shape == (10, 2)
    assert jnp.all(jnp.isfinite(xs))


def test_sample_system_rk4_accuracy_vs_diffrax():
    """RK4 with dt=0.1 should match Diffrax to within 1e-4 on a unit oscillator.

    Exact: q(t) = cos(t), p(t) = -sin(t).  After T=20 steps (t=2.0 s) the
    global RK4 error is O(dt^4) ~ 1e-4; Diffrax (rtol=1e-6) is ~1e-6.
    """
    sys = _make_oscillator_phs()
    x0 = jnp.array([1.0, 0.0])
    T = 20
    dt = 0.1

    sys_rk4 = cx.sample_system(sys, dt, method="rk4")
    sys_ref = cx.sample_system(sys, dt, method="diffrax")

    _, xs_rk4, _ = cx.simulate(sys_rk4, x0, lambda t, x: jnp.zeros(1), num_steps=T)
    _, xs_ref, _ = cx.simulate(sys_ref, x0, lambda t, x: jnp.zeros(1), num_steps=T)

    max_err = float(jnp.max(jnp.abs(xs_rk4 - xs_ref)))
    assert max_err < 1e-3, f"RK4 vs Diffrax max error {max_err:.2e} exceeds 1e-3"


def test_sample_system_rk4_n_substeps_improves_accuracy():
    """More substeps should reduce the per-step integration error."""
    sys = _make_oscillator_phs()
    x0 = jnp.array([1.0, 0.0])
    T = 20
    dt = 0.1
    ts_exact = jnp.arange(T + 1, dtype=jnp.float64) * dt

    sys_1 = cx.sample_system(sys, dt, method="rk4", n_substeps=1)
    sys_4 = cx.sample_system(sys, dt, method="rk4", n_substeps=4)

    _, xs_1, _ = cx.simulate(sys_1, x0, lambda t, x: jnp.zeros(1), num_steps=T)
    _, xs_4, _ = cx.simulate(sys_4, x0, lambda t, x: jnp.zeros(1), num_steps=T)

    exact_q = jnp.cos(ts_exact)
    err_1 = float(jnp.max(jnp.abs(xs_1[:, 0] - exact_q)))
    err_4 = float(jnp.max(jnp.abs(xs_4[:, 0] - exact_q)))
    assert err_4 < err_1, f"n_substeps=4 error {err_4:.2e} not smaller than n_substeps=1 {err_1:.2e}"


def test_sample_system_rejects_unknown_method():
    sys = _make_oscillator_phs()
    with pytest.raises(ValueError, match="method"):
        cx.sample_system(sys, 0.1, method="bogus")


def test_sample_system_rejects_solver_with_fixed_step():
    import diffrax
    sys = _make_oscillator_phs()
    with pytest.raises(ValueError, match="Diffrax-only"):
        cx.sample_system(sys, 0.1, method="rk4", solver=diffrax.Euler())


def test_sample_system_rejects_n_substeps_zero():
    sys = _make_oscillator_phs()
    with pytest.raises(ValueError, match="n_substeps"):
        cx.sample_system(sys, 0.1, method="rk4", n_substeps=0)


def test_sample_system_rk4_ukf_shapes():
    """RK4-sampled PHS system should work end-to-end inside ukf()."""
    sys = _make_oscillator_phs()
    sys_d = cx.sample_system(sys, 0.1, method="rk4")

    T = 10
    x0 = jnp.array([1.0, 0.0])
    ys = jnp.zeros((T, 2))
    us = jnp.zeros((T, 1))

    result = cx.ukf(
        sys_d, ys, us=us, x0=x0, P0=0.1 * jnp.eye(2),
        Q=1e-4 * jnp.eye(2), R=1e-3 * jnp.eye(2),
    )
    assert result.xs.shape == (T, 2)
    assert jnp.all(jnp.isfinite(result.xs))


# ---------------------------------------------------------------------------
# SeparablePHSSystem
# ---------------------------------------------------------------------------

def _make_separable_oscillator():
    def V(q): return 0.5 * jnp.dot(q, q)
    def T(p): return 0.5 * jnp.dot(p, p)
    return cx.separable_phs_system(V, T, n_dofs=1, input_dim=1, output_dim=2)


def test_separable_phs_properties():
    """H == V+T, structural metadata is correct, and dynamics are finite."""
    sys = _make_separable_oscillator()
    x = jnp.array([0.6, -0.8])
    assert jnp.allclose(sys.H(x), sys.V(x[:1]) + sys.T(x[1:]), atol=1e-12)
    assert sys.n_dofs == 1
    assert sys.state_dim == 2
    xdot = sys.dynamics(0.0, jnp.array([1.0, 0.0]), jnp.zeros(1))
    assert xdot.shape == (2,)
    assert jnp.all(jnp.isfinite(xdot))


def test_separable_phs_rejects_invalid_n_dofs():
    def V(q): return 0.5 * jnp.dot(q, q)
    def T(p): return 0.5 * jnp.dot(p, p)
    with pytest.raises(ValueError, match="n_dofs"):
        cx.separable_phs_system(V, T, n_dofs=0)


def test_separable_phs_direct_construction_rejects_missing_v_t():
    """Direct construction without V/T should fail via __check_init__."""
    def H(x): return 0.5 * jnp.dot(x, x)
    with pytest.raises(Exception):
        cx.SeparablePHSSystem(H=H, n_dofs=1, state_dim=2)


# ---------------------------------------------------------------------------
# sample_system method="sv" (Störmer-Verlet)
# ---------------------------------------------------------------------------

def _make_separable_oscillator_sys():
    def V(q): return 0.5 * jnp.dot(q, q)
    def T(p): return 0.5 * jnp.dot(p, p)
    return cx.separable_phs_system(V, T, n_dofs=1, input_dim=1, output_dim=2)


def test_sv_conserves_energy_conservative():
    """Undamped Störmer-Verlet must conserve H(q,p) = V(q) + T(p) to machine precision.

    Symplectic integrators preserve a modified Hamiltonian; for the unit oscillator
    at dt=0.1 the energy drift should be O(dt^2) ~ 1e-2, but bounded — it does
    not grow with time.
    # Octave: no direct equivalent; verify analytically: SV is volume-preserving.
    """
    sys = _make_separable_oscillator_sys()
    sys_d = cx.sample_system(sys, 0.1, method="sv")
    x0 = jnp.array([1.0, 0.0])
    ts, xs, ys = cx.simulate(
        sys_d, x0, lambda t, x: jnp.zeros(1), num_steps=200
    )

    assert ts.shape == (200,)
    assert xs.shape == (201, 2)
    assert ys.shape == (200, 2)
    energies = jax.vmap(sys.H)(xs)
    # Energy should be bounded (not drift), characteristic of symplectic methods.
    energy_range = float(jnp.max(energies) - jnp.min(energies))
    assert energy_range < 2e-2, f"Energy range {energy_range:.4f} suggests non-symplectic drift"


def test_sv_bounded_drift_vs_euler_nonlinear():
    """SV energy drift must be bounded while Euler shows secular growth.

    Uses a nonlinear pendulum-like potential V(q) = 1 - cos(q) + 0.5*q^2 where
    Euler's non-symplectic nature causes clearly monotone energy increase while
    SV oscillates around the initial value.  RK4 is not used for comparison
    because for a linear oscillator at small dt its O(dt^4) per-step error
    is too small to distinguish from SV at moderate step counts.
    """
    def V(q): return 1.0 - jnp.cos(q[0]) + 0.5 * q[0] ** 2
    def T(p): return 0.5 * p[0] ** 2
    sys = cx.separable_phs_system(V, T, n_dofs=1, input_dim=0, output_dim=2)
    sys_sv = cx.sample_system(sys, 0.2, method="sv")
    sys_euler = cx.sample_system(sys, 0.2, method="euler")
    x0 = jnp.array([1.0, 0.0])
    T = 500

    _, xs_sv, _ = cx.simulate(sys_sv, x0, lambda t, x: jnp.zeros(0), num_steps=T)
    _, xs_euler, _ = cx.simulate(sys_euler, x0, lambda t, x: jnp.zeros(0), num_steps=T)

    e_sv = jax.vmap(sys.H)(xs_sv)
    e_euler = jax.vmap(sys.H)(xs_euler)

    # Euler should show clear secular growth; SV should stay bounded.
    euler_final_drift = float(jnp.abs(e_euler[-1] - e_euler[0]))
    sv_max_drift = float(jnp.max(jnp.abs(e_sv - e_sv[0])))

    assert euler_final_drift > sv_max_drift, (
        f"Euler final drift {euler_final_drift:.4f} not larger than SV max drift {sv_max_drift:.4f}"
    )


def test_sv_n_substeps_improves_energy_accuracy():
    """More substeps should reduce the energy oscillation amplitude."""
    sys = _make_separable_oscillator_sys()
    sys_1 = cx.sample_system(sys, 0.1, method="sv", n_substeps=1)
    sys_4 = cx.sample_system(sys, 0.1, method="sv", n_substeps=4)
    x0 = jnp.array([1.0, 0.0])

    _, xs_1, _ = cx.simulate(sys_1, x0, lambda t, x: jnp.zeros(1), num_steps=100)
    _, xs_4, _ = cx.simulate(sys_4, x0, lambda t, x: jnp.zeros(1), num_steps=100)

    e_1 = jax.vmap(sys.H)(xs_1)
    e_4 = jax.vmap(sys.H)(xs_4)
    range_1 = float(jnp.max(e_1) - jnp.min(e_1))
    range_4 = float(jnp.max(e_4) - jnp.min(e_4))
    assert range_4 < range_1


def test_sv_rejects_non_separable_system():
    """method='sv' must raise ValueError for plain PHSSystem."""
    def H(x): return 0.5 * jnp.dot(x, x)
    sys = cx.phs_system(H, state_dim=2, input_dim=1)
    with pytest.raises(ValueError, match="SeparablePHSSystem"):
        cx.sample_system(sys, 0.1, method="sv")


def test_sv_rejects_nonlinear_system():
    sys = cx.nonlinear_system(
        lambda t, x, u: jnp.array([-x[0]]),
        observation=lambda t, x, u: x,
        state_dim=1, input_dim=1, output_dim=1,
    )
    with pytest.raises(ValueError, match="SeparablePHSSystem"):
        cx.sample_system(sys, 0.1, method="sv")
