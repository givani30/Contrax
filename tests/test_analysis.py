"""Tests for contrax.analysis helpers.

Octave verification snippets:
  pkg load control
  sysc = ss(-2, 1, 3, 0.5);
  evalfr(sysc, 1i)      % -> 1.7 - 0.6i
  dcgain(sysc)          % -> 2.0

  sysd = ss(0.8, 2, 1.5, 0.25, 0.1);
  evalfr(sysd, exp(0.3i))
  dcgain(sysd)          % -> 15.25
"""

import jax
import jax.numpy as jnp
import numpy as np

import contrax as cx
from contrax.analysis import ctrb_gramian, obsv_gramian


def _double_integrator():
    return cx.ss(
        A=jnp.array([[0.0, 1.0], [0.0, 0.0]]),
        B=jnp.array([[0.0], [1.0]]),
        C=jnp.array([[1.0, 0.0]]),
        D=jnp.zeros((1, 1)),
    )


def test_ctrb_double_integrator_matches_manual_matrix():
    """Controllability matrix is [B, AB] for the double integrator."""
    sys = _double_integrator()
    expected = jnp.array([[0.0, 1.0], [1.0, 0.0]])

    assert jnp.allclose(cx.ctrb(sys), expected, atol=1e-12)


def test_obsv_double_integrator_matches_manual_matrix():
    """Observability matrix is [C; CA] for the double integrator."""
    sys = _double_integrator()
    expected = jnp.array([[1.0, 0.0], [0.0, 1.0]])

    assert jnp.allclose(cx.obsv(sys), expected, atol=1e-12)


def test_gramian_helpers_are_exported_at_top_level():
    assert cx.ctrb_gramian is ctrb_gramian
    assert cx.obsv_gramian is obsv_gramian


def test_ctrb_and_obsv_work_for_discrete_systems():
    sys = cx.dss(
        A=jnp.array([[0.9, 0.1], [0.0, 0.8]]),
        B=jnp.array([[0.0], [1.0]]),
        C=jnp.array([[1.0, 0.0]]),
        D=jnp.zeros((1, 1)),
        dt=0.1,
    )

    assert cx.ctrb(sys).shape == (2, 2)
    assert cx.obsv(sys).shape == (2, 2)
    assert jnp.linalg.matrix_rank(cx.ctrb(sys)) == 2
    assert jnp.linalg.matrix_rank(cx.obsv(sys)) == 2


def test_poles_matches_state_matrix_eigenvalues():
    sys = _double_integrator()
    expected = jnp.linalg.eigvals(sys.A)

    assert jnp.allclose(jnp.sort_complex(cx.poles(sys)), jnp.sort_complex(expected))


def test_evalfr_continuous_matches_scalar_analytic_value():
    sys = cx.ss(
        A=jnp.array([[-2.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[3.0]]),
        D=jnp.array([[0.5]]),
    )

    actual = cx.evalfr(sys, 1j)
    expected = jnp.array([[1.7 - 0.6j]])

    assert jnp.allclose(actual, expected, atol=1e-12)


def test_evalfr_discrete_matches_scalar_analytic_value():
    sys = cx.dss(
        A=jnp.array([[0.8]]),
        B=jnp.array([[2.0]]),
        C=jnp.array([[1.5]]),
        D=jnp.array([[0.25]]),
        dt=0.1,
    )
    z = jnp.exp(0.3j)

    actual = cx.evalfr(sys, z)
    expected = jnp.array([[1.5 * 2.0 / (z - 0.8) + 0.25]])

    assert jnp.allclose(actual, expected, atol=1e-12)


def test_freqresp_continuous_matches_batched_evalfr():
    sys = cx.ss(
        A=jnp.array([[-2.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[3.0]]),
        D=jnp.array([[0.5]]),
    )
    omega = jnp.array([0.0, 1.0, 2.0])

    actual = cx.freqresp(sys, omega)
    expected = jnp.stack([cx.evalfr(sys, 1j * w) for w in omega])

    assert jnp.allclose(actual, expected, atol=1e-12)


def test_freqresp_discrete_evaluates_on_unit_circle():
    sys = cx.dss(
        A=jnp.array([[0.8]]),
        B=jnp.array([[2.0]]),
        C=jnp.array([[1.5]]),
        D=jnp.array([[0.25]]),
        dt=0.1,
    )
    omega = jnp.array([0.0, 1.0, 2.0])

    actual = cx.freqresp(sys, omega)
    expected = jnp.stack([cx.evalfr(sys, jnp.exp(1j * w * sys.dt)) for w in omega])

    assert jnp.allclose(actual, expected, atol=1e-12)


def test_dcgain_continuous_matches_scalar_analytic_value():
    sys = cx.ss(
        A=jnp.array([[-2.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[3.0]]),
        D=jnp.array([[0.5]]),
    )

    assert jnp.allclose(cx.dcgain(sys), jnp.array([[2.0]]), atol=1e-12)


def test_dcgain_discrete_matches_scalar_analytic_value():
    sys = cx.dss(
        A=jnp.array([[0.8]]),
        B=jnp.array([[2.0]]),
        C=jnp.array([[1.5]]),
        D=jnp.array([[0.25]]),
        dt=0.1,
    )

    assert jnp.allclose(cx.dcgain(sys), jnp.array([[15.25]]), atol=1e-12)


def test_frequency_analysis_is_jittable():
    sys = cx.ss(
        A=jnp.array([[-2.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[3.0]]),
        D=jnp.array([[0.5]]),
    )
    omega = jnp.array([0.0, 1.0, 2.0])

    jitted = jax.jit(cx.freqresp)(sys, omega)

    assert jitted.shape == (3, 1, 1)
    assert np.iscomplexobj(np.asarray(jitted))


def test_scalar_controllability_gramian_matches_analytic_value():
    """For A=-1, B=2: W_c(t)=∫0^t 4 exp(-2s) ds = 2(1-exp(-2t))."""
    sys = cx.ss(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[2.0]]),
        C=jnp.array([[1.0]]),
        D=jnp.zeros((1, 1)),
    )
    t = 5.0
    W = ctrb_gramian(sys, t=t)
    expected = 2.0 * (1.0 - jnp.exp(-2.0 * t))

    assert jnp.allclose(W[0, 0], expected, atol=1e-3, rtol=1e-3)


def test_scalar_observability_gramian_matches_analytic_value():
    """For A=-1, C=3: W_o(t)=∫0^t 9 exp(-2s) ds = 4.5(1-exp(-2t))."""
    sys = cx.ss(
        A=jnp.array([[-1.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[3.0]]),
        D=jnp.zeros((1, 1)),
    )
    t = 5.0
    W = obsv_gramian(sys, t=t)
    expected = 4.5 * (1.0 - jnp.exp(-2.0 * t))

    assert jnp.allclose(W[0, 0], expected, atol=2e-3, rtol=1e-3)


def test_controllability_gramian_satisfies_finite_horizon_lyapunov_identity():
    sys = cx.ss(
        A=jnp.array([[-0.7, 0.2], [0.0, -0.4]]),
        B=jnp.array([[1.0], [0.3]]),
        C=jnp.eye(2),
        D=jnp.zeros((2, 1)),
    )
    t = 1.7
    W = ctrb_gramian(sys, t=t)
    phi = jax.scipy.linalg.expm(sys.A * t)
    residual = (
        sys.A @ W + W @ sys.A.T + sys.B @ sys.B.T - phi @ (sys.B @ sys.B.T) @ phi.T
    )

    assert jnp.max(jnp.abs(residual)) < 1e-10


def test_analysis_gradients_are_finite():
    def loss(a):
        sys = cx.ss(
            A=jnp.array([[-jnp.exp(a)]]),
            B=jnp.array([[1.0]]),
            C=jnp.array([[1.0]]),
            D=jnp.zeros((1, 1)),
        )
        return ctrb_gramian(sys, t=2.0)[0, 0]

    grad = jax.grad(loss)(jnp.array(0.0))

    assert jnp.isfinite(grad)


# ── Lyapunov solvers ───────────────────────────────────────────────────────


def test_lyap_stable_diagonal():
    # Octave: pkg load control; lyap(diag([-1,-2]), eye(2))
    # -> X = [[0.5, 0], [0, 0.25]]
    A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
    Q = jnp.eye(2)
    X = cx.lyap(A, Q)
    residual = A @ X + X @ A.T + Q
    assert jnp.max(jnp.abs(residual)) < 1e-12
    assert jnp.allclose(X, jnp.diag(jnp.array([0.5, 0.25])), atol=1e-10)


def test_lyap_symmetric_solution():
    A = jnp.array([[-3.0, 1.0], [-1.0, -2.0]])
    Q = jnp.array([[2.0, 0.5], [0.5, 1.0]])
    X = cx.lyap(A, Q)
    assert jnp.max(jnp.abs(X - X.T)) < 1e-12
    assert jnp.max(jnp.abs(A @ X + X @ A.T + Q)) < 1e-10


def test_dlyap_stable_diagonal():
    # Octave: pkg load control; dlyap(diag([0.5, 0.3]), eye(2))
    # -> X = [[4/3, 0], [0, 100/91]]
    A = jnp.array([[0.5, 0.0], [0.0, 0.3]])
    Q = jnp.eye(2)
    X = cx.dlyap(A, Q)
    residual = A @ X @ A.T - X + Q
    assert jnp.max(jnp.abs(residual)) < 1e-12
    expected = jnp.diag(jnp.array([1.0 / (1 - 0.25), 1.0 / (1 - 0.09)]))
    assert jnp.allclose(X, expected, atol=1e-10)


def test_dlyap_symmetric_solution():
    A = jnp.array([[0.6, 0.1], [-0.1, 0.5]])
    Q = jnp.array([[1.0, 0.2], [0.2, 0.5]])
    X = cx.dlyap(A, Q)
    assert jnp.max(jnp.abs(X - X.T)) < 1e-12
    assert jnp.max(jnp.abs(A @ X @ A.T - X + Q)) < 1e-10


# ── Transmission zeros ─────────────────────────────────────────────────────


def test_zeros_siso_d0_no_zeros():
    # Double integrator: G(s) = 1/s^2, no finite zeros
    sys = _double_integrator()
    z = cx.zeros(sys)
    assert z.shape[0] == 0


def test_zeros_siso_d0_single_zero():
    # G(s) = (s+3)/((s+1)(s+2)), zero at s=-3
    # Octave: pkg load control; tzero(ss([0,1;-2,-3],[1;0],[1,0],0))
    # -> -3
    sys = cx.ss(
        A=jnp.array([[0.0, 1.0], [-2.0, -3.0]]),
        B=jnp.array([[1.0], [0.0]]),
        C=jnp.array([[1.0, 0.0]]),
        D=jnp.zeros((1, 1)),
    )
    z = cx.zeros(sys)
    assert z.shape[0] == 1
    assert abs(complex(z[0]) - (-3.0)) < 1e-8


def test_zeros_invertible_d():
    # G(s) = 3/(s+2) + 2 = (2s+7)/(s+2), zero at s=-3.5
    # Octave: pkg load control; tzero(ss(-2,1,3,2)) -> -3.5
    sys = cx.ss(
        A=jnp.array([[-2.0]]),
        B=jnp.array([[1.0]]),
        C=jnp.array([[3.0]]),
        D=jnp.array([[2.0]]),
    )
    z = cx.zeros(sys)
    assert z.shape[0] == 1
    assert abs(complex(z[0]) - (-3.5)) < 1e-8


def test_phs_to_ss_returns_lti():
    import jax; jax.config.update("jax_enable_x64", True)

    def H(x):
        return 0.5 * jnp.dot(x, x)

    phs = cx.phs_system(H, state_dim=2, input_dim=1)
    lti = cx.phs_to_ss(phs, jnp.zeros(2), jnp.zeros(1))
    assert isinstance(lti, cx.ContLTI)
    assert lti.A.shape == (2, 2)
    assert lti.B.shape == (2, 1)
