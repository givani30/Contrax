"""Tests for contrax.control.

Octave ground truth (octave --no-gui):
  pkg load control
  A=[-2.2 -0.4 0;1 0 0;0 1 0]; B=[0.25;0;0]; C=[0 0 0.4]; D=0;
  dsys=c2d(ss(A,B,C,D),0.20039,'zoh');

  % Pole placement
  zeta=0.6901; omega=1.0; h=0.20039;
  p1=exp((-zeta*omega+1i*omega*sqrt(1-zeta^2))*h); p2=conj(p1); p3=0.2*abs(p1);
  L=place(dsys.A,dsys.B,[p1;p2;p3])
  % L = [15.0789  27.1937  17.7725]

  % LQR Q=I R=1
  [K,S,e]=dlqr(dsys.A,dsys.B,eye(3),1)
  % K = [1.4020  3.2270  0.9646]
  % closed-loop |eig| = [0.6681  0.9466  0.9466]
"""

import jax
import jax.numpy as jnp
import numpy as np
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


def _desired_poles(omega=1.0, zeta=0.6901, h=DT):
    s = -zeta * omega + 1j * omega * np.sqrt(1 - zeta**2)
    p1 = np.exp(s * h)
    p2 = np.conj(p1)
    p3 = 0.2 * abs(p1)
    return np.array([p1, p2, p3])


def _dare_residual(A, B, Q, R, S):
    return (
        S
        - Q
        - A.T @ S @ A
        + A.T @ S @ B @ jnp.linalg.solve(R + B.T @ S @ B, B.T @ S @ A)
    )


def _care_residual(A, B, Q, R, S):
    return A.T @ S + S @ A - S @ B @ jnp.linalg.solve(R, B.T @ S) + Q


# -- pole placement ------------------------------------------------------------


def test_place_gain_matches_octave(disc_sys):
    """L ~ [15.07, 27.19, 17.77]. Octave: place(Phi,Gamma,[p1,p2,p3]).

    Design: zeta=0.6901, omega=1 rad/s, third pole at 0.2*|dominant|.
    (SC42095 Assignment 21, Section 5 uses the same design objectives but a
    different MATLAB-internal state-space realization, giving different numbers.)
    """
    poles = _desired_poles()
    L = cx.place(disc_sys, poles)
    L_expected = jnp.array([[15.0789, 27.1937, 17.7725]])
    assert jnp.allclose(L, L_expected, atol=0.1), (
        f"Pole placement mismatch.\n  got:      {L}\n  expected: {L_expected}"
    )


def test_place_achieves_desired_poles(disc_sys):
    """Closed-loop eigenvalue magnitudes after place() must match targets."""
    poles = _desired_poles()
    L = cx.place(disc_sys, poles)
    cl_sys = cx.feedback(disc_sys, L)
    actual = np.sort(np.abs(np.linalg.eigvals(np.array(cl_sys.A))))
    expected = np.sort(np.abs(poles))
    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_state_feedback_matches_feedback_alias(disc_sys):
    L = cx.place(disc_sys, _desired_poles())
    via_explicit_name = cx.state_feedback(disc_sys, L)
    via_alias = cx.feedback(disc_sys, L)

    assert jnp.allclose(via_explicit_name.A, via_alias.A, atol=1e-12)
    assert jnp.allclose(via_explicit_name.B, via_alias.B, atol=1e-12)
    assert jnp.allclose(via_explicit_name.C, via_alias.C, atol=1e-12)
    assert jnp.allclose(via_explicit_name.D, via_alias.D, atol=1e-12)


def test_place_shape(disc_sys):
    L = cx.place(disc_sys, _desired_poles())
    assert L.shape == (1, 3)


def test_place_mimo_achieves_requested_real_poles():
    sys = cx.dss(
        jnp.array([[1.0, 0.0], [0.0, 0.9]]),
        jnp.array([[1.0, 0.0], [0.0, 1.0]]),
        jnp.eye(2),
        jnp.zeros((2, 2)),
        dt=0.1,
    )
    poles = jnp.array([0.5, 0.6])

    L = cx.place(sys, poles)
    cl = cx.feedback(sys, L)
    actual = np.sort_complex(np.linalg.eigvals(np.array(cl.A)))

    assert L.shape == (2, 2)
    np.testing.assert_allclose(actual, np.sort_complex(np.array(poles)), atol=1e-8)


def test_place_continuous_double_integrator_achieves_desired_poles():
    sys = cx.ss(
        jnp.array([[0.0, 1.0], [0.0, 0.0]]),
        jnp.array([[0.0], [1.0]]),
        jnp.eye(2),
        jnp.zeros((2, 1)),
    )
    poles = jnp.array([-1.0, -2.0])

    L = cx.place(sys, poles)
    cl = cx.feedback(sys, L)
    actual = np.sort_complex(np.linalg.eigvals(np.array(cl.A)))

    np.testing.assert_allclose(actual, np.sort_complex(np.array(poles)), atol=1e-8)


def test_place_continuous_mimo_complex_pair_uses_yt_path():
    sys = cx.ss(
        jnp.array([[0.0, 1.0, 0.0], [-1.0, -0.02, 0.1], [0.0, 0.0, -0.5]]),
        jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        jnp.eye(3),
        jnp.zeros((3, 2)),
    )
    poles = jnp.asarray(np.array([-0.2 + 1.2j, -0.2 - 1.2j, -1.5]))

    L = cx.place(sys, poles)
    cl = cx.feedback(sys, L)
    actual = np.sort_complex(np.linalg.eigvals(np.array(cl.A)))

    np.testing.assert_allclose(actual, np.sort_complex(np.array(poles)), atol=1e-8)


def test_place_clustered_real_poles_matches_targets():
    sys = cx.dss(
        jnp.array([[0.92, 0.1, 0.0], [0.0, 0.95, 0.08], [0.0, 0.0, 0.97]]),
        jnp.array([[1.0, 0.0], [0.0, 1.0], [0.2, 0.1]]),
        jnp.eye(3),
        jnp.zeros((3, 2)),
        dt=0.1,
    )
    poles = jnp.array([0.7, 0.701, 0.702])

    L = cx.place(sys, poles)
    cl = cx.feedback(sys, L)
    actual = np.sort_complex(np.linalg.eigvals(np.array(cl.A)))

    np.testing.assert_allclose(actual, np.sort_complex(np.array(poles)), atol=1e-8)


# -- feedback ------------------------------------------------------------------


def test_feedback_closed_loop_stable(disc_sys):
    L = cx.place(disc_sys, _desired_poles())
    cl = cx.feedback(disc_sys, L)
    eigs = jnp.abs(jnp.linalg.eigvals(cl.A))
    assert jnp.all(eigs < 1.0), f"Closed-loop unstable: {eigs}"


def test_feedback_preserves_bcd(disc_sys):
    """feedback() must not modify B, C, D, dt."""
    L = jnp.zeros((1, 3))
    cl = cx.feedback(disc_sys, L)
    assert jnp.allclose(cl.B, disc_sys.B)
    assert jnp.allclose(cl.C, disc_sys.C)
    assert jnp.allclose(cl.D, disc_sys.D)
    assert jnp.allclose(cl.dt, disc_sys.dt)


def test_feedback_continuous_preserves_bcd():
    sys = cx.ss(
        A_CONT,
        B_CONT,
        C_CONT,
        D_CONT,
    )
    L = jnp.zeros((1, 3))
    cl = cx.feedback(sys, L)

    assert isinstance(cl, cx.ContLTI)
    assert jnp.allclose(cl.B, sys.B)
    assert jnp.allclose(cl.C, sys.C)
    assert jnp.allclose(cl.D, sys.D)


def test_feedback_continuous_lqr_closed_loop_stable():
    sys = cx.ss(A_CONT, B_CONT, C_CONT, D_CONT)
    result = cx.lqr(sys, jnp.eye(3), jnp.array([[1.0]]))
    cl = cx.feedback(sys, result.K)
    eigs = jnp.real(jnp.linalg.eigvals(cl.A))

    assert jnp.all(eigs < 0.0), f"Closed-loop unstable: {eigs}"


# -- LQR -----------------------------------------------------------------------


def test_lqr_gain_matches_octave(disc_sys):
    """K ~ [1.4020, 3.2270, 0.9646]. Octave: dlqr(Phi, Gamma, eye(3), 1)."""
    Q = jnp.eye(3)
    R = jnp.array([[1.0]])
    result = cx.lqr(disc_sys, Q, R)
    K_expected = jnp.array([[1.4020, 3.2270, 0.9646]])
    assert jnp.allclose(result.K, K_expected, atol=0.05), (
        f"LQR gain mismatch.\n  got:      {result.K}\n  expected: {K_expected}"
    )


def test_lqr_stabilizes(disc_sys):
    Q = jnp.eye(3)
    R = jnp.array([[1.0]])
    result = cx.lqr(disc_sys, Q, R)
    cl = cx.feedback(disc_sys, result.K)
    eigs = jnp.abs(jnp.linalg.eigvals(cl.A))
    assert jnp.all(eigs < 1.0), f"LQR unstable: {eigs}"


def test_lqr_result_fields(disc_sys):
    result = cx.lqr(disc_sys, jnp.eye(3), jnp.array([[1.0]]))
    assert result.K.shape == (1, 3)
    assert result.S.shape == (3, 3)
    assert result.poles.shape == (3,)
    assert result.residual_norm.shape == ()
    assert result.residual_norm < 1e-8


def test_lqr_riccati_residual(disc_sys):
    """S must satisfy the DARE to within numerical tolerance."""
    Q = jnp.eye(3)
    R = jnp.array([[1.0]])
    result = cx.lqr(disc_sys, Q, R)
    residual = _dare_residual(disc_sys.A, disc_sys.B, Q, R, result.S)
    assert jnp.max(jnp.abs(residual)) < 1e-4
    assert jnp.allclose(result.residual_norm, jnp.max(jnp.abs(residual)))


def test_dare_double_integrator_residual_and_stability():
    A = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    B = jnp.array([[0.5], [1.0]])
    Q = jnp.eye(2)
    R = jnp.array([[1.0]])

    result = cx.dare(A, B, Q, R)

    residual = _dare_residual(A, B, Q, R, result.S)
    eigs = jnp.abs(jnp.linalg.eigvals(A - B @ result.K))
    assert jnp.max(jnp.abs(residual)) < 1e-10
    assert jnp.all(eigs < 1.0)


def test_dare_jit_matches_eager(disc_sys):
    Q = jnp.eye(3)
    R = jnp.array([[1.0]])

    eager = cx.dare(disc_sys.A, disc_sys.B, Q, R)
    jitted = jax.jit(cx.dare)(disc_sys.A, disc_sys.B, Q, R)

    assert jnp.allclose(jitted.K, eager.K, atol=1e-10)
    assert jnp.allclose(jitted.S, eager.S, atol=1e-10)
    assert jnp.allclose(jitted.residual_norm, eager.residual_norm, atol=1e-12)
    assert jnp.all(jnp.abs(jitted.poles) < 1.0)


def test_lqr_output_weighting_stabilizes(disc_sys):
    """LQR with output weighting Q = C^T C * 1e5 must stabilize."""
    Q = disc_sys.C.T @ disc_sys.C * 1e5
    R = jnp.array([[1.0]])
    result = cx.lqr(disc_sys, Q, R)
    cl = cx.feedback(disc_sys, result.K)
    eigs = jnp.abs(jnp.linalg.eigvals(cl.A))
    assert jnp.all(eigs < 1.0)


def test_augment_integrator_discrete_matches_manual_dt_scaled_model():
    sys = cx.dss(
        jnp.array([[1.0, 0.1], [0.0, 1.0]]),
        jnp.array([[0.0], [0.1]]),
        jnp.array([[1.0, 0.0]]),
        jnp.zeros((1, 1)),
        dt=0.1,
    )
    augmented = cx.augment_integrator(sys)

    A_expected = jnp.array(
        [
            [1.0, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.0, 1.0],
        ]
    )
    B_expected = jnp.array([[0.0], [0.1], [0.0]])
    C_expected = jnp.array([[1.0, 0.0, 0.0]])

    assert isinstance(augmented, cx.DiscLTI)
    assert jnp.allclose(augmented.A, A_expected)
    assert jnp.allclose(augmented.B, B_expected)
    assert jnp.allclose(augmented.C, C_expected)
    assert jnp.allclose(augmented.D, sys.D)
    assert jnp.allclose(augmented.dt, sys.dt)


def test_augment_integrator_continuous_matches_manual_model():
    sys = cx.ss(
        jnp.array([[0.0, 1.0], [0.0, 0.0]]),
        jnp.array([[0.0], [1.0]]),
        jnp.array([[1.0, 0.0]]),
        jnp.zeros((1, 1)),
    )
    augmented = cx.augment_integrator(sys, sign=-1.0)

    A_expected = jnp.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    B_expected = jnp.array([[0.0], [1.0], [0.0]])

    assert isinstance(augmented, cx.ContLTI)
    assert jnp.allclose(augmented.A, A_expected)
    assert jnp.allclose(augmented.B, B_expected)
    assert jnp.allclose(augmented.C, jnp.array([[1.0, 0.0, 0.0]]))


def test_lqi_matches_lqr_on_integrator_augmented_model():
    sys = cx.dss(
        jnp.array([[1.0, 0.1], [0.0, 1.0]]),
        jnp.array([[0.0], [0.1]]),
        jnp.array([[1.0, 0.0]]),
        jnp.zeros((1, 1)),
        dt=0.1,
    )
    Q = jnp.diag(jnp.array([1.0, 0.1, 10.0]))
    R = jnp.array([[1.0]])

    lqi_result = cx.lqi(sys, Q, R)
    manual_result = cx.lqr(cx.augment_integrator(sys), Q, R)

    assert lqi_result.K.shape == (1, 3)
    assert jnp.allclose(lqi_result.K, manual_result.K, atol=1e-12)
    assert jnp.allclose(lqi_result.S, manual_result.S, atol=1e-12)
    assert jnp.all(jnp.abs(lqi_result.poles) < 1.0)


def test_augment_integrator_jit_dispatches_on_system_type():
    sys = cx.dss(
        jnp.array([[1.0, 0.1], [0.0, 1.0]]),
        jnp.array([[0.0], [0.1]]),
        jnp.array([[1.0, 0.0]]),
        jnp.zeros((1, 1)),
        dt=0.1,
    )

    augmented = jax.jit(cx.augment_integrator)(sys)

    assert isinstance(augmented, cx.DiscLTI)
    assert augmented.A.shape == (3, 3)
    assert augmented.B.shape == (3, 1)


# -- continuous LQR / CARE -----------------------------------------------------


def test_care_gain_matches_octave():
    """Continuous LQR should match Octave lqr(A,B,eye(3),1).

    Octave:
      pkg load control
      A=[-2.2 -0.4 0;1 0 0;0 1 0]; B=[0.25;0;0]; Q=eye(3); R=1;
      [K,S,e]=lqr(A,B,Q,R)
      % K = [1.4404  3.3031  1.0000]
    """
    Q = jnp.eye(3)
    R = jnp.array([[1.0]])

    result = cx.care(A_CONT, B_CONT, Q, R)

    K_expected = jnp.array([[1.4404, 3.3031, 1.0000]])
    assert jnp.allclose(result.K, K_expected, atol=0.05)


def test_care_double_integrator_matches_octave_reference():
    """Double-integrator CARE should match the standard Octave reference.

    Octave:
      pkg load control
      A = [0, 1; 0, 0];
      B = [0; 1];
      Q = eye(2);
      R = 1;
      [K, S, poles] = lqr(A, B, Q, R)
      % K = [1.0000, 1.7321]
      % S = [1.7321, 1.0000; 1.0000, 1.7321]
      % poles = [-0.8660 +/- 0.5000i]
    """
    A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    B = jnp.array([[0.0], [1.0]])
    Q = jnp.eye(2)
    R = jnp.array([[1.0]])

    result = cx.care(A, B, Q, R)

    K_expected = jnp.array([[1.0, jnp.sqrt(3.0)]])
    S_expected = jnp.array([[jnp.sqrt(3.0), 1.0], [1.0, jnp.sqrt(3.0)]])
    poles_expected = jnp.array(
        [
            -0.5 * jnp.sqrt(3.0) + 0.5j,
            -0.5 * jnp.sqrt(3.0) - 0.5j,
        ]
    )
    residual = _care_residual(A, B, Q, R, result.S)

    assert jnp.allclose(result.K, K_expected, atol=1e-6)
    assert jnp.allclose(result.S, S_expected, atol=1e-6)
    assert jnp.allclose(
        jnp.sort_complex(result.poles), jnp.sort_complex(poles_expected), atol=1e-6
    )
    assert jnp.max(jnp.abs(residual)) < 1e-8
    assert jnp.allclose(result.residual_norm, jnp.max(jnp.abs(residual)))


def test_care_riccati_residual_and_stability():
    Q = jnp.eye(3)
    R = jnp.array([[1.0]])

    result = cx.care(A_CONT, B_CONT, Q, R)

    residual = _care_residual(A_CONT, B_CONT, Q, R, result.S)
    assert jnp.max(jnp.abs(residual)) < 1e-8
    assert jnp.all(jnp.real(result.poles) < 0.0)


def test_continuous_lqr_dispatch_matches_care():
    sys = cx.ss(A_CONT, B_CONT, C_CONT, D_CONT)
    Q = jnp.eye(3)
    R = jnp.array([[1.0]])

    direct = cx.care(A_CONT, B_CONT, Q, R)
    dispatched = cx.lqr(sys, Q, R)

    assert jnp.allclose(dispatched.K, direct.K, atol=1e-10)
    assert jnp.allclose(dispatched.S, direct.S, atol=1e-10)


def test_care_jit_matches_eager():
    Q = jnp.eye(3)
    R = jnp.array([[1.0]])

    eager = cx.care(A_CONT, B_CONT, Q, R)
    jitted = jax.jit(cx.care)(A_CONT, B_CONT, Q, R)

    assert jnp.allclose(jitted.K, eager.K, atol=1e-10)
    assert jnp.allclose(jitted.S, eager.S, atol=1e-10)
    assert jnp.allclose(jitted.residual_norm, eager.residual_norm, atol=1e-12)
    assert jnp.all(jnp.real(jitted.poles) < 0.0)


def test_care_raises_when_hamiltonian_has_no_stable_subspace_split():
    A = jnp.zeros((1, 1))
    B = jnp.zeros((1, 1))
    Q = jnp.eye(1)
    R = jnp.eye(1)

    with pytest.raises(ValueError, match="stabilizing Hamiltonian subspace"):
        cx.care(A, B, Q, R)


# -- gradient tests ------------------------------------------------------------


def test_dare_custom_vjp_matches_finite_difference():
    A = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    B = jnp.array([[0.5], [1.0]])
    Q_diag = jnp.array([1.0, 1.0])
    R_val = jnp.array(1.0)
    eps = 1e-5

    def cost(Q_diag, R_val):
        result = cx.dare(A, B, jnp.diag(Q_diag), R_val[None, None])
        return jnp.sum(result.K**2)

    dQ, dR = jax.grad(cost, argnums=(0, 1))(Q_diag, R_val)
    fd_q0 = (
        cost(Q_diag + jnp.array([eps, 0.0]), R_val)
        - cost(Q_diag - jnp.array([eps, 0.0]), R_val)
    ) / (2 * eps)
    fd_q1 = (
        cost(Q_diag + jnp.array([0.0, eps]), R_val)
        - cost(Q_diag - jnp.array([0.0, eps]), R_val)
    ) / (2 * eps)
    fd_r = (cost(Q_diag, R_val + eps) - cost(Q_diag, R_val - eps)) / (2 * eps)

    assert jnp.allclose(dQ, jnp.array([fd_q0, fd_q1]), atol=1e-6, rtol=1e-5)
    assert jnp.allclose(dR, fd_r, atol=1e-6, rtol=1e-5)


def test_vmap_lqr_over_q_weights(disc_sys):
    """vmap over Q weightings must execute as a single JIT call."""
    key = jax.random.PRNGKey(0)
    Q_diags = jax.random.uniform(key, (10, 3), minval=1.0, maxval=1e3)
    R = jnp.array([[1.0]])

    Ks = jax.vmap(lambda q: cx.lqr(disc_sys, jnp.diag(q), R).K)(Q_diags)
    assert Ks.shape == (10, 1, 3)
    assert jnp.all(jnp.isfinite(Ks))


def test_care_custom_vjp_matches_finite_difference():
    Q_diag = jnp.array([1.0, 1.0, 1.0])
    R_val = jnp.array(1.0)
    eps = 1e-5

    def cost(Q_diag, R_val):
        result = cx.care(A_CONT, B_CONT, jnp.diag(Q_diag), R_val[None, None])
        return jnp.sum(result.K**2)

    dQ, dR = jax.grad(cost, argnums=(0, 1))(Q_diag, R_val)
    fd_q0 = (
        cost(Q_diag + jnp.array([eps, 0.0, 0.0]), R_val)
        - cost(Q_diag - jnp.array([eps, 0.0, 0.0]), R_val)
    ) / (2 * eps)
    fd_q1 = (
        cost(Q_diag + jnp.array([0.0, eps, 0.0]), R_val)
        - cost(Q_diag - jnp.array([0.0, eps, 0.0]), R_val)
    ) / (2 * eps)
    fd_q2 = (
        cost(Q_diag + jnp.array([0.0, 0.0, eps]), R_val)
        - cost(Q_diag - jnp.array([0.0, 0.0, eps]), R_val)
    ) / (2 * eps)
    fd_r = (cost(Q_diag, R_val + eps) - cost(Q_diag, R_val - eps)) / (2 * eps)

    assert jnp.allclose(dQ, jnp.array([fd_q0, fd_q1, fd_q2]), atol=1e-5, rtol=1e-4)
    assert jnp.allclose(dR, fd_r, atol=1e-5, rtol=1e-4)
