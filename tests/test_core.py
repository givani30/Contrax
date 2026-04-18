"""Tests for contrax.core.

Octave ground truth (octave --no-gui):
  pkg load control
  A=[-2.2 -0.4 0;1 0 0;0 1 0]; B=[0.25;0;0]; C=[0 0 0.4]; D=0;
  sys=ss(A,B,C,D); dsys=c2d(sys,0.20039,'zoh');
  dsys.A   % Phi[0,0]=0.6375, Phi[1,0]=0.1616, Phi[2,2]=1.0
  dsys.B   % Gamma[0,0]=0.04041

Note on SC42095 Assignment 21: the report shows Phi[0,0]=2.631. That value comes
from a *different* MATLAB-internal realization produced by ss(tf(G)), which builds
a different state basis than the hand-written A,B,C. Computing c2d on A,B,C directly
(verified here against Octave) gives Phi[0,0]=0.6375. Both are valid representations
of the same system; the Octave values are used as ground truth.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import test_util as jtu

import contrax as cx
from contrax.core import _safe_expm

A_CONT = jnp.array([[-2.2, -0.4, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
B_CONT = jnp.array([[0.25], [0.0], [0.0]])
C_CONT = jnp.array([[0.0, 0.0, 0.4]])
D_CONT = jnp.zeros((1, 1))
DT = 0.20039


@pytest.fixture
def cont_sys():
    return cx.ss(A_CONT, B_CONT, C_CONT, D_CONT)


@pytest.fixture
def disc_sys(cont_sys):
    return cx.c2d(cont_sys, DT)


# -- ss / dss ------------------------------------------------------------------


def test_ss_creates_cont_lti(cont_sys):
    assert isinstance(cont_sys, cx.ContLTI)
    assert cont_sys.A.shape == (3, 3)
    assert cont_sys.B.shape == (3, 1)
    assert cont_sys.C.shape == (1, 3)
    assert cont_sys.D.shape == (1, 1)


def test_dss_round_trip():
    dsys = cx.dss(A_CONT, B_CONT, C_CONT, D_CONT, DT)
    assert isinstance(dsys, cx.DiscLTI)
    assert dsys.dt.shape == ()


def test_dt_is_dynamic_leaf():
    """dt must be a plain Array leaf so vmap over different sample rates works.

    If dt were eqx.field(static=True) it would be embedded in the treedef,
    breaking vmap over systems with different sampling periods.
    """
    dsys = cx.dss(A_CONT, B_CONT, C_CONT, D_CONT, DT)
    leaves = jax.tree_util.tree_leaves(dsys)
    dt_leaves = [leaf for leaf in leaves if leaf.shape == ()]
    assert len(dt_leaves) >= 1


# -- c2d ZOH: Octave-verified numerical targets --------------------------------


def test_c2d_zoh_phi_00(disc_sys):
    """Phi[0,0] ~ 0.6375. Octave: c2d(ss(A,B,C,D),0.20039,'zoh').A(1,1)"""
    assert abs(float(disc_sys.A[0, 0]) - 0.6375) < 1e-3


def test_c2d_zoh_gamma_00(disc_sys):
    """Gamma[0,0] ~ 0.04041. Octave: c2d(ss(A,B,C,D),0.20039,'zoh').B(1,1)"""
    assert abs(float(disc_sys.B[0, 0]) - 0.04041) < 1e-4


def test_c2d_zoh_phi_22(disc_sys):
    """Phi[2,2] = 1.0: x2 is a pure integrator of x1, so its diagonal is 1."""
    assert abs(float(disc_sys.A[2, 2]) - 1.0) < 1e-6


def test_c2d_zoh_shapes(disc_sys):
    assert disc_sys.A.shape == (3, 3)
    assert disc_sys.B.shape == (3, 1)


def test_c2d_zoh_c_unchanged(disc_sys, cont_sys):
    """ZOH discretization must leave the C matrix unchanged."""
    assert jnp.allclose(disc_sys.C, cont_sys.C, atol=1e-12)


def test_c2d_zoh_eigenvalues(disc_sys):
    """Discrete eigenvalues must equal exp(lambda_c * dt).

    Continuous eigenvalues: {0, -2, -0.2}
    Expected discrete: {exp(0), exp(-2*0.20039), exp(-0.2*0.20039)}
              ~= {1.0, 0.670, 0.961}
    """
    import numpy as np

    expected = np.sort(
        np.abs(
            [
                np.exp(0.0 * DT),
                np.exp(-2.0 * DT),
                np.exp(-0.2 * DT),
            ]
        )
    )
    actual = np.sort(np.abs(jnp.linalg.eigvals(disc_sys.A)))
    assert jnp.allclose(jnp.array(actual), jnp.array(expected), atol=1e-5)


def test_c2d_tustin_matches_bilinear_transfer_map():
    """Tustin must preserve Hc(s) under s=(2/dt)(z-1)/(z+1).

    This checks the transfer-function identity directly instead of comparing
    one particular realization. It also catches the common mistake of leaving
    C and D unchanged after discretizing only the state update.
    """

    A = jnp.array([[-1.2, 0.4], [-0.3, -0.8]])
    B = jnp.array([[1.0], [0.2]])
    C = jnp.array([[0.5, -0.1]])
    D = jnp.array([[0.3]])
    dt = 0.1
    sys_c = cx.ss(A, B, C, D)
    sys_d = cx.c2d(sys_c, dt, method="tustin")

    def Hc(s):
        return C @ jnp.linalg.solve(s * jnp.eye(A.shape[0]) - A, B) + D

    def Hd(z):
        return (
            sys_d.C @ jnp.linalg.solve(z * jnp.eye(A.shape[0]) - sys_d.A, sys_d.B)
            + sys_d.D
        )

    for z in [
        0.6 + 0.2j,
        0.9 + 0.4j,
        -0.3 + 0.7j,
    ]:
        s = (2.0 / dt) * (z - 1.0) / (z + 1.0)
        expected = Hc(s)
        actual = Hd(z)
        assert jnp.allclose(actual, expected, atol=1e-10, rtol=1e-10)


# -- c2d gradient tests --------------------------------------------------------


def _expm_frechet(M, E):
    """Frechet derivative L_exp(M, E) via Al-Mohy/Higham block formula."""
    n = M.shape[0]
    aug = jnp.block([[M, E], [jnp.zeros_like(M), M]])
    return jax.scipy.linalg.expm(aug)[:n, n:]


def test_safe_expm_vjp_matches_adjoint_frechet_identity():
    """_safe_expm VJP must equal L_exp(M.T, G), the adjoint Frechet derivative."""
    M = jnp.array([[0.1, 0.7], [-0.3, -0.2]])
    G = jnp.array([[1.3, -0.4], [0.2, 0.9]])

    _, vjp_fn = jax.vjp(_safe_expm, M)
    (actual,) = vjp_fn(G)
    expected = _expm_frechet(M.T, G)

    assert jnp.allclose(actual, expected, atol=1e-11, rtol=1e-11)


def test_safe_expm_vjp_satisfies_inner_product_identity():
    """<G, L_exp(M,E)> must equal <VJP(G), E> for arbitrary directions."""
    M = jnp.array([[0.2, -0.5, 0.1], [0.3, -0.1, 0.4], [-0.2, 0.6, -0.3]])
    E = jnp.array([[0.7, 0.2, -0.1], [-0.4, 0.5, 0.3], [0.1, -0.6, 0.8]])
    G = jnp.array([[0.9, -0.2, 0.4], [0.5, -0.7, 0.1], [-0.3, 0.6, 0.2]])

    _, vjp_fn = jax.vjp(_safe_expm, M)
    (vjp_g,) = vjp_fn(G)

    lhs = jnp.vdot(G, _expm_frechet(M, E))
    rhs = jnp.vdot(vjp_g, E)

    assert jnp.allclose(lhs, rhs, atol=1e-11, rtol=1e-11)


def test_safe_expm_reverse_gradient_matches_finite_difference():
    """Reverse-mode gradient through _safe_expm should match finite differences."""
    M = jnp.array([[0.1, 0.7], [-0.3, -0.2]])

    def cost(X):
        return jnp.sum(_safe_expm(X) ** 2)

    jtu.check_grads(cost, (M,), order=1, modes=["rev"], atol=1e-5, rtol=1e-5)


def test_c2d_gradient_finite(cont_sys):
    """jax.grad through c2d must produce finite gradients.

    Without the custom_vjp on _safe_expm, native autodiff through the
    scaling-and-squaring algorithm accumulates catastrophic numerical error
    for stiff systems, producing NaN gradients.
    Reference: Al-Mohy & Higham (2009), 'Computing the Frechet Derivative
    of the Matrix Exponential', SIAM J. Matrix Anal. Appl.
    """

    def cost(A_flat):
        A = A_flat.reshape(3, 3)
        sys = cx.ss(A, B_CONT, C_CONT, D_CONT)
        dsys = cx.c2d(sys, DT)
        return jnp.sum(dsys.A**2) + jnp.sum(dsys.B**2)

    grad = jax.grad(cost)(A_CONT.ravel())
    assert jnp.all(jnp.isfinite(grad)), f"Non-finite gradient: {grad}"


def test_c2d_reverse_gradient_matches_finite_difference():
    """Reverse-mode gradients through ZOH c2d should match finite differences."""
    A = jnp.array([[-1.0, 0.2], [0.0, -0.3]])
    B = jnp.array([[1.0], [0.4]])
    C = jnp.eye(2)
    D = jnp.zeros((2, 1))
    dt = jnp.array(0.05)

    def cost(A, B, dt):
        sys = cx.ss(A, B, C, D)
        dsys = cx.c2d(sys, dt)
        return jnp.sum(dsys.A**2) + jnp.sum(dsys.B**2)

    jtu.check_grads(cost, (A, B, dt), order=1, modes=["rev"], atol=1e-5, rtol=1e-5)


def test_c2d_gradient_finite_for_stiff_system():
    """ZOH c2d gradients should stay finite for eigenvalues spanning orders."""
    A = jnp.diag(jnp.array([-1.0, -50.0, -1000.0]))
    B = jnp.array([[1.0], [0.1], [0.01]])
    C = jnp.eye(3)
    D = jnp.zeros((3, 1))
    dt = jnp.array(0.01)

    def cost(A_flat, B_flat, dt):
        sys = cx.ss(A_flat.reshape(3, 3), B_flat.reshape(3, 1), C, D)
        dsys = cx.c2d(sys, dt)
        return jnp.sum(dsys.A**2) + jnp.sum(dsys.B**2)

    grads = jax.grad(cost, argnums=(0, 1, 2))(A.ravel(), B.ravel(), dt)

    assert all(jnp.all(jnp.isfinite(g)) for g in grads)


def test_c2d_jit_stable(cont_sys):
    """c2d must JIT-compile and produce identical results on second call."""
    c2d_jit = jax.jit(lambda sys: cx.c2d(sys, DT))
    r1 = c2d_jit(cont_sys)
    r2 = c2d_jit(cont_sys)
    assert jnp.allclose(r1.A, r2.A)
