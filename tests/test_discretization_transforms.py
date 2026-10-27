"""Forward/reverse and higher derivatives of ZOH discretization.

Independent oracles: SciPy expm_frechet for stiff/nonnormal matrices; for the
scalar ODE x'=-a*x+b*u, Phi=exp(-a*h), Gamma=b*(1-Phi)/a.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm_frechet

import contrax as cx
from contrax.core import _safe_expm


@pytest.mark.parametrize("scale", [10.0, 100.0, 1000.0, 10000.0])
def test_stiff_nonnormal_forward_reverse_match_independent_frechet(scale):
    M = (
        jnp.array([[-1.0, scale, 0.0], [0.0, -scale, scale], [0.0, 0.0, -2 * scale]])
        * 0.1
    )
    E = jnp.array([[0.2, -0.1, 0.3], [0.0, -0.2, 0.1], [0.1, 0.0, 0.2]])
    G = jnp.array([[1.0, -0.2, 0.4], [0.1, 0.5, -0.6], [0.3, 0.2, 0.7]])
    _, forward = jax.jit(lambda m, e: jax.jvp(_safe_expm, (m,), (e,)))(M, E)
    _, pullback = jax.vjp(_safe_expm, M)
    reverse = pullback(G)[0]
    np.testing.assert_allclose(
        forward,
        expm_frechet(np.asarray(M), np.asarray(E), compute_expm=False),
        atol=1e-10,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        reverse,
        expm_frechet(np.asarray(M.T), np.asarray(G), compute_expm=False),
        atol=1e-10,
        rtol=1e-9,
    )
    np.testing.assert_allclose(jnp.vdot(forward, G), jnp.vdot(E, reverse), atol=1e-10)


def test_c2d_forward_reverse_hessian_and_batch_match_scalar_analytic_map():
    def outputs(theta):
        a, b, h = theta
        model = cx.ss(-a[None, None], b[None, None], jnp.eye(1), jnp.zeros((1, 1)))
        discrete = cx.c2d(model, h)
        return jnp.stack([discrete.A[0, 0], discrete.B[0, 0]])

    def reference(theta):
        a, b, h = theta
        phi = jnp.exp(-a * h)
        return jnp.stack([phi, b * (1 - phi) / a])

    theta = jnp.array([1.2, 0.7, 0.2])
    np.testing.assert_allclose(
        jax.jit(jax.jacfwd(outputs))(theta), jax.jacfwd(reference)(theta), atol=1e-11
    )
    np.testing.assert_allclose(
        jax.jacrev(outputs)(theta), jax.jacrev(reference)(theta), atol=1e-11
    )

    def cost(t):
        return jnp.sum(outputs(t) ** 2)

    def oracle(t):
        return jnp.sum(reference(t) ** 2)

    np.testing.assert_allclose(
        jax.jit(jax.hessian(cost))(theta), jax.hessian(oracle)(theta), atol=1e-10
    )
    np.testing.assert_allclose(
        jax.jit(jax.vmap(outputs))(jnp.stack([theta, theta * 1.1])),
        jax.vmap(reference)(jnp.stack([theta, theta * 1.1])),
        atol=1e-11,
    )


def test_higher_order_discretize_design_workflow_matches_scalar_riccati_root():
    """Scalar DARE: b² S² + [r(1-a²)-q b²]S - qr=0."""

    def designed(theta):
        a, b, h = theta
        model = cx.ss(-a[None, None], b[None, None], jnp.eye(1), jnp.zeros((1, 1)))
        result = cx.lqr(cx.c2d(model, h), jnp.eye(1), 0.5 * jnp.eye(1))
        return result.K[0, 0] ** 2 + 0.1 * result.S[0, 0]

    def reference(theta):
        a, b, h = theta
        phi = jnp.exp(-a * h)
        gamma = b * (1 - phi) / a
        coefficient = 0.5 * (1 - phi**2) - gamma**2
        S = (-coefficient + jnp.sqrt(coefficient**2 + 2 * gamma**2)) / (2 * gamma**2)
        K = gamma * S * phi / (0.5 + gamma**2 * S)
        return K**2 + 0.1 * S

    theta = jnp.array([0.7, 1.0, 0.2])
    np.testing.assert_allclose(
        jax.jit(jax.hessian(designed))(theta),
        jax.hessian(reference)(theta),
        rtol=1e-8,
        atol=1e-8,
    )


def test_care_gain_hessian_is_independent_of_unused_pole_diagnostics():
    def design(theta):
        a, b = theta
        result = cx.care(-a[None, None], b[None, None], jnp.eye(1), 0.5 * jnp.eye(1))
        return result.K[0, 0] ** 2 + 0.1 * result.S[0, 0]

    def reference(theta):
        a, b = theta
        S = 0.5 * (-a + jnp.sqrt(a * a + 2 * b * b)) / (b * b)
        return (2 * b * S) ** 2 + 0.1 * S

    theta = jnp.array([0.7, 1.0])
    np.testing.assert_allclose(
        jax.jit(jax.hessian(design))(theta),
        jax.hessian(reference)(theta),
        rtol=1e-8,
        atol=1e-8,
    )
