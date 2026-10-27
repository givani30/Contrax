"""Tests for contrax.linearize and contrax.linearize_ss.

linearize() and linearize_ss() use jacfwd (forward-mode AD) — results are exact
analytical Jacobians, not finite-difference approximations.
"""

import jax
import jax.numpy as jnp
import pytest

import contrax as cx

# ── fixtures ──────────────────────────────────────────────────────────────


def _double_integrator(t, x, u):
    """ẋ = Ax + Bu with A=[[0,1],[0,0]], B=[[0],[1]]."""
    return jnp.array([x[1], u[0]])


def _damped_oscillator(t, x, u):
    """ẋ = [x1, -omega^2*x0 - 2*zeta*omega*x1 + u0], omega=2, zeta=0.5."""
    omega, zeta = 2.0, 0.5
    return jnp.array([x[1], -(omega**2) * x[0] - 2 * zeta * omega * x[1] + u[0]])


def _identity_output(x, u):
    """h(x, u) = x — full state output, D=0."""
    return x


def _feedthrough_output(x, u):
    """h(x, u) = [x0, u0] — partial state with feedthrough."""
    return jnp.array([x[0], u[0]])


# ── analytical Jacobians for double integrator ─────────────────────────────

_A_di = jnp.array([[0.0, 1.0], [0.0, 0.0]])
_B_di = jnp.array([[0.0], [1.0]])


# ── linearize tests ────────────────────────────────────────────────────────


class TestLinearize:
    def test_double_integrator_exact(self):
        """The linear model has exact Jacobians and full-state output by default."""
        sys = cx.linearize(_double_integrator, jnp.zeros(2), jnp.zeros(1))
        assert isinstance(sys, cx.ContLTI)
        assert sys.A.shape == (2, 2)
        assert sys.B.shape == (2, 1)
        assert jnp.allclose(sys.A, _A_di, atol=1e-12)
        assert jnp.allclose(sys.B, _B_di, atol=1e-12)
        assert jnp.allclose(sys.C, jnp.eye(2), atol=1e-12)
        assert jnp.allclose(sys.D, jnp.zeros((2, 1)), atol=1e-12)

    def test_nonlinear_pendulum_at_origin(self):
        """Linearize nonlinear pendulum f(t,x,u)=[x1, -sin(x0)+u0] at origin.

        Analytical: A=[[0,1],[-1,0]], B=[[0],[1]].
        """

        def pendulum(t, x, u):
            return jnp.array([x[1], -jnp.sin(x[0]) + u[0]])

        sys = cx.linearize(pendulum, jnp.zeros(2), jnp.zeros(1))
        assert jnp.allclose(sys.A, jnp.array([[0.0, 1.0], [-1.0, 0.0]]), atol=1e-12)
        assert jnp.allclose(sys.B, jnp.array([[0.0], [1.0]]), atol=1e-12)
        upright = cx.linearize(pendulum, jnp.array([jnp.pi, 0.0]), jnp.zeros(1))
        assert jnp.allclose(upright.A, jnp.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-12)

    def test_jit_compatible(self):
        @jax.jit
        def get_sys(x0, u0):
            return cx.linearize(_double_integrator, x0, u0)

        sys = get_sys(jnp.zeros(2), jnp.zeros(1))
        assert jnp.allclose(sys.A, _A_di, atol=1e-12)

    def test_vmap_over_equilibria(self):
        """vmap over a batch of operating points (gain scheduling)."""

        def pendulum(t, x, u):
            return jnp.array([x[1], -jnp.sin(x[0]) + u[0]])

        x0s = jnp.stack([jnp.zeros(2), jnp.array([0.1, 0.0]), jnp.array([-0.1, 0.0])])
        u0s = jnp.zeros((3, 1))
        systems = jax.vmap(cx.linearize, in_axes=(None, 0, 0))(pendulum, x0s, u0s)
        assert systems.A.shape == (3, 2, 2)
        assert systems.B.shape == (3, 2, 1)
        assert jnp.allclose(systems.A[0, 1, 0], -1.0, atol=1e-12)
        assert jnp.allclose(systems.A[1, 1, 0], -jnp.cos(0.1), atol=1e-12)

    def test_grad_through_linearize(self):
        """Gradient through linearize() must be finite (AD-over-AD)."""

        def loss(x0):
            sys = cx.linearize(_damped_oscillator, x0, jnp.zeros(1))
            return jnp.sum(sys.A**2)

        grad = jax.grad(loss)(jnp.zeros(2))
        assert jnp.all(jnp.isfinite(grad))

    def test_accepts_nonlinear_system_object(self):
        sys_nl = cx.nonlinear_system(
            _double_integrator,
            observation=lambda t, x, u: x,
            state_dim=2,
            input_dim=1,
            output_dim=2,
        )

        sys_lin = cx.linearize(sys_nl, jnp.zeros(2), jnp.zeros(1))
        assert jnp.allclose(sys_lin.A, _A_di, atol=1e-12)
        assert jnp.allclose(sys_lin.B, _B_di, atol=1e-12)
        sys_ss = cx.linearize_ss(sys_nl, jnp.zeros(2), jnp.zeros(1))
        assert jnp.allclose(sys_ss.A, _A_di, atol=1e-12)
        assert jnp.allclose(sys_ss.B, _B_di, atol=1e-12)
        assert jnp.allclose(sys_ss.C, jnp.eye(2), atol=1e-12)
        assert jnp.allclose(sys_ss.D, jnp.zeros((2, 1)), atol=1e-12)
        discrete = cx.nonlinear_system(_double_integrator, dt=0.2)
        local = jax.jit(lambda x: cx.linearize(discrete, x, jnp.zeros(1)))(jnp.zeros(2))
        assert isinstance(local, cx.DiscLTI)
        assert jnp.allclose(local.A, _A_di) and local.dt == 0.2
        assert cx.linearize(local, jnp.zeros(2), jnp.zeros(1)) is local

        def sampled_local(a):
            field = cx.nonlinear_system(lambda t, x, u: a * x + u)
            sampled = cx.sample_system(field, 0.2, method="rk4")
            return cx.linearize(sampled, jnp.zeros(1), jnp.zeros(1))

        z = -0.2
        expected = 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
        sampled = sampled_local(-1.0)
        assert isinstance(sampled, cx.DiscLTI)
        assert jnp.allclose(sampled.A[0, 0], expected, atol=1e-12)
        assert jnp.allclose(sampled.B[0, 0], 1 - expected, atol=1e-12)
        def state_jacobian(a):
            return sampled_local(a).A[0, 0]

        derivative = 0.2 * (1 + z + z**2 / 2 + z**3 / 6)
        assert jnp.allclose(jax.jacfwd(state_jacobian)(-1.0), derivative, atol=1e-12)
        assert jnp.allclose(jax.grad(state_jacobian)(-1.0), derivative, atol=1e-12)
        with pytest.raises(ValueError, match="vectors"):
            cx.linearize(discrete, jnp.zeros(2), jnp.zeros((2, 1)))


# ── linearize_ss tests ─────────────────────────────────────────────────────


class TestLinearizeSS:
    def test_abcd_values_double_integrator(self):
        """A, B from linearize_ss must match analytical for linear system."""
        sys = cx.linearize_ss(
            _double_integrator,
            jnp.zeros(2),
            jnp.zeros(1),
            output=_identity_output,
        )
        assert isinstance(sys, cx.ContLTI)
        assert sys.A.shape == (2, 2)
        assert sys.B.shape == (2, 1)
        assert sys.C.shape == (2, 2)
        assert sys.D.shape == (2, 1)
        assert jnp.allclose(sys.A, _A_di, atol=1e-12)
        assert jnp.allclose(sys.B, _B_di, atol=1e-12)
        assert jnp.allclose(sys.C, jnp.eye(2), atol=1e-12)
        assert jnp.allclose(sys.D, jnp.zeros((2, 1)), atol=1e-12)

    def test_feedthrough_D(self):
        """h(x,u)=[x0,u0] should give D=[[0],[1]]."""
        sys = cx.linearize_ss(
            _double_integrator,
            jnp.zeros(2),
            jnp.zeros(1),
            output=_feedthrough_output,
        )
        assert jnp.allclose(sys.D, jnp.array([[0.0], [1.0]]), atol=1e-12)

    def test_jit_compatible(self):
        @jax.jit
        def get_sys(x0, u0):
            return cx.linearize_ss(_double_integrator, x0, u0, output=_identity_output)

        sys = get_sys(jnp.zeros(2), jnp.zeros(1))
        assert jnp.allclose(sys.A, _A_di, atol=1e-12)

    def test_nonlinear_system_accepts_output_alias(self):
        sys = cx.nonlinear_system(
            _double_integrator,
            output=lambda t, x, u: x,
            state_dim=2,
            input_dim=1,
            output_dim=2,
        )

        lin = cx.linearize_ss(sys, jnp.zeros(2), jnp.zeros(1))
        assert jnp.allclose(lin.C, jnp.eye(2), atol=1e-12)
