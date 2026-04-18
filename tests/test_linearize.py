"""Tests for contrax.linearize and contrax.linearize_ss.

linearize() and linearize_ss() use jacfwd (forward-mode AD) — results are exact
analytical Jacobians, not finite-difference approximations.
"""

import jax
import jax.numpy as jnp

import contrax as cx

# ── fixtures ──────────────────────────────────────────────────────────────


def _double_integrator(x, u):
    """ẋ = Ax + Bu with A=[[0,1],[0,0]], B=[[0],[1]]."""
    return jnp.array([x[1], u[0]])


def _damped_oscillator(x, u):
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
        """A, B match analytical Jacobians for linear system (should be exact)."""
        A, B = cx.linearize(_double_integrator, jnp.zeros(2), jnp.zeros(1))
        assert jnp.allclose(A, _A_di, atol=1e-12)
        assert jnp.allclose(B, _B_di, atol=1e-12)

    def test_shapes(self):
        A, B = cx.linearize(_double_integrator, jnp.zeros(2), jnp.zeros(1))
        assert A.shape == (2, 2)
        assert B.shape == (2, 1)

    def test_nonlinear_pendulum_at_origin(self):
        """Linearize nonlinear pendulum f(x,u)=[x1, -sin(x0)+u0] at origin.

        Analytical: A=[[0,1],[-1,0]], B=[[0],[1]].
        """

        def pendulum(x, u):
            return jnp.array([x[1], -jnp.sin(x[0]) + u[0]])

        A, B = cx.linearize(pendulum, jnp.zeros(2), jnp.zeros(1))
        assert jnp.allclose(A, jnp.array([[0.0, 1.0], [-1.0, 0.0]]), atol=1e-12)
        assert jnp.allclose(B, jnp.array([[0.0], [1.0]]), atol=1e-12)

    def test_nonlinear_at_nontrivial_eq(self):
        """Linearize pendulum at (pi, 0) — upright equilibrium.

        Analytical: A=[[0,1],[+1,0]] (d/dx0 of -sin(x0) at x0=pi = -cos(pi) = +1).
        """

        def pendulum(x, u):
            return jnp.array([x[1], -jnp.sin(x[0]) + u[0]])

        A, _ = cx.linearize(pendulum, jnp.array([jnp.pi, 0.0]), jnp.zeros(1))
        assert jnp.allclose(A, jnp.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-12)

    def test_jit_compatible(self):
        @jax.jit
        def get_AB(x0, u0):
            return cx.linearize(_double_integrator, x0, u0)

        A, B = get_AB(jnp.zeros(2), jnp.zeros(1))
        assert jnp.allclose(A, _A_di, atol=1e-12)

    def test_vmap_over_equilibria(self):
        """vmap over a batch of operating points (gain scheduling)."""

        def pendulum(x, u):
            return jnp.array([x[1], -jnp.sin(x[0]) + u[0]])

        x0s = jnp.stack([jnp.zeros(2), jnp.array([0.1, 0.0]), jnp.array([-0.1, 0.0])])
        u0s = jnp.zeros((3, 1))
        As, Bs = jax.vmap(cx.linearize, in_axes=(None, 0, 0))(pendulum, x0s, u0s)
        assert As.shape == (3, 2, 2)
        assert Bs.shape == (3, 2, 1)
        assert jnp.allclose(As[0, 1, 0], -1.0, atol=1e-12)
        assert jnp.allclose(As[1, 1, 0], -jnp.cos(0.1), atol=1e-12)

    def test_grad_through_linearize(self):
        """Gradient through linearize() must be finite (AD-over-AD)."""

        def loss(x0):
            A, B = cx.linearize(_damped_oscillator, x0, jnp.zeros(1))
            return jnp.sum(A**2)

        grad = jax.grad(loss)(jnp.zeros(2))
        assert jnp.all(jnp.isfinite(grad))

    def test_round_trip_with_ss_and_c2d(self):
        """Full workflow: linearize -> ss -> c2d -> lqr produces finite K."""
        A, B = cx.linearize(_double_integrator, jnp.zeros(2), jnp.zeros(1))
        sys_c = cx.ss(A, B, jnp.eye(2), jnp.zeros((2, 1)))
        sys_d = cx.c2d(sys_c, dt=0.05)
        result = cx.lqr(sys_d, jnp.eye(2), jnp.ones((1, 1)))
        assert jnp.all(jnp.isfinite(result.K))
        assert result.K.shape == (1, 2)

    def test_accepts_nonlinear_system_object(self):
        sys = cx.nonlinear_system(
            lambda t, x, u: _double_integrator(x, u),
            observation=lambda t, x, u: x,
            state_dim=2,
            input_dim=1,
            output_dim=2,
        )

        A, B = cx.linearize(sys, jnp.zeros(2), jnp.zeros(1))
        assert jnp.allclose(A, _A_di, atol=1e-12)
        assert jnp.allclose(B, _B_di, atol=1e-12)


# ── linearize_ss tests ─────────────────────────────────────────────────────


class TestLinearizeSS:
    def test_returns_contlti(self):
        sys = cx.linearize_ss(
            _double_integrator,
            jnp.zeros(2),
            jnp.zeros(1),
            output=_identity_output,
        )
        assert isinstance(sys, cx.ContLTI)

    def test_abcd_shapes(self):
        sys = cx.linearize_ss(
            _double_integrator,
            jnp.zeros(2),
            jnp.zeros(1),
            output=_feedthrough_output,
        )
        assert sys.A.shape == (2, 2)
        assert sys.B.shape == (2, 1)
        assert sys.C.shape == (2, 2)
        assert sys.D.shape == (2, 1)

    def test_abcd_values_double_integrator(self):
        """A, B from linearize_ss must match analytical for linear system."""
        sys = cx.linearize_ss(
            _double_integrator,
            jnp.zeros(2),
            jnp.zeros(1),
            output=_identity_output,
        )
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

    def test_full_workflow_linearize_ss_to_lqr(self):
        """linearize_ss -> c2d -> lqr produces finite K."""
        sys_c = cx.linearize_ss(
            _double_integrator,
            jnp.zeros(2),
            jnp.zeros(1),
            output=_identity_output,
        )
        sys_d = cx.c2d(sys_c, dt=0.05)
        result = cx.lqr(sys_d, jnp.eye(2), jnp.ones((1, 1)))
        assert jnp.all(jnp.isfinite(result.K))
        assert result.K.shape == (1, 2)

    def test_accepts_nonlinear_system_object(self):
        sys = cx.nonlinear_system(
            lambda t, x, u: _double_integrator(x, u),
            observation=lambda t, x, u: x,
            state_dim=2,
            input_dim=1,
            output_dim=2,
        )

        lin = cx.linearize_ss(sys, jnp.zeros(2), jnp.zeros(1))
        assert isinstance(lin, cx.ContLTI)
        assert jnp.allclose(lin.A, _A_di, atol=1e-12)
        assert jnp.allclose(lin.B, _B_di, atol=1e-12)
        assert jnp.allclose(lin.C, jnp.eye(2), atol=1e-12)
        assert jnp.allclose(lin.D, jnp.zeros((2, 1)), atol=1e-12)

    def test_nonlinear_system_accepts_output_alias(self):
        sys = cx.nonlinear_system(
            lambda t, x, u: _double_integrator(x, u),
            output=lambda t, x, u: x,
            state_dim=2,
            input_dim=1,
            output_dim=2,
        )

        lin = cx.linearize_ss(sys, jnp.zeros(2), jnp.zeros(1))
        assert jnp.allclose(lin.C, jnp.eye(2), atol=1e-12)
