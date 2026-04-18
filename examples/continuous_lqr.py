"""End-to-end continuous LQR example.

Double integrator `x = [position, velocity]` with unit cost weights:
  min ∫(x'Qx + u'Ru) dt
  s.t. ẋ = Ax + Bu

Shows care()-based lqr(), closed-loop simulate(), and a gradient smoke test
confirming that jax.grad works through care().
"""

from __future__ import annotations

# --8<-- [start:setup]
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import contrax as cx

# --8<-- [end:setup]

# ── System definition ──────────────────────────────────────────────────────
# Double integrator:  ẍ = u  →  ẋ = [ẋ, ẍ] = [v, u]
# --8<-- [start:system]
A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
B = jnp.array([[0.0], [1.0]])
C = jnp.eye(2)
D = jnp.zeros((2, 1))
SYS = cx.ss(A, B, C, D)

Q = jnp.eye(2)
R = jnp.array([[1.0]])
X0 = jnp.array([1.0, 0.0])  # start at position=1, velocity=0
# --8<-- [end:system]


# ── LQR design ────────────────────────────────────────────────────────────
# --8<-- [start:design-and-simulate]
def design_and_simulate(q_scale: float = 1.0, duration: float = 10.0):
    """Design a CARE-backed LQR and simulate the closed-loop response.

    Args:
        q_scale: Scalar multiplier on Q. Increase for faster settling.
        duration: Simulation duration in seconds.

    Returns:
        dict with ts, xs, K, and closed-loop poles.
    """
    result = cx.lqr(SYS, q_scale * Q, R)
    K = result.K

    ts, xs, _ = cx.simulate(SYS, X0, lambda t, x: -K @ x, duration=duration, dt=0.05)
    return {
        "ts": ts,
        "xs": xs,
        "K": K,
        "poles": result.poles,
        "residual_norm": result.residual_norm,
    }


# --8<-- [end:design-and-simulate]


# ── Gradient smoke test ────────────────────────────────────────────────────
# Verify that jax.grad flows through care() all the way to a scalar loss.
# --8<-- [start:gradient-check]


def settling_cost(log_q_scale: float) -> float:
    """Scalar cost: integral of x'x under the optimal continuous-time LQR."""
    q_scale = jnp.exp(log_q_scale)
    K = cx.lqr(SYS, q_scale * Q, R).K
    _, xs, _ = cx.simulate(SYS, X0, lambda t, x: -K @ x, duration=8.0, dt=0.05)
    return jnp.sum(xs**2)


def run_gradient_check():
    grad_fn = jax.jit(jax.grad(settling_cost))
    g = grad_fn(jnp.array(0.0))
    assert jnp.isfinite(g), f"gradient is not finite: {g}"
    return float(g)


# --8<-- [end:gradient-check]


# ── Main ───────────────────────────────────────────────────────────────────


def run_example():
    out = design_and_simulate()
    xs = np.asarray(out["xs"])
    poles = np.asarray(out["poles"])
    grad = run_gradient_check()
    stable = bool(np.all(np.real(poles) < 0.0))

    assert stable, "LQR poles must be stable"
    assert np.all(np.isfinite(xs)), "trajectory contains NaN/Inf"
    assert np.allclose(xs[-1], 0.0, atol=1e-2), "state did not converge to zero"
    assert np.isfinite(grad), "gradient through care() is not finite"

    return {
        "K": np.asarray(out["K"]),
        "poles": poles,
        "stable": stable,
        "residual_norm": float(out["residual_norm"]),
        "initial_state": xs[0],
        "final_state": xs[-1],
        "time_horizon": float(np.asarray(out["ts"])[-1]),
        "gradient": grad,
    }


def main():
    result = run_example()

    print("Continuous LQR — double integrator")
    print(f"  K = {result['K']}")
    print(f"  closed-loop poles = {result['poles']}")
    print(f"  all poles stable  = {result['stable']}")
    print(f"  residual norm     = {result['residual_norm']:.6e}")
    print(f"  x[0]  = {result['initial_state']}")
    print(f"  x[-1] = {result['final_state']}  (should be near zero)")
    print(f"  time horizon      = {result['time_horizon']:.3f} s")

    print("\nGradient smoke test (d/d(log q) of settling cost):")
    grad_is_finite = np.isfinite(result["gradient"])
    print(f"  grad = {result['gradient']:.6f}  (finite: {grad_is_finite})")

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
