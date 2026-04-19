# --8<-- [start:script]
"""Linearize nonlinear dynamics, design LQR, and simulate the closed loop."""

from __future__ import annotations

import jax

# --8<-- [start:setup]
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import contrax as cx

# --8<-- [end:setup]


# --8<-- [start:dynamics]
def pendulum(t, x, u):
    theta, theta_dot = x
    torque = u[0]
    return jnp.array([theta_dot, -jnp.sin(theta) + torque])


def sensor(x, u):
    return x


# --8<-- [end:dynamics]


# --8<-- [start:design-and-simulate]
@jax.jit
def design_and_simulate(x_eq, u_eq, x0):
    sys_c = cx.linearize_ss(pendulum, x_eq, u_eq, output=sensor)
    sys_d = cx.c2d(sys_c, dt=0.05)
    result = cx.lqr(sys_d, jnp.eye(2), jnp.ones((1, 1)))
    ts, xs, ys = cx.simulate(sys_d, x0, lambda t, x: -result.K @ x, num_steps=80)
    return result.poles, ts, xs, ys


# --8<-- [end:design-and-simulate]


# --8<-- [start:validate]
def run_example():
    x_eq = jnp.array([0.1, 0.0])
    u_eq = jnp.array([jnp.sin(0.1)])
    x0 = jnp.array([0.25, 0.0])

    poles, ts, xs, ys = design_and_simulate(x_eq, u_eq, x0)

    initial_norm = float(jnp.linalg.norm(xs[0]))
    final_norm = float(jnp.linalg.norm(xs[-1]))
    assert xs.shape == (81, 2)
    assert ys.shape == (80, 2)
    assert final_norm < initial_norm * 0.15

    return {
        "poles": np.asarray(poles),
        "initial_norm": initial_norm,
        "final_norm": final_norm,
        "final_state": np.asarray(xs[-1]),
        "time_horizon": float(ts[-1]) if ts.size else 0.0,
    }


# --8<-- [end:validate]


def main():
    result = run_example()
    print("Linearize -> c2d -> lqr -> simulate")
    print(f"closed-loop poles = {result['poles']}")
    print(f"initial state norm = {result['initial_norm']:.6f}")
    print(f"final state norm   = {result['final_norm']:.6f}")
    print(f"final state        = {result['final_state']}")
    print(f"time horizon       = {result['time_horizon']:.3f} s")


if __name__ == "__main__":
    main()
# --8<-- [end:script]
