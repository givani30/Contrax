# --8<-- [start:script]
"""Gradient-based tuning of discrete LQR weights."""

from __future__ import annotations

import jax

# --8<-- [start:setup]
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import contrax as cx

A = jnp.array([[1.0, 0.05], [0.0, 1.0]])
B = jnp.array([[0.0], [0.05]])
C = jnp.eye(2)
D = jnp.zeros((2, 1))
SYS = cx.dss(A, B, C, D, dt=0.05)
X0 = jnp.array([1.0, 0.0])
# --8<-- [end:setup]


# --8<-- [start:objective]
def closed_loop_cost(log_q_diag, log_r):
    q_diag = jnp.exp(log_q_diag)
    r = jnp.exp(log_r)[None, None]
    K = cx.lqr(SYS, jnp.diag(q_diag), r).K
    _, xs, _ = cx.simulate(SYS, X0, lambda t, x: -K @ x, num_steps=80)
    control_energy = jnp.sum((xs[:-1] @ K.T) ** 2)
    return jnp.sum(xs**2) + 1e-2 * control_energy


# --8<-- [end:objective]


# --8<-- [start:optimization-loop]
def run_example(num_steps: int = 60, learning_rate: float = 0.08):
    params = (
        jnp.log(jnp.array([1.0, 1.0])),
        jnp.array(0.0),
    )
    objective_and_grad = jax.jit(jax.value_and_grad(closed_loop_cost, argnums=(0, 1)))

    initial_cost, _ = objective_and_grad(*params)
    history = [float(initial_cost)]

    for _ in range(num_steps):
        loss, grads = objective_and_grad(*params)
        dq, dr = grads
        params = (
            params[0] - learning_rate * dq,
            params[1] - learning_rate * dr,
        )
        history.append(float(loss))

    final_cost = float(closed_loop_cost(*params))
    Q_final = jnp.diag(jnp.exp(params[0]))
    R_final = jnp.exp(params[1])[None, None]
    K_final = cx.lqr(SYS, Q_final, R_final).K

    assert np.isfinite(final_cost)
    assert final_cost < history[0]

    return {
        "initial_cost": history[0],
        "final_cost": final_cost,
        "Q_diag_final": np.asarray(jnp.diag(Q_final)),
        "R_final": float(R_final[0, 0]),
        "K_final": np.asarray(K_final),
    }


# --8<-- [end:optimization-loop]


def main():
    result = run_example()
    print("Differentiable LQR tuning")
    print(f"initial cost = {result['initial_cost']:.6f}")
    print(f"final cost   = {result['final_cost']:.6f}")
    print(f"final Q diag = {result['Q_diag_final']}")
    print(f"final R      = {result['R_final']:.6f}")
    print(f"final K      = {result['K_final']}")


if __name__ == "__main__":
    main()
# --8<-- [end:script]
