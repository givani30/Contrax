# How To Tune LQR Weights With Gradients

This guide shows how to optimize discrete LQR weights inside an ordinary JAX
objective.

## Complete Working Code

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import contrax as cx

A = jnp.array([[1.0, 0.05], [0.0, 1.0]])
B = jnp.array([[0.0], [0.05]])
sys = cx.dss(A, B, jnp.eye(2), jnp.zeros((2, 1)), dt=0.05)
x0 = jnp.array([1.0, 0.0])


def closed_loop_cost(log_q_diag, log_r):
    Q = jnp.diag(jnp.exp(log_q_diag))
    R = jnp.exp(log_r)[None, None]
    K = cx.lqr(sys, Q, R).K
    _, xs, _ = cx.simulate(sys, x0, lambda t, x: -K @ x, num_steps=80)
    control_energy = jnp.sum((xs[:-1] @ K.T) ** 2)
    return jnp.sum(xs**2) + 1e-2 * control_energy


objective_and_grad = jax.jit(jax.value_and_grad(closed_loop_cost, argnums=(0, 1)))

params = (jnp.zeros(2), jnp.array(0.0))
for _ in range(40):
    loss, (dq, dr) = objective_and_grad(*params)
    params = (
        params[0] - 0.08 * dq,
        params[1] - 0.08 * dr,
    )

final_Q = jnp.diag(jnp.exp(params[0]))
final_R = jnp.exp(params[1])[None, None]
final_K = cx.lqr(sys, final_Q, final_R).K
```

## Why This Recipe Works

The recipe is optimizing a discrete closed-loop objective of the form

$$
x_{k+1} = A x_k + B u_k, \qquad
u_k = -K x_k
$$

with

$$
Q = \operatorname{diag}(\exp(\theta_Q)), \qquad
R = \exp(\theta_R)
$$

so positivity is enforced by construction while the optimization still runs on
unconstrained parameters $\theta_Q$ and $\theta_R$.

- The discrete `lqr()` path goes through `dare()`, which is the most mature
  Riccati path in the library.
- Discrete `simulate()` is a pure JAX scan on fixed-shape inputs.
- Log-parameterizing `Q` and `R` keeps the weights positive while still letting
  you optimize unconstrained parameters.

## Key Choices

- Use the discrete path first when you want the cleanest gradient story.
- Keep the objective scalar and fixed-shape.
- Inspect `LQRResult.residual_norm` on unfamiliar systems if solver quality is
  part of the optimization logic.

## Check

At minimum, verify:

- the objective decreases over optimization steps
- the final gain is finite
- the tuned weights remain in a numerically plausible range

## Related Pages

- [Differentiable LQR](../tutorials/differentiable-lqr.md)
- [Control API](../api/control.md)
- [Riccati solvers](../theory/riccati-solvers.md)
