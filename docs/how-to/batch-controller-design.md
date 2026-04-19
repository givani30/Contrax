# How To Batch Controller Design Over Operating Points

This guide shows how to use `vmap` over linearization points for gain
scheduling or operating-point sweeps.

## Complete Working Code

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import contrax as cx


def pendulum(t, x, u):
    return jnp.array([x[1], -jnp.sin(x[0]) + u[0]])


def output(x, u):
    return x


def design(x_eq, u_eq):
    sys_c = cx.linearize(pendulum, x_eq, u_eq, output=output)
    sys_d = cx.c2d(sys_c, dt=0.05)
    return cx.lqr(sys_d, jnp.eye(2), jnp.ones((1, 1))).K


x_eqs = jnp.array([[0.0, 0.0], [0.1, 0.0], [-0.1, 0.0]])
u_eqs = jnp.zeros((3, 1))
Ks = jax.jit(jax.vmap(design))(x_eqs, u_eqs)
```

## Why This Recipe Works

- `linearize()` is designed to compose with `vmap`
- `DiscLTI.dt` stays as an array leaf rather than static metadata
- the design function keeps one fixed-shape workflow:
  `linearize -> c2d -> lqr`

## Key Choices

- Keep every operating point the same state and input shape.
- Enable float64 before calling `c2d()` or `lqr()`.
- Start with a small grid of operating points before building a full
  gain-scheduling workflow.

## Check

Verify:

- `Ks.shape == (num_points, m, n)`
- the gains are finite
- nearby operating points produce sensibly related gains

## Related Pages

- [JAX-native workflows](../examples/jax-native-workflows.md)
- [Systems API](../api/systems.md)
- [JAX transform contract](../theory/jax-transform-contract.md)
