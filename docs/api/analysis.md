# Analysis

Analysis is the public namespace for structural checks and transfer-oriented
inspection of state-space models.

The namespace includes:

- `ctrb()` and `obsv()` for controllability and observability matrices
- `poles()` for eigenvalue inspection
- `evalfr()`, `freqresp()`, and `dcgain()` for transfer evaluation
- `ctrb_gramian()` and `obsv_gramian()` for finite-horizon continuous Gramians

## Minimal Example

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import contrax as cx

sys = cx.ss(
    jnp.array([[0.0, 1.0], [-2.0, -0.4]]),
    jnp.array([[0.0], [1.0]]),
    jnp.array([[1.0, 0.0]]),
    jnp.zeros((1, 1)),
)

poles = cx.poles(sys)
dc = cx.dcgain(sys)
Wc = cx.ctrb_gramian(sys, t=2.0)
```

## Conventions

- `poles(sys)` returns the system eigenvalues
- `evalfr(sys, point)` evaluates the transfer map at one complex point
- `freqresp(sys, omega)` evaluates over a frequency grid
- `dcgain(sys)` evaluates the zero-frequency or steady-state gain

These helpers are intentionally focused on state-space analysis rather than a
full transfer-function algebra layer.

## Transform Behavior

Most analysis helpers are direct array computations on fixed system data, so
they fit naturally into `jit` and batched workflows when the underlying linear
algebra path does.

The Gramian helpers are continuous-time, matrix-exponential-based utilities
rather than hot-path primitives.

## Numerical Notes

Finite-horizon Gramians use a Van Loan block-matrix exponential construction
and require `jax_enable_x64=True`.

When comparing realizations, prefer invariant checks such as rank,
conditioning, pole location, and transfer behavior. Exact matrix equality is
only meaningful when the realization basis is intentionally identical.

## Related Pages

- [Systems](systems.md) for the models these helpers inspect
- [Simulation](simulation.md) for time-domain validation alongside frequency-
  domain or structural checks
- [JAX-native workflows](../examples/jax-native-workflows.md) for broader
  end-to-end usage patterns

::: contrax.analysis
