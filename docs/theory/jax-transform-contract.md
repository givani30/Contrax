# JAX Transform Contract

JAX transform behavior is part of the public contract. APIs document how they
compose with `jit`, `vmap`, and `grad` when those details affect use.

## Core Idea

Assess a solver or simulation helper by its forward result and by its behavior
under compilation, batching, and differentiation.

## `jit`

Functions intended for compiled use should avoid Python control flow over JAX
runtime values and should keep traced computation inside JAX primitives.

This matters especially in:

- time loops for simulation and filtering
- iterative solver paths
- type dispatch that must happen outside traced runtime values

## `vmap`

Batching should work naturally when the underlying data model is compatible.

One concrete design choice here is that physically meaningful scalars such as
`dt` stay as array leaves instead of static metadata. That keeps batching over
systems or operating points much more natural.

## `grad`

Differentiable paths should say how gradients work, not merely whether they
exist.

Important distinctions include:

- plain autodiff through algebraic code
- native forward/reverse derivatives through algebraic primitives
- implicit differentiation through a fixed-point or Riccati solve

Document limitations when a public function cannot compose with `grad`.

In practice, the important distinction is simple:

\[
\text{forward solve} \neq \text{backward contract}
\]

A method can produce an acceptable forward result while its derivative path
unrolls unstable iterations, leaves JAX, or uses excessive memory.

## Practical Design Rules

- use `jax.lax.scan` for time loops and fixed-count iterative solvers
- use `jax.lax.while_loop` for convergence loops
- prefer `jnp.linalg.solve` to explicit inverses
- keep SciPy out of traced hot paths
- keep type dispatch simple and explicit at the JIT boundary

## User Expectations

The main expectations are:

- Importing the package leaves `jax_enable_x64` unchanged; precision-sensitive
  solvers require users to enable float64 explicitly
- discrete and continuous LTI `lsim()` support fixed-shape `jit`, `vmap`, JVP
  and VJP; continuous calls require an explicit sample grid and hold convention
- `simulate()` uses `num_steps` for discrete systems and `duration` for
  continuous systems so horizon semantics stay explicit
- continuous `simulate()` returns values on a fixed save grid; adaptive
  internal solver steps remain internal
- simulation results expose backend status and integration step counts;
  `throw=True` checks failures even when compiled code consumes just one field
- continuous simulation defaults to a checkpoint adjoint for reverse losses;
  sampled models default to a direct adjoint for forward state Jacobians
- `step_response()` and `impulse_response()` use fixed event times; gradients
  with respect to event timing require a smooth input parameterization
- `lqr()` on the discrete path supports finite gradients through representative
  objectives
- Riccati and Lyapunov solvers support implicit JVP and VJP without unrolling
  their convergence loops
- vmapped workflows such as `linearize -> c2d -> lqr` are part of the
  intended use case
- `place()` is a design-time helper: its numerical core is JAX-native, but
  method selection and fallback logic use ordinary Python dispatch

## Validation

Contrax validates transform behavior with dedicated `jit`, `vmap`, and `grad`
tests on representative workflows such as:

- discrete simulation and filtering scans
- `linearize -> c2d -> lqr` design pipelines
- gradients through Riccati-based controller design objectives
- continuous-time design-and-simulate examples where the solver path supports it

## Related Pages

- [Riccati solvers](riccati-solvers.md) for solver-specific gradient contracts
- [Control API](../api/control.md)
- [Simulation API](../api/simulation.md)
- [Estimation API](../api/estimation.md)
- [JAX-native workflows](../examples/jax-native-workflows.md)
