# JAX Transform Contract

Contrax is not only a control library. It is a control library designed to
behave predictably under JAX transforms.

That means public APIs should be explicit about `jit`, `vmap`, and `grad`
behavior rather than treating transforms as an implementation detail.

## Core Idea

In Contrax, transform behavior is part of the feature contract. A solver or
simulation helper is judged not only by its forward result, but also by how it
behaves inside `jit`, under batching, and inside differentiable objectives.

## `jit`

Functions intended for compiled use should avoid Python control flow over JAX
runtime values and should keep traced computation inside JAX primitives.

For Contrax, that especially matters in:

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
- custom VJP through a numerically sensitive primitive
- implicit differentiation through a fixed-point or Riccati solve

If a public function does not compose well with `grad`, the docs should say so.

In practice, the important distinction is simple:

\[
\text{forward solve} \neq \text{backward contract}
\]

A method can be numerically acceptable in forward mode and still be a poor fit
for differentiation if the backward pass requires unrolling unstable
iterations, leaves the JAX execution model, or creates the wrong memory story.

## Practical Design Rules

- use `jax.lax.scan` for time loops and fixed-count iterative solvers
- use `jax.lax.while_loop` for convergence loops
- prefer `jnp.linalg.solve` to explicit inverses
- keep SciPy out of traced hot paths
- keep type dispatch simple and explicit at the JIT boundary

## User Expectations

The main expectations are:

- Contrax does not mutate `jax_enable_x64` on import; precision-sensitive
  solvers raise a clear error unless users opt into float64 explicitly
- discrete `lsim()` and `simulate()` are JIT-compatible on fixed-shape inputs
- `simulate()` uses `num_steps` for discrete systems and `duration` for
  continuous systems so horizon semantics stay explicit
- continuous `simulate()` returns values on a fixed save grid rather than
  exposing adaptive internal solver steps directly
- `step_response()` and `impulse_response()` are analysis helpers; they are
  not the preferred path for gradients with respect to nonsmooth event timing
- `lqr()` on the discrete path supports finite gradients through representative
  objectives
- vmapped workflows such as `linearize_ss -> c2d -> lqr` are part of the
  intended use case
- `place()` is a design-time helper: its numerical core is JAX-native, but
  method selection and fallback logic use ordinary Python dispatch

The docs should treat these transform contracts as part of the feature, not as
optional extra detail.

## Documentation Structure

The docs structure mirrors the transform story:

- tutorials show a full end-to-end success path
- how-to guides isolate a single practical task
- API pages record exact argument, shape, and return conventions
- theory pages explain the why behind the public contracts

That split matters for a JAX-native numerical library because transform
behavior is easy to accidentally hide if the docs only show signatures or only
show polished tutorials.

## Validation

Contrax validates transform behavior with dedicated `jit`, `vmap`, and `grad`
tests on representative workflows such as:

- discrete simulation and filtering scans
- `linearize_ss -> c2d -> lqr` design pipelines
- gradients through Riccati-based controller design objectives
- continuous-time design-and-simulate examples where the solver path supports it

## Related Pages

- [Riccati solvers](riccati-solvers.md) for solver-specific gradient contracts
- [Control API](../api/control.md)
- [Simulation API](../api/simulation.md)
- [Estimation API](../api/estimation.md)
- [JAX-native workflows](../examples/jax-native-workflows.md)
