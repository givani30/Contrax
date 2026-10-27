# Simulation

Simulation is the public namespace for open-loop, closed-loop, and fixed-horizon
trajectory generation in Contrax.

The namespace includes:

- `lsim()` for open-loop discrete or sampled continuous LTI simulation
- `simulate()` for closed-loop discrete and continuous simulation
- `rollout()` for fixed-horizon nonlinear transitions without a system object
- `sample_system()` for solver-backed continuous-to-discrete nonlinear model
  construction
- `foh_inputs()` for first-order-hold endpoint pairing on sampled inputs
- `step_response()`, `impulse_response()`, and `initial_response()` for
  standard response views
- `as_ode_term()` as a lower-level bridge into direct Diffrax use

## Minimal Example

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import contrax as cx

sys = cx.dss(
    jnp.array([[1.0, 0.05], [0.0, 1.0]]),
    jnp.array([[0.0], [0.05]]),
    jnp.eye(2),
    jnp.zeros((2, 1)),
    dt=0.05,
)
K = cx.lqr(sys, jnp.eye(2), jnp.array([[1.0]])).K
ts, xs, ys = cx.simulate(
    sys,
    jnp.array([1.0, 0.0]),
    lambda t, x: -K @ x,
    num_steps=80,
)
```

## Horizon Conventions

`simulate()` is explicit about the horizon type:

- use `num_steps=...` for discrete systems
- use `duration=...` for continuous systems

LTI response functions and `simulate` return a [`SimResult`][contrax.types.SimResult]
with `.ts`, `.xs`, `.ys`, `.status`, `.successful` and `.num_steps` fields.
`SimResult` supports tuple unpacking of the trajectory fields:
``ts, xs, ys = cx.simulate(...)``.

For returned trajectories:

Use `plot(result.state_times, result.xs)` for states and
`plot(result.output_times, result.ys)` for outputs. Discrete simulations retain
T outputs and T+1 states: `.state_times` includes the terminal time even when
T is zero. `.ts` aliases output times for tuple unpacking. Continuous results
use the same sample vector for states and outputs. `.final_state` returns the
last state and raises `ValueError` when the trajectory is empty.

- discrete `simulate()` returns `xs` including `x0`
- continuous `simulate()` returns samples on a fixed save grid
- `lsim()` returns a `SimResult` for an open-loop input sequence
- response helpers (`step_response`, `impulse_response`, `initial_response`)
  also return `SimResult` with `.ts`, `.xs`, `.ys`

Discrete simulation and responses accept zero steps: the initial state remains
in `.xs` and the time/output arrays are empty. Continuous `simulate(duration=0)`
returns one initial-state/output sample and performs no integration.

## Continuous Open-Loop Inputs

Continuous LTI `lsim` takes an explicit sample-time vector and one input per
sample. The initial state belongs to `ts[0]`; sample times may be nonuniform.

```python
sys_c = cx.ss([[-1.0]], [[1.0]], [[1.0]], [[0.2]])
times = jnp.array([0.0, 0.1, 0.4, 1.0])
inputs = times[:, None]
response = cx.lsim(sys_c, inputs, ts=times, input_interpolation="foh")
assert response.successful
```

`zoh` holds the left input over each interval. `foh` interpolates linearly
between adjacent inputs. Block matrix exponentials evaluate both exactly up to
floating-point error, including for singular or marginal state matrices. The
continuous result has N states and outputs for N time/input samples; direct
feedthrough uses the input at each sample. The last input contributes to the
final output and to the last FOH interval. Empty sample vectors return empty
continuous states and outputs; one sample returns the initial state/output.

Discrete `lsim` retains T inputs/outputs and T+1 states. It uses the model's
sample period and accepts neither `ts=` nor continuous FOH interpolation.
Both paths support `jit`, `vmap`, JVP and VJP for fixed sample counts.

Use `rollout()` when there is no system object and you simply want to scan a
transition function over a fixed input sequence.
It accepts state and input pytrees, including scalar state leaves, and returns
the state structure with a leading time axis on every leaf. Empty inputs retain
one initial-state sample per leaf.

Use `sample_system()` when the underlying plant model is continuous-time but
the estimation or control workflow lives on a discrete grid. The returned
object is a discrete [`NonlinearSystem`][contrax.systems.NonlinearSystem], so
it plugs directly into `ekf()`, `ukf()`, `simulate(..., num_steps=...)`, and
other discrete-time workflows.

With the default `input_interpolation="zoh"`, each step input has shape `(m,)`
and is held constant across the interval. With
`input_interpolation="foh"`, each step input has shape `(2, m)` and is treated
as the endpoint pair `(u_k, u_{k+1})`; `foh_inputs()` builds that paired
sequence from ordinary sampled inputs.

For sampled models, `model.step(t, x, u)` returns
[`SampleResult`][contrax.types.SampleResult] with terminal `.x`, `.status`,
`.successful` and integration `.num_steps`. Close over the model in `jax.jit`,
or apply `eqx.filter_jit` to its bound `step` method. `model.dynamics(t, x, u)`
returns the state alone and produces a nonfinite state when integration fails.
Sampled numerical status is retained by discrete `simulate` as well.

For a separable Hamiltonian, `sample_system(sys, dt, method="sv")` uses
Störmer–Verlet for the conservative flow and second-order explicit midpoint
for the dissipative/input half-steps. It requires `SeparablePHSSystem` with
the default `J=None`; custom structure maps are rejected. Under FOH the two
external half-steps sample their own midpoint inputs. The full method is
second order for smooth data and supports `jit`, `vmap`, JVP and VJP.
Conservative Verlet preserves the symplectic structure; dissipative/input
steps require a resolved step size and do not guarantee energy inequalities
at arbitrary step sizes. Increase `n_substeps` to check discretization error.

## Transform Behavior

The main transform contracts are:

- discrete `lsim()`, `simulate()`, and `rollout()` are built around
  fixed-shape JAX scans
- continuous `simulate()` uses Diffrax internally but keeps the public output
  on a fixed save grid
- `rollout()` is intended to compose with `jit`, `vmap`, and `grad`

That makes simulation suitable for both ordinary control scripts and compiled
design loops.

## Numerical Notes

`simulate` and `sample_system(method="diffrax")` accept native Diffrax solver
and `stepsize_controller` objects, an initial step `dt0` and `max_steps`.
For example, use `diffrax.PIDController(rtol=1e-8, atol=1e-10)` to set
tolerances and `solver=diffrax.Kvaerno3()` for a validated implicit integration
route. Tsit5 is the default explicit solver. Response helpers forward the same
continuous integration controls. Incompatible discrete/fixed-step options
raise immediately.

`throw=False` returns numerical failure diagnostics. `throw=True` enables
checked failures in eager and compiled execution, including when only a state
or output field is consumed. Iteration limits report `max_steps_reached` even
when some saved values are nonfinite; nonfinite output after a successful
backend reports `numerical_failure`.

The default continuous simulation adjoint is `RecursiveCheckpointAdjoint` for
reverse-mode trajectory losses. Sampled models default to `DirectAdjoint` to
support forward state Jacobians as required by EKF. Replacing an adjoint changes
the supported differentiation modes; in particular, the checkpoint adjoint does
not provide forward-mode JVP. Both explicit Tsit5 and implicit Kvaerno3 have
sampled-Jacobian and reverse-loss coverage. Precision is determined by input
arrays; the library does not change global precision or switch devices.

`step_response()` and `impulse_response()` are analysis helpers. They are good
for inspection and sanity-checking, but they are not the preferred path for
optimization over nonsmooth event timing.

## Related Pages

- [Systems](systems.md) for model construction
- [Control](control.md) for controller design before simulation
- [JAX transform contract](../theory/jax-transform-contract.md) for scan and
  solver behavior under `jit`, `vmap`, and `grad`
- [Linearize, LQR, simulate](../tutorials/linearize-lqr-simulate.md) for the
  main discrete control workflow

::: contrax.simulation
