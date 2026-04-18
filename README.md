Contrax
=======

Contrax is a JAX-native systems, estimation, and control toolbox with
MATLAB-familiar names at the API surface and JAX-first behavior underneath.

Contrax is not a clone of MATLAB or `python-control`. It is built for control
workflows that remain differentiable, batchable, and compilable inside the same
JAX program as the rest of your model.

Public namespaces follow the user mental model:

- `contrax.systems`
- `contrax.control`
- `contrax.estimation`
- `contrax.simulation`
- `contrax.analysis`
- `contrax.types`

That includes workflows such as:

- differentiate through `lqr` and closed-loop simulation,
- linearize nonlinear dynamics directly into state-space form,
- simulate nonlinear and structured systems with the same JAX-first workflow,
- run filtering, smoothing, and fixed-horizon estimation inside JAX,
- `vmap` controller design across operating points,
- keep design and simulation in one `jit`-compiled workflow.

## Why Contrax

Contrax is built for control workflows that belong inside a larger JAX program.
A controller design step can sit inside an ordinary objective instead of being a
separate offline calculation:

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import contrax as cx

A = jnp.array([[1.0, 0.05], [0.0, 1.0]])
B = jnp.array([[0.0], [0.05]])
C = jnp.eye(2)
D = jnp.zeros((2, 1))
SYS = cx.dss(A, B, C, D, dt=0.05)
X0 = jnp.array([1.0, 0.0])


def closed_loop_cost(log_q_diag, log_r):
    q_diag = jnp.exp(log_q_diag)
    r = jnp.exp(log_r)[None, None]
    K = cx.lqr(SYS, jnp.diag(q_diag), r).K
    _, xs, _ = cx.simulate(SYS, X0, lambda t, x: -K @ x, num_steps=80)
    control_energy = jnp.sum((xs[:-1] @ K.T) ** 2)
    return jnp.sum(xs**2) + 1e-2 * control_energy


objective_and_grad = jax.jit(jax.value_and_grad(closed_loop_cost, argnums=(0, 1)))
cost, (dq, dlog_r) = objective_and_grad(
    jnp.zeros(2), jnp.array(0.0)
)
```

This is the central Contrax idea: control primitives that behave like normal
JAX building blocks.

## Scope

The main public surface includes:

- `ss`, `dss`, `c2d`
- `nonlinear_system`, `phs_system`, `canonical_J`, `schedule_phs`
- `series`, `parallel`
- `linearize`, `linearize_ss`
- `rollout`
- `lsim`, `simulate`, `step_response`, `impulse_response`, `initial_response`
- `lqr`, `dare`, `care`, `place`, `state_feedback`
- `kalman`, `ekf`, `ukf`, `rts`, `uks`
- `kalman_predict`, `kalman_update`, `kalman_step`
- `ekf_predict`, `ekf_update`, `ekf_step`
- `steady_state_kalman`, `augment_integrator`, `lqi`
- `mhe_objective`, `mhe`
- `ctrb`, `obsv`, `poles`, `evalfr`, `freqresp`, `dcgain`

Numerical outputs are validated against Octave where possible. JAX behavior is
validated with `jit`, `vmap`, and `grad` tests, and runnable examples under
`examples/` are smoke-tested.

LTI workflows are the most mature slice. Nonlinear models, PHS support, and
fixed-window MHE are real public capabilities, but they should still be read
with more explicit solver-maturity caution than the core discrete design path.

## Solver Status

Contrax is explicit about solver maturity:

- `dare` is the most mature Riccati solver path: structured doubling forward
  solve with an implicit-differentiation custom VJP.
- `care` is a validated continuous-time solver using a Hamiltonian
  stable-subspace solve with an implicit backward pass. It is less benchmarked
  than `dare`, but it is a real public solver path.
- `place` uses JAX-native KNV0/YT-style robust assignment paths for
  design-time SISO and MIMO pole placement, with Ackermann retained only as a
  small SISO fallback.
- `simulate` supports discrete closed-loop simulation and a Diffrax-backed
  continuous closed-loop path with fixed sampled outputs.
- `mhe()` is a useful fixed-window estimation primitive, but it is not yet a
  broad constrained optimization framework and still needs stronger nonlinear
  examples.

## Docs

Start here:

- `docs/getting-started.md` for the fastest path to a working system
- `docs/tutorials/differentiable-lqr.md` for an end-to-end differentiable design workflow
- `docs/tutorials/linearize-lqr-simulate.md` for the core control path
- `docs/tutorials/continuous-lqr.md` for the continuous-time LQR path
- `docs/tutorials/kalman-filtering.md` for a complete estimation workflow
- `examples/` for runnable scripts mirrored by tests

## Development

Install and test with `uv`:

```bash
uv sync --group dev
uv run pre-commit install
uv run pre-commit run --all-files
uv run pytest tests/ -q
uv run python -m build
```

See `BUILD_PLAN.md` for the roadmap and `DOCUMENTATION_CONTRACT.md` for the
public docs contract.
