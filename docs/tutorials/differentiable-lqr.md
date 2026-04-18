# Differentiable LQR

This tutorial shows one of Contrax's defining workflows: optimizing controller
weights inside an ordinary JAX objective.

The idea is simple: treat controller design as part of a larger JAX objective.
Instead of choosing `Q` and `R` offline and then freezing them, make them
parameters in a differentiable loop.

Runnable script: `examples/differentiable_lqr.py`

This tutorial mainly spans the [Control API](../api/control.md) and
[Simulation API](../api/simulation.md). If you want the shorter recipe version,
see [Tune LQR with gradients](../how-to/tune-lqr-with-gradients.md).

## The Workflow

```python
--8<-- "examples/differentiable_lqr.py:setup"

--8<-- "examples/differentiable_lqr.py:objective"

objective_and_grad = jax.jit(jax.value_and_grad(closed_loop_cost, argnums=(0, 1)))
dq, dlog_r = objective_and_grad(
    jnp.zeros(2), jnp.array(0.0)
)[1]
```

The important part is not the syntax. It is the workflow shape: `lqr()` and
`simulate()` inside an ordinary compiled JAX objective with gradients that stay
finite and usable.

At the control level, the script is optimizing the discrete LQR workflow

$$
x_{k+1} = A x_k + B u_k, \qquad u_k = -K x_k
$$

with controller-design weights

$$
Q(\theta_Q) = \operatorname{diag}(\exp(\theta_Q)), \qquad
R(\theta_R) = \exp(\theta_R)
$$

and gain map

$$
K(\theta_Q, \theta_R) =
\operatorname{lqr}\!\left(A, B, Q(\theta_Q), R(\theta_R)\right)
$$

The rollout is then evaluated under a separate closed-loop objective of the
form

$$
J_{\mathrm{cl}}(\theta_Q, \theta_R) =
\sum_{k=0}^{T} x_k^\top x_k +
\lambda_u \sum_{k=0}^{T-1} u_k^\top u_k
$$

where the dependence on $\theta_Q$ and $\theta_R$ enters through
$K(\theta_Q, \theta_R)$ and the resulting closed-loop trajectory.

<figure class="contrax-figure">
  <img src="/assets/images/differentiable-lqr-loop.svg"
       alt="Differentiable LQR loop from log-parameters to weights, lqr solve, closed-loop simulation, scalar loss, and gradient update" />
  <figcaption>
    <strong>The optimization view:</strong> the weights are treated as trainable
    parameters, the Riccati solve lives inside the objective, and gradients
    flow through the full design-and-simulate loop.
  </figcaption>
</figure>

## What The Script Prints

Running `examples/differentiable_lqr.py` prints the initial and final objective,
plus the tuned weights and gain:

```text
Differentiable LQR tuning
initial cost = 29.906683
final cost   = 22.984589
final Q diag = [4.30274642 4.1384593 ]
final R      = 0.056159
final K      = [[6.90505027 7.89914659]]
```

The exact tuned weights depend on the optimization loop configuration. The
important check is simpler: the cost goes down and the gradients stay finite.

## Why This Works

Three pieces matter:

1. `lqr()` on discrete systems goes through `dare()`, which has an
   implicit-differentiation custom VJP rather than a backward pass that unrolls
   the forward solver iterations.
2. `simulate()` is a pure JAX closed-loop scan for discrete systems.
3. the system object is a pytree-friendly Equinox module rather than a class
   with hidden runtime behavior.

That combination keeps controller design, simulation, and optimization in the
same JAX world.

## What the Script Demonstrates

The runnable example in `examples/differentiable_lqr.py` performs a small
gradient-descent loop over log-parameterized `Q` and `R`. The optimization step
itself is ordinary JAX: call the compiled value-and-gradient function, update
the parameters, and repeat.

The script prints:

- initial cost
- final cost
- the final tuned `Q` diagonal
- the final tuned `R`
- the resulting feedback gain

The exact numbers are not the main point. The point is that the optimization
loop is ordinary JAX code rather than a separate control-design procedure.

From a control perspective, the important object is still the closed-loop
matrix

$$
A_{\mathrm{cl}} = A - B K
$$

but the workflow treats `K` as the result of a differentiable design map
$(Q, R) \mapsto K$.

## Validate The Result

For this workflow, the first checks are:

- the final cost should be lower than the initial cost
- the tuned `Q` and `R` should remain positive because they are log-parameterized
- the resulting feedback gain should be finite

The runnable script asserts the cost decrease directly, and the printed values
give a fast sanity check that the optimization stayed in a plausible regime.

## Scope Of This Tutorial

This tutorial intentionally stays on the discrete LTI path. It is the cleanest
path for differentiating through controller design: `lqr()` uses `dare()`, and
the closed-loop rollout uses the pure JAX discrete `simulate()` scan.

Continuous-time LQR and continuous simulation are available through `care()` and
the Diffrax-backed `simulate()` path, but they are a different workflow with
different solver and adjoint choices. Pole placement is also available for
design-time feedback design, but it is not the right primitive for gradient
tuning of `Q` and `R`.

## Where to Go Next

- [Getting started](../getting-started.md) for the fastest route into the library
- [Control API](../api/control.md) for `lqr`, `dare`, and `state_feedback`
- [Simulation API](../api/simulation.md) for discrete closed-loop rollout semantics
- [Linearize, LQR, simulate](linearize-lqr-simulate.md) for the classic control
  flow
- [JAX-native workflows](../examples/jax-native-workflows.md) for the broader pattern map
- [Riccati solvers](../theory/riccati-solvers.md) for the numerical details
