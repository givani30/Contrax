# LQR Optimal Execution

This example starts from a simple liquidation problem: a trader has a position
to sell and wants to work out of it over a fixed horizon without waiting too
long or trading too aggressively.

In a quadratic execution model, those two pressures become the usual LQR
tradeoff:

- penalize remaining inventory because holding it is risky
- penalize large trades because fast execution is expensive

The result is a one-state discrete control problem with a very clean
state-space interpretation.

Runnable script: `examples/lqr_optimal_execution.py`

## Problem Setup

Let `x_k` denote remaining inventory at step `k`, normalized so `x_0 = 1`
means the full order is still unsold. Let `u_k` denote the signed inventory
change over one step. The dynamics are

$$
x_{k+1} = x_k + u_k.
$$

If the controller sells, inventory goes down, so `u_k < 0`. The finance-facing
sell quantity is therefore

$$
q_k^{\mathrm{sell}} = x_k - x_{k+1} = -u_k.
$$

The design objective is the standard infinite-horizon LQR cost

$$
J = \sum_{k=0}^{\infty}
\left(x_k^\top Q x_k + u_k^\top R u_k\right),
$$

where `Q` represents inventory risk and `R` represents trading cost. Increasing
`Q` pushes the controller to liquidate faster. Increasing `R` makes it trade
more slowly.

## Build The Execution Model

```python
--8<-- "examples/lqr_optimal_execution.py:setup"
```

This is a tiny model, but it already shows the useful part of the Contrax API:
the execution problem is just a `DiscLTI` system plus an LQR solve.

## Solve The Baseline Schedule

```python
--8<-- "examples/lqr_optimal_execution.py:baseline-execution"
```

For the baseline choice `Q = 2.5` and `R = 0.4`, the controller is strongly
inventory-averse, so it sells most of the position immediately and then cleans
up the remainder very quickly.

<figure class="contrax-figure">
  <img src="/assets/images/lqr-optimal-execution.svg"
       alt="Baseline LQR execution inventory path over twenty time steps, compared with a smoother reference urgency curve" />
  <figcaption>
    <strong>The baseline execution path:</strong> the teal curve is the LQR
    liquidation schedule, and the orange curve is a smoother target profile
    used later in the tuning section.
  </figcaption>
</figure>

That plot is the center of the example. Inventory is the state, liquidation is
the control effect, and the design question is the familiar balance between
state penalty and control penalty.

## Tune The Execution Urgency With Gradients

The same script then places the Riccati solve inside a JAX objective:

```python
--8<-- "examples/lqr_optimal_execution.py:differentiable-tuning"
```

Here the goal is to tune `Q` and `R` so the resulting inventory path tracks a
chosen urgency curve while still keeping turnover and terminal inventory under
control.

That gives the workflow

$$
(\theta_Q, \theta_R)
\longrightarrow
\bigl(Q(\theta_Q), R(\theta_R)\bigr)
\longrightarrow
\operatorname{lqr}(A, B, Q, R)
\longrightarrow
\text{inventory path}
\longrightarrow
\text{loss}.
$$

This is the part that feels especially native to Contrax: the controller
design step is not a separate offline calculation. It lives inside the same
differentiable JAX program as the rest of the objective.

## Batch The Same Design Across Many Assets

Once the execution problem is written as an ordinary fixed-shape control
workflow, batching becomes just another `vmap`:

```python
--8<-- "examples/lqr_optimal_execution.py:batched-design"
```

That is a natural extension of the same story. Instead of solving one execution
schedule, solve many independent schedules with different risk and impact
weights in one compiled pass.

## What The Script Prints

Running `examples/lqr_optimal_execution.py` prints a compact summary of the
baseline controller and the tuned design:

```text
LQR optimal execution
baseline gain          = [[0.87695257]]
initial tuning loss    = 0.052402
final tuning loss      = 0.046743
tuned inventory risk   = 0.116998
tuned trading cost     = 0.156547
first sell quantities  = [8.76952567e-01 1.07906744e-01 1.32776339e-02 ...]
batched first sells    = [0.65586909 0.87695257 0.96291202]
```

The useful checks are simple:

- the baseline controller produces a monotone liquidation path
- the tuning loop lowers its objective
- the batched version returns different schedules for different weight choices

## Read This Example For What It Is

This page is intentionally about the control mapping, not about full execution
microstructure realism. The model is linear, quadratic, single-asset, and
deliberately small.

That is also what makes it useful here. You can see the state, the control,
the cost, and the feedback law immediately, and then see how Contrax extends
that classical setup with differentiation and batching.

## Related Pages

- [JAX-native workflows](jax-native-workflows.md)
- [Differentiable LQR](../tutorials/differentiable-lqr.md)
- [Tune LQR with gradients](../how-to/tune-lqr-with-gradients.md)
- [Control API](../api/control.md)
- [Simulation API](../api/simulation.md)
