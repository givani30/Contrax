# Kalman Filtering

This tutorial shows a standard estimation workflow on a small discrete system.
We will define a state-space model, run a linear Kalman filter on a measurement
sequence, smooth the result with RTS, and finish with checks that tell us
whether the estimate is behaving plausibly.

<figure class="contrax-figure">
  <img src="/assets/images/estimation-pipeline.svg"
       alt="Estimation pipeline from model and measurements through filtering, smoothing, and MHE" />
  <figcaption>
    <strong>The tutorial path:</strong> start from a discrete state-space
    model, run the batch Kalman filter, then use RTS to revisit the same
    sequence with future information available.
  </figcaption>
</figure>

Runnable script: `examples/kalman_filtering.py`

This tutorial sits primarily in the [Estimation API](../api/estimation.md),
with [Systems](../api/systems.md) providing the underlying discrete model.

The model and measurement equations are

$$
x_{k+1} = A x_k + w_k, \qquad
y_k = C x_k + v_k
$$

with Gaussian noise covariances `Q_noise` and `R_noise`. The batch filter uses
the update-first convention:

$$
\hat{x}_k^+ = \hat{x}_k^- + K_k\bigl(y_k - C\hat{x}_k^-\bigr), \qquad
\hat{x}_{k+1}^- = A\hat{x}_k^+
$$

and RTS adds the backward pass

$$
\hat{x}_k^{\,s} = \hat{x}_k^{\,f} + G_k\bigl(\hat{x}_{k+1}^{\,s} - \hat{x}_{k+1}^{-}\bigr)
$$

## Define The Estimation Problem

The model is a constant-velocity discrete system with position-only
measurements. That is a familiar place to start: the hidden state contains both
position and velocity, but the sensor only reports position.

```python
--8<-- "examples/kalman_filtering.py:setup"
```

## Run The Filter And Smoother

`kalman()` runs the batch discrete-time filter as a `jax.lax.scan`, so the full
pass stays in JAX. `rts()` then applies an offline backward smoothing pass to
the filtered trajectory using the same process model and process noise.

```python
--8<-- "examples/kalman_filtering.py:filter-and-smooth"
```

In Contrax, the filter treats `(x0, P0)` as the prior on `x_0` and updates
first with `y_0`. That convention matters when you compare the batch result to
the one-step `kalman_update()` and `kalman_step()` helpers in runtime loops.

## What The Script Prints

Running `examples/kalman_filtering.py` prints a compact summary of the final
estimate:

```text
Kalman filtering and RTS smoothing
final measurement        = 1.010000
mid filtered position    = 1.060639
mid smoothed position    = 1.003812
final filtered position  = 1.024387
final smoothed position  = 1.024387
final filtered velocity  = 0.055978
innovation norm          = 0.932813
final covariance trace   = 0.007035
```

The exact numbers are not the point. The useful checks are whether the filter
produces a state estimate consistent with the measurements, keeps covariance
bounded, and shows the role of smoothing clearly: the interior smoothed state
can differ from the filtered one, while the terminal RTS state matches the
final filtered state by construction.

## Validate The Result

For a first estimation workflow, the checks should be easy to interpret:

- the filtered state shape should match the measurement horizon and state size
- the final filtered position should be close to the last measurement
- the covariance trace should stay finite and reasonably small
- the innovation sequence should not blow up

The runnable example mirrors those checks with assertions so this tutorial stays
tied to executable code rather than drifting into static documentation.

## Where To Go Next

- [Getting started](../getting-started.md) for the fastest route into the library
- [Estimation API reference](../api/estimation.md) for batch and one-step filter helpers
- [Handle missing measurements](../how-to/handle-missing-measurements.md) for runtime loop behavior when sensors drop out
- [Build an MHE objective](../how-to/build-mhe-objective.md) for the optimization-based estimation sibling workflow
- [Estimation pipelines](../theory/estimation-pipelines.md) for how filters, smoothers, and MHE fit together
- [JAX transform contract](../theory/jax-transform-contract.md) for scan, `jit`, and differentiation expectations
