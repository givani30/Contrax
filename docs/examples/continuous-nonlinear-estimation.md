# Continuous Nonlinear Estimation

This example starts from a simple but realistic estimation problem: the plant is
continuous-time, the measurements arrive on a discrete grid, the input varies
inside each sample interval, and the sensor only reports part of the state.

That combination is where Contrax's continuous-to-discrete estimation bridge is
supposed to matter. The model stays continuous and nonlinear, `sample_system()`
turns it into a discrete transition map, and the UKF/UKS pipeline works on the
measurement grid without application-side glue code.

Runnable script: `examples/continuous_nonlinear_estimation.py`

## Problem Setup

The plant is a damped pendulum with state

$$
x = \begin{bmatrix} \theta \\ \dot{\theta} \end{bmatrix},
$$

input torque `u(t)`, and dynamics

$$
\dot{\theta} = \dot{\theta},
\qquad
\ddot{\theta} = -0.35\dot{\theta} - \sin(\theta) + u(t).
$$

The sensor only measures angle:

$$
y_k = \theta(t_k) + v_k.
$$

So the estimator has to reconstruct angular rate from model structure and the
history of partial observations.

## Build The Continuous Model And FOH Bridge

```python
--8<-- "examples/continuous_nonlinear_estimation.py:setup"
```

Two pieces matter here.

First, the plant remains a continuous `NonlinearSystem`. We are not rewriting it
as an ad hoc discrete transition by hand.

Second, the sampled estimator model uses
`sample_system(..., input_interpolation="foh")`. Each discrete input step is an
endpoint pair `(u_k, u_{k+1})`, so the internal one-step solver sees a
piecewise-linear torque profile instead of a zero-order hold.

## Run Filtering, Smoothing, And Diagnostics

```python
--8<-- "examples/continuous_nonlinear_estimation.py:run-example"
```

The workflow is:

1. simulate the continuous pendulum under a smooth torque profile
2. sample noisy angle measurements on the estimator grid
3. build a discrete estimation model with `sample_system()`
4. run `ukf()` on the sampled measurements and `uks()` on the stored forward-pass intermediates
5. inspect `ukf_diagnostics()` for innovation scale, conditioning, and likelihood summaries

The measurement samples in this script use seeded Gaussian noise, so the example stays reproducible without hard-coding a synthetic offset pattern.

<figure class="contrax-figure">
  <img src="/assets/images/continuous-nonlinear-estimation.svg"
       alt="Continuous nonlinear pendulum estimation example showing the torque input and the true, filtered, and smoothed angle and angular-rate trajectories" />
  <figcaption>
    <strong>Continuous nonlinear estimation with FOH inputs:</strong> the top panel shows the sampled torque profile, the middle panel shows angle measurements against the filtered and smoothed estimates, and the bottom panel shows recovery of the unmeasured angular rate.
  </figcaption>
</figure>

The important behavior is in the bottom panel. Angle is observed directly, but
angular rate is latent. The smoother recovers that hidden state more cleanly
than the forward filter because it can use future measurements as well as past
ones.

## What The Script Prints

Running `examples/continuous_nonlinear_estimation.py` prints a compact summary
of estimation quality:

```text
Continuous nonlinear estimation
filtered theta rmse     = 0.047379
smoothed theta rmse     = 0.029366
filtered rate rmse      = 0.076574
smoothed rate rmse      = 0.039148
mean NIS                = 0.849424
max innovation cond     = 1.000000
total log likelihood    = 38.985644
```

The useful checks are straightforward:

- the smoothed angle RMSE is lower than the filtered angle RMSE
- the smoothed angular-rate RMSE is lower than the filtered angular-rate RMSE
- the innovation covariance stays well-conditioned across the run

## Related Pages

- [Structured nonlinear estimation](structured-nonlinear-estimation.md)
- [JAX-native workflows](jax-native-workflows.md)
- [Simulation API](../api/simulation.md)
- [Estimation API](../api/estimation.md)
