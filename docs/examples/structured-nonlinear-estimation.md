# Structured Nonlinear Estimation

This example shows what the structured nonlinear layer is for in practice: keep
the model in a port-Hamiltonian form, observe only part of the state, and still
run the same recursive-estimation workflow as any other nonlinear system.

The central estimation question here is simple: can a position-only sensor and a
structured model recover hidden momentum cleanly enough that smoothing adds
visible value?

Runnable script: `examples/structured_nonlinear_estimation.py`

## Problem Setup

The state is `(q, p)`, where `q` is configuration and `p` is momentum. The
system is a damped, forced port-Hamiltonian oscillator with Hamiltonian

$$
H(q, p) = \tfrac{1}{2}(1.2q^2 + p^2).
$$

The structure map `J` is the canonical skew-symmetric interconnection, the
dissipation map `R(x)` damps momentum, and the input map injects force only into
the momentum channel. The measurement is position-only:

$$
y_k = q_k + v_k.
$$

So the hidden quantity is exactly the one the structured dynamics care most
about: momentum.

## Build The Structured Model

```python
--8<-- "examples/structured_nonlinear_estimation.py:setup"
```

The useful parts of the public surface are all visible here:

- `phs_system()` defines the continuous structured model
- `block_observation()` builds the position-only measurement map without custom indexing glue
- `sample_system()` turns the continuous structured dynamics into a discrete estimation model

## Run Filtering, Smoothing, Diagnostics, And Structure Checks

```python
--8<-- "examples/structured_nonlinear_estimation.py:run-example"
```

This example goes slightly beyond “run a filter and print a state.” It also uses
`phs_diagnostics()` to evaluate the local structure contract on the same model
that drives the estimator. The measurement samples use seeded Gaussian noise so the run stays reproducible while still reading like a real estimation problem.

<figure class="contrax-figure">
  <img src="/assets/images/structured-nonlinear-estimation.svg"
       alt="Structured nonlinear estimation example showing input forcing and the true, filtered, and smoothed position and momentum trajectories" />
  <figcaption>
    <strong>Position-only observation of a PHS oscillator:</strong> the filter tracks configuration well from the measured channel, while smoothing substantially improves the hidden momentum estimate.
  </figcaption>
</figure>

The middle and bottom panels tell the story. Position is the measured state, so
filtered and smoothed `q` stay close to the truth. Momentum is not measured at
all, so `p` is the stronger structural test.

## What The Script Prints

Running `examples/structured_nonlinear_estimation.py` prints:

```text
Structured nonlinear estimation
filtered q rmse         = 0.020113
smoothed q rmse         = 0.013961
filtered p rmse         = 0.104952
smoothed p rmse         = 0.017463
mean NIS                = 0.780241
max innovation cond     = 1.000000
total log likelihood    = 39.698811
skew residual           = 0.000000e+00
min dissipation eig     = 0.000000
```

The useful checks are:

- smoothing lowers both position and momentum RMSE
- the innovation covariance remains well-conditioned
- the structure diagnostics stay consistent with the model definition

## Related Pages

- [Continuous nonlinear estimation](continuous-nonlinear-estimation.md)
- [JAX-native workflows](jax-native-workflows.md)
- [Systems API](../api/systems.md)
- [Estimation API](../api/estimation.md)
