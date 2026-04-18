# SC42095 Reference Workflow

This page collects one of the reference-oriented validation workflows in the
repo: a continuous system, discretized at a fixed sample time, checked against
Octave-backed targets for discretization, pole placement, and discrete LQR.

Runnable script: `examples/sc42095_reference.py`

## Why This Example Exists

Most tutorials show how to use Contrax. This page shows how to trust it on a
small reference problem.

The workflow checks:

- `c2d()` against known discrete matrix entries
- `place()` against a reference gain
- `lqr()` against a reference gain

That makes it a good sanity-check example when you want to verify that a local
edit has not drifted away from an established target.

## The Script

```python
--8<-- "examples/sc42095_reference.py:script"
```

## What The Script Checks

The script starts from a continuous-time realization, discretizes it at
`DT = 0.20039`, and then verifies:

- `Phi[0,0] ≈ 0.6375`
- `Gamma[0,0] ≈ 0.04041`
- the pole-placement gain is close to `[15.0789, 27.1937, 17.7725]`
- the discrete LQR gain is close to `[1.4020, 3.2270, 0.9646]`

These values are tied to the same reference targets used in the test suite.

## What The Script Prints

Running the script prints a compact summary:

```text
SC42095 reference workflow
Phi[0,0]  = 0.637500
Gamma[0,0]= 0.040410
place gain = [[15.0789 27.1937 17.7725]]
lqr gain   = [[1.402  3.227  0.9646]]
```

The exact formatting may vary slightly, but the point is that Contrax agrees
with the established targets within the documented tolerances.

## When To Use This Page

Use this page when you want:

- a trust-building example rather than a first-learning tutorial
- a small reference workflow for regression checking
- a concrete example of Octave-backed validation in the current library

## Related Pages

- [Systems API](../api/systems.md)
- [Control API](../api/control.md)
- [Analysis API](../api/analysis.md)
