"""Public nonlinear system models and helpers."""

from __future__ import annotations

from typing import Callable, Protocol, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from contrax.types import SampleResult


class NonlinearSystem(eqx.Module):
    """General nonlinear system container.

    The reusable model-object contract in Contrax is:

    - `dynamics(t, x, u)` for the state transition or vector field
    - `output(t, x, u)` for the measured/output map

    Pass the model directly to estimators to use its dynamics, input-dependent
    output and sample period. Plain recursive-estimator transitions also take
    `(t, x, u)`; passing one separately requires an observation function.
    For discrete models, `step(t, x, u)` returns numerical diagnostics alongside
    the next state. Sampled models retain their integration diagnostics there.
    """

    dynamics: Callable
    observation: Callable | None = None
    dt: Array | None = None
    state_dim: int | None = eqx.field(static=True, default=None)
    input_dim: int | None = eqx.field(static=True, default=None)
    output_dim: int | None = eqx.field(static=True, default=None)
    _step_fn: Callable | None = None
    _input_shape: tuple[int, ...] | None = eqx.field(static=True, default=None)

    def output(self, t: Array | float, x: Array, u: Array) -> Array:
        """Evaluate the observation map, defaulting to the full state."""
        if self.observation is None:
            return x
        return self.observation(t, x, u)

    def step(self, t: Array | float, x: Array, u: Array) -> SampleResult:
        """Evaluate a discrete transition and retain integration diagnostics.

        Close over the model in jax.jit, or use Equinox filter_jit on this bound
        method. For sampled models, the model's integration adjoint determines
        derivative support; ordinary discrete transitions use their JAX graph.

        Args:
            t: Transition start time. Shape: scalar.
            x: Current state. Shape: `(n,)`.
            u: Transition input, or an FOH endpoint pair for a sampled FOH model.

        Returns:
            SampleResult: Next state, integration status and step count.
        """
        if self.dt is None:
            raise ValueError(
                "step requires a discrete system; sample the vector field first"
            )
        if self._step_fn is not None:
            return self._step_fn(t, x, u)
        next_x = self.dynamics(t, x, u)
        return SampleResult(
            next_x, jnp.where(jnp.all(jnp.isfinite(next_x)), 0, 3), jnp.asarray(1)
        )


class SupportsNonlinearSystem(Protocol):
    """Structural protocol for objects implementing the nonlinear-system contract."""

    dt: Array | None

    def dynamics(self, t: Array | float, x: Array, u: Array) -> Array: ...

    def output(self, t: Array | float, x: Array, u: Array) -> Array: ...


DynamicsFn: TypeAlias = Callable[[Array, Array, Array], Array]  # (t, x, u) → x_next
ObservationFn: TypeAlias = Callable[[Array], Array]  # h(x) → y
OutputFn: TypeAlias = Callable[[Array, Array], Array]  # output(x, u) → y
DynamicsLike: TypeAlias = SupportsNonlinearSystem | DynamicsFn
ObservationLike: TypeAlias = SupportsNonlinearSystem | ObservationFn


def nonlinear_system(
    dynamics: Callable,
    output: Callable | None = None,
    *,
    observation: Callable | None = None,
    dt: ArrayLike | None = None,
    state_dim: int | None = None,
    input_dim: int | None = None,
    output_dim: int | None = None,
) -> NonlinearSystem:
    """Construct a reusable nonlinear system model.

    Contrax uses [NonlinearSystem][contrax.systems.NonlinearSystem] as the
    shared contract for nonlinear simulation, linearization, and estimation.
    The intended callable shape is `dynamics(t, x, u)` and, when present,
    `output(t, x, u)`.

    Args:
        dynamics: State transition or vector-field callable with signature
            `(t, x, u)`.
        output: Optional output/measurement map with signature `(t, x, u)`.
            When omitted, the full state is used as the output.
        observation: Deprecated synonym for `output`. Pass only one of
            `output` or `observation`.
        dt: Optional discrete sample time. Omit for continuous models.
        state_dim: Optional static state dimension hint. Contrax does not
            require this — dimensions are inferred from the matrices passed to
            `lqr`, `ekf`, etc. Useful only as documentation metadata.
        input_dim: Optional static input dimension. Enables omitted zero controls
            in recursive estimators; otherwise provide us for controlled models.
        output_dim: Optional static output dimension hint. Same caveat as
            `state_dim`.

    Returns:
        [NonlinearSystem][contrax.systems.NonlinearSystem]: A reusable
            nonlinear system model.
    """
    if output is not None and observation is not None:
        raise ValueError("Pass only one of output= or observation=.")
    observation = output if output is not None else observation
    dt_arr = None if dt is None else jnp.asarray(dt, dtype=float)
    return NonlinearSystem(
        dynamics=dynamics,
        observation=observation,
        dt=dt_arr,
        state_dim=state_dim,
        input_dim=input_dim,
        output_dim=output_dim,
    )


def _is_system_model(obj: object) -> bool:
    """Return whether an object implements the Contrax system contract."""
    dynamics = getattr(obj, "dynamics", None)
    output = getattr(obj, "output", None)
    return (callable(dynamics) and callable(output)) or all(
        hasattr(obj, name) for name in ("A", "B", "C", "D")
    )


def _system_dt(sys: object) -> Array | None:
    """Return the discrete sampling time for a system object, if any."""
    return getattr(sys, "dt", None)


def _coerce_dynamics(
    model_or_f: DynamicsLike,
) -> Callable[[Array | float, Array, Array], Array]:
    """Return a `(t, x, u) → x_next` callable from a system model or plain function."""
    if _is_system_model(model_or_f):
        if _system_dt(model_or_f) is None:
            raise ValueError(
                "Recursive estimators require a discrete model; use sample_system first"
            )
        if hasattr(model_or_f, "A"):
            return lambda t, x, u: model_or_f.A @ x + model_or_f.B @ u
        return model_or_f.dynamics
    return model_or_f  # must be (t, x, u)


def _coerce_observation(
    model_or_h: ObservationLike,
    h: ObservationFn | None = None,
) -> Callable[[Array | float, Array, Array], Array]:
    """Return a `(t, x, u) → y` callable from a system model or plain h(x)."""
    if _is_system_model(model_or_h):
        if hasattr(model_or_h, "C"):
            return lambda t, x, u: model_or_h.C @ x + model_or_h.D @ u
        return model_or_h.output  # already (t, x, u)
    if h is None:
        raise ValueError(
            "An observation function is required for nonlinear estimation."
        )
    return lambda t, x, u: h(x)


__all__ = [
    "DynamicsFn",
    "DynamicsLike",
    "NonlinearSystem",
    "ObservationFn",
    "ObservationLike",
    "OutputFn",
    "SupportsNonlinearSystem",
    "nonlinear_system",
]
