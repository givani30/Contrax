"""Public nonlinear system models and helpers."""

from __future__ import annotations

from typing import Callable, Protocol, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


class NonlinearSystem(eqx.Module):
    """General nonlinear system container.

    The reusable model-object contract in Contrax is:

    - `dynamics(t, x, u)` for the state transition or vector field
    - `output(t, x, u)` for the measured/output map

    The constructor keeps the underlying field name `observation` so existing
    pytrees stay stable, while the public constructor prefers the more
    control-oriented `output=` spelling.
    """

    dynamics: Callable
    observation: Callable | None = None
    dt: Array | None = None
    state_dim: int | None = eqx.field(static=True, default=None)
    input_dim: int | None = eqx.field(static=True, default=None)
    output_dim: int | None = eqx.field(static=True, default=None)

    def output(self, t: Array | float, x: Array, u: Array) -> Array:
        """Evaluate the observation map, defaulting to the full state."""
        if self.observation is None:
            return x
        return self.observation(t, x, u)


class SupportsNonlinearSystem(Protocol):
    """Structural protocol for objects implementing the nonlinear-system contract."""

    dt: Array | None

    def dynamics(self, t: Array | float, x: Array, u: Array) -> Array: ...

    def output(self, t: Array | float, x: Array, u: Array) -> Array: ...


DynamicsFn: TypeAlias = Callable[[Array, Array], Array]
ObservationFn: TypeAlias = Callable[[Array], Array]
OutputFn: TypeAlias = Callable[[Array, Array], Array]
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
        state_dim: Optional static state dimension metadata.
        input_dim: Optional static input dimension metadata.
        output_dim: Optional static output dimension metadata.

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
    return callable(dynamics) and callable(output)


def _system_dt(sys: object) -> Array | None:
    """Return the discrete sampling time for a system object, if any."""
    return getattr(sys, "dt", None)


def _coerce_dynamics(
    model_or_f: DynamicsLike,
) -> Callable[[Array, Array, Array | float], Array]:
    """Normalize a system model or plain dynamics function to `(x, u, t)`."""
    if _is_system_model(model_or_f):
        return lambda x, u, t: model_or_f.dynamics(t, x, u)
    return lambda x, u, t: model_or_f(x, u)


def _coerce_observation(
    model_or_h: ObservationLike,
    h: ObservationFn | None = None,
) -> Callable[[Array, Array, Array | float], Array]:
    """Normalize a system model or plain observation function to `(x, u, t)`."""
    if _is_system_model(model_or_h):
        return lambda x, u, t: model_or_h.output(t, x, u)
    if h is None:
        raise ValueError(
            "An observation function is required for nonlinear estimation."
        )
    return lambda x, u, t: h(x)


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
