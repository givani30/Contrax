"""Pure discrete rollout and sampled-input utilities."""

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array


def rollout(
    f: Callable[..., Any],
    x0: Any,
    us: Any,
    params: Any = None,
) -> Any:
    """Roll out arbitrary discrete-time dynamics over a fixed input sequence.

    This is the generic nonlinear trajectory primitive in Contrax. Unlike
    `lsim()` and `simulate()`, it does not require an LTI system object and does
    not compute outputs. It simply applies a transition function with
    `jax.lax.scan`.

    Args:
        f: Transition function. If `params` is omitted, `f(x, u)` is called.
            Otherwise, `f(x, u, params)` is called.
        x0: Initial state. Shape: `(n,)` or any fixed-shape state pytree.
        us: Input sequence. Shape: `(T, m)` or any fixed-leading-axis input
            pytree accepted by `jax.lax.scan`.
        params: Optional static or array-valued parameters passed to `f`.

    Returns:
        PyTree: State trajectory with the same structure as x0. Every leaf
            includes the initial state and has shape `(T + 1, ...)`.

    Examples:
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> def f(x, u, gain):
        ...     return gain * x + u
        >>> xs = cx.rollout(f, jnp.array([1.0]), jnp.zeros((10, 1)), 0.9)
    """

    def step(x, u):
        x_next = f(x, u) if params is None else f(x, u, params)
        return x_next, x_next

    _, xs = jax.lax.scan(step, x0, us)
    return jax.tree.map(
        lambda initial, history: jnp.concatenate(
            [jnp.asarray(initial)[None], history], axis=0
        ),
        x0,
        xs,
    )


def foh_inputs(us: Array) -> Array:
    """Build first-order-hold endpoint pairs from a sample sequence.

    Given inputs `u_k`, returns a sequence with shape `(T, 2, m)` where each
    item contains `(u_k, u_{k+1})`, using `u_{T-1}` for the final right
    endpoint.
    """
    if us.ndim < 2:
        raise ValueError("foh_inputs() expects an input array with leading time axis.")
    return jnp.stack([us, jnp.concatenate([us[1:], us[-1:]], axis=0)], axis=1)
