"""contrax.interconnect — series and parallel LTI interconnections."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax import core as jax_core

from contrax.core import ContLTI, DiscLTI


def _is_tracing(*arrays: Array) -> bool:
    return any(isinstance(arr, jax_core.Tracer) for arr in arrays)


def _check_same_family(sys2: DiscLTI | ContLTI, sys1: DiscLTI | ContLTI) -> None:
    if isinstance(sys2, DiscLTI) and isinstance(sys1, DiscLTI):
        if not _is_tracing(sys2.dt, sys1.dt) and not np.allclose(
            np.asarray(sys2.dt), np.asarray(sys1.dt)
        ):
            raise ValueError("Discrete interconnections require matching dt.")
        return
    if isinstance(sys2, ContLTI) and isinstance(sys1, ContLTI):
        return
    raise TypeError(
        "Interconnection requires both systems to be continuous or both discrete."
    )


def series(sys2: DiscLTI | ContLTI, sys1: DiscLTI | ContLTI) -> DiscLTI | ContLTI:
    """Connect two systems in series as `sys2 @ sys1`.

    The output of `sys1` feeds the input of `sys2`. State ordering follows the
    MATLAB-style convention: `sys1` states first, then `sys2` states.

    Args:
        sys2: Downstream system.
        sys1: Upstream system.

    Returns:
        [DiscLTI][contrax.systems.DiscLTI] |
            [ContLTI][contrax.systems.ContLTI]: Interconnected system.
    """
    _check_same_family(sys2, sys1)
    if sys1.C.shape[0] != sys2.B.shape[1]:
        raise ValueError(
            "Series interconnection requires sys1 output dimension to match "
            "sys2 input dimension."
        )

    A1, B1, C1, D1 = sys1.A, sys1.B, sys1.C, sys1.D
    A2, B2, C2, D2 = sys2.A, sys2.B, sys2.C, sys2.D
    n1, n2 = A1.shape[0], A2.shape[0]

    A = jnp.block(
        [
            [A1, jnp.zeros((n1, n2), dtype=A1.dtype)],
            [B2 @ C1, A2],
        ]
    )
    B = jnp.vstack([B1, B2 @ D1])
    C = jnp.hstack([D2 @ C1, C2])
    D = D2 @ D1

    if isinstance(sys1, DiscLTI):
        return DiscLTI(A=A, B=B, C=C, D=D, dt=sys1.dt)
    return ContLTI(A=A, B=B, C=C, D=D)


def parallel(
    sys1: DiscLTI | ContLTI,
    sys2: DiscLTI | ContLTI,
    *,
    sign: float = 1.0,
) -> DiscLTI | ContLTI:
    """Connect two systems in parallel and sum their outputs.

    Args:
        sys1: First system.
        sys2: Second system.
        sign: Output sign applied to `sys2`. Use `-1.0` for subtraction.

    Returns:
        [DiscLTI][contrax.systems.DiscLTI] |
            [ContLTI][contrax.systems.ContLTI]: Interconnected system.
    """
    _check_same_family(sys1, sys2)
    if sys1.B.shape[1] != sys2.B.shape[1]:
        raise ValueError("Parallel interconnection requires matching input dimensions.")
    if sys1.C.shape[0] != sys2.C.shape[0]:
        raise ValueError(
            "Parallel interconnection requires matching output dimensions."
        )

    A1, B1, C1, D1 = sys1.A, sys1.B, sys1.C, sys1.D
    A2, B2, C2, D2 = sys2.A, sys2.B, sys2.C, sys2.D
    n1, n2 = A1.shape[0], A2.shape[0]
    sign_arr = jnp.asarray(sign, dtype=A1.dtype)

    A = jnp.block(
        [
            [A1, jnp.zeros((n1, n2), dtype=A1.dtype)],
            [jnp.zeros((n2, n1), dtype=A2.dtype), A2],
        ]
    )
    B = jnp.vstack([B1, B2])
    C = jnp.hstack([C1, sign_arr * C2])
    D = D1 + sign_arr * D2

    if isinstance(sys1, DiscLTI):
        return DiscLTI(A=A, B=B, C=C, D=D, dt=sys1.dt)
    return ContLTI(A=A, B=B, C=C, D=D)
