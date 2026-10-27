"""Bidirectional LTI conversion with python-control.

Import this optional module when the ``control`` package is installed:

    from contrax.compat.python_control import from_python_control, to_python_control

The root package loads this bridge only on explicit import.
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

import jax.numpy as jnp

from contrax.core import ContLTI, DiscLTI, dss, ss

if TYPE_CHECKING:
    try:
        import control  # noqa: F401
    except ImportError:
        pass

__all__ = ["from_python_control", "to_python_control"]


def _require_control() -> ModuleType:
    try:
        import control

        return control
    except ImportError as e:
        raise ImportError(
            "python-control is required for contrax.compat.python_control. "
            "Install it with: uv add control"
        ) from e


def from_python_control(
    sys: "control.StateSpace",
    *,
    dt: float | None = None,
) -> ContLTI | DiscLTI:
    """Convert a python-control StateSpace system to an LTI system.

    Converts state-space matrices `(A, B, C, D)`. Convert transfer functions
    and other model types to `StateSpace` before calling.

    Args:
        sys: python-control ``StateSpace`` instance.
        dt: Override a discrete sample time with a finite positive real scalar.
            Required for dt=True or dt=None. For dt=None, dt=0 selects a
            continuous model. Ignored for an explicitly continuous source.

    Returns:
        ContLTI | DiscLTI: Equivalent continuous- or discrete-time system.

    Notes:
        Host conversion only; not jit-, vmap- or differentiation-compatible.
        Matrix precision follows JAX's configured dtype support; enable x64
        before conversion to retain double precision source arrays.

    Raises:
        TypeError: If ``sys`` is not a ``control.StateSpace``.
        ValueError: If ``sys`` is discrete with unspecified sample time and
            ``dt`` is not provided.

    Examples:
        >>> import control
        >>> import contrax.compat.python_control as cx_compat
        >>> pc_sys = control.ss([[-1]], [[1]], [[1]], [[0]])
        >>> cx_sys = cx_compat.from_python_control(pc_sys)
        >>> isinstance(cx_sys, cx_compat.ContLTI)
        True
    """
    control = _require_control()
    if not isinstance(sys, control.StateSpace):
        raise TypeError(
            f"from_python_control() expects a control.StateSpace; "
            f"got {type(sys).__name__}. "
            "Convert transfer functions with control.ss(sys) first."
        )

    A, B, C, D = (jnp.asarray(value) for value in (sys.A, sys.B, sys.C, sys.D))

    # python-control uses sys.dt: 0 or 0.0 for continuous, True for unspecified
    # discrete, or a positive float for discrete with known dt.
    sys_dt = sys.dt

    if sys_dt == 0 or sys_dt is False:
        return ss(A, B, C, D)

    if sys_dt is True or sys_dt is None:
        if dt is None:
            raise ValueError(
                "The python-control system has an unspecified discrete sample "
                "time (dt=True or dt=None). Provide dt= explicitly."
            )
    period = sys_dt if dt is None else dt
    raw_period = jnp.asarray(period)
    if (
        raw_period.ndim != 0
        or jnp.iscomplexobj(raw_period)
        or jnp.issubdtype(raw_period.dtype, jnp.bool_)
    ):
        raise ValueError("dt must be a real scalar")
    if sys_dt is None and float(raw_period) == 0:
        return ss(A, B, C, D)
    return dss(A, B, C, D, period)


def to_python_control(sys: ContLTI | DiscLTI) -> "control.StateSpace":
    """Convert an LTI system to a python-control StateSpace object.

    Args:
        sys: Continuous or discrete LTI system.

    Returns:
        control.StateSpace: Equivalent python-control system.

    Notes:
        Copies arrays to the host. This bridge does not support JAX transforms.

    Raises:
        TypeError: If ``sys`` is not a ContLTI or DiscLTI.

    Examples:
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> import contrax.compat.python_control as cx_compat
        >>> lti = cx.ss(jnp.array([[-1.0]]), jnp.array([[1.0]]),
        ...             jnp.array([[1.0]]), jnp.zeros((1, 1)))
        >>> pc = cx_compat.to_python_control(lti)
    """
    control = _require_control()
    if not isinstance(sys, (ContLTI, DiscLTI)):
        raise TypeError(
            f"to_python_control() expects ContLTI or DiscLTI; got {type(sys).__name__}."
        )

    import numpy as np

    A = np.asarray(sys.A)
    B = np.asarray(sys.B)
    C = np.asarray(sys.C)
    D = np.asarray(sys.D)

    if isinstance(sys, ContLTI):
        return control.ss(A, B, C, D)

    dt = float(sys.dt)
    return control.ss(A, B, C, D, dt)
