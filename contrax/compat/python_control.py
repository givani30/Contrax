"""contrax.compat.python_control — bidirectional LTI conversion with python-control.

This module is optional. It is only importable when the ``control`` package is
installed. Import it explicitly:

    from contrax.compat.python_control import from_python_control, to_python_control

Importing ``contrax`` does not import this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from contrax.core import ContLTI, DiscLTI

if TYPE_CHECKING:
    try:
        import control  # noqa: F401
    except ImportError:
        pass

__all__ = ["from_python_control", "to_python_control"]


def _require_control() -> "control":
    try:
        import control

        return control
    except ImportError as e:
        raise ImportError(
            "python-control is required for contrax.compat.python_control. "
            "Install it with: pip install control"
        ) from e


def from_python_control(
    sys: "control.StateSpace",
    *,
    dt: float | None = None,
) -> ContLTI | DiscLTI:
    """Convert a python-control StateSpace system to a Contrax LTI system.

    Only state-space (A, B, C, D) conversion is supported. Transfer functions
    and other python-control model types must be converted to StateSpace first.

    Args:
        sys: python-control ``StateSpace`` instance.
        dt: Override the discrete sample time. Required if ``sys.dt`` is
            ``True`` (unspecified dt) or missing. Ignored for continuous
            systems.

    Returns:
        ContLTI | DiscLTI: Equivalent Contrax system.

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
            f"from_python_control() expects a control.StateSpace; got {type(sys).__name__}. "
            "Convert transfer functions with control.ss(sys) first."
        )

    A = jnp.asarray(sys.A, dtype=float)
    B = jnp.asarray(sys.B, dtype=float)
    C = jnp.asarray(sys.C, dtype=float)
    D = jnp.asarray(sys.D, dtype=float)

    # python-control uses sys.dt: 0 or 0.0 for continuous, True for unspecified
    # discrete, or a positive float for discrete with known dt.
    sys_dt = getattr(sys, "dt", 0)

    if sys_dt == 0 or sys_dt is False:
        return ContLTI(A=A, B=B, C=C, D=D)

    if sys_dt is True:
        if dt is None:
            raise ValueError(
                "The python-control system has an unspecified discrete sample "
                "time (dt=True). Provide dt= explicitly."
            )
        return DiscLTI(A=A, B=B, C=C, D=D, dt=jnp.asarray(float(dt)))

    # sys_dt is a positive float
    dt_val = float(dt) if dt is not None else float(sys_dt)
    return DiscLTI(A=A, B=B, C=C, D=D, dt=jnp.asarray(dt_val))


def to_python_control(sys: ContLTI | DiscLTI) -> "control.StateSpace":
    """Convert a Contrax LTI system to a python-control StateSpace object.

    Args:
        sys: Contrax continuous or discrete LTI system.

    Returns:
        control.StateSpace: Equivalent python-control system.

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
