"""Host bridge parity against python-control's transfer evaluation."""

import numpy as np
import pytest

import contrax as cx
from contrax.compat.python_control import from_python_control, to_python_control


def test_state_space_bridge_timebases_and_transfer():
    control = pytest.importorskip("control")
    matrices = ([[-0.5]], [[2.0]], [[3.0]], [[0.25]])
    for period in (0, 0.2):
        original = control.ss(*matrices, dt=period)
        converted = from_python_control(original)
        restored = to_python_control(converted)
        assert restored.dt == pytest.approx(period)
        for name in ("A", "B", "C", "D"):
            np.testing.assert_array_equal(
                getattr(converted, name), getattr(original, name)
            )
        for point in (0.3j, 1.2 + 0.1j):
            expected = control.evalfr(original, point)
            np.testing.assert_allclose(control.evalfr(restored, point), expected)
    for unspecified in (True, None):
        original = control.ss(*matrices, dt=unspecified)
        with pytest.raises(ValueError, match="Provide dt"):
            from_python_control(original)
        assert float(from_python_control(original, dt=0.4).dt) == pytest.approx(0.4)
        for invalid in (True, -1, np.nan, np.inf, [0.1], 0.1j):
            with pytest.raises(ValueError):
                from_python_control(original, dt=invalid)
    unspecified = control.ss(*matrices, dt=None)
    assert isinstance(from_python_control(unspecified, dt=0), cx.ContLTI)
    with pytest.raises(TypeError, match="StateSpace"):
        from_python_control(control.tf([1], [1, 1]))
