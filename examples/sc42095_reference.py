# --8<-- [start:script]
"""SC42095-style reference workflow with Octave-backed targets."""

from __future__ import annotations

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import contrax as cx

DT = 0.20039
A_CONT = jnp.array([[-2.2, -0.4, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
B_CONT = jnp.array([[0.25], [0.0], [0.0]])
C_CONT = jnp.array([[0.0, 0.0, 0.4]])
D_CONT = jnp.zeros((1, 1))


def desired_poles(omega: float = 1.0, zeta: float = 0.6901, h: float = DT):
    s = -zeta * omega + 1j * omega * np.sqrt(1.0 - zeta**2)
    p1 = np.exp(s * h)
    p2 = np.conj(p1)
    p3 = 0.2 * abs(p1)
    return np.array([p1, p2, p3])


def run_example():
    sys_c = cx.ss(A_CONT, B_CONT, C_CONT, D_CONT)
    sys_d = cx.c2d(sys_c, DT)

    phi00 = float(sys_d.A[0, 0])
    gamma00 = float(sys_d.B[0, 0])
    place_gain = cx.place(sys_d, desired_poles())
    lqr_result = cx.lqr(sys_d, jnp.eye(3), jnp.array([[1.0]]))

    # Octave reference values from tests/test_core.py and tests/test_control.py.
    assert abs(phi00 - 0.6375) < 1e-4
    assert abs(gamma00 - 0.04041) < 1e-4
    assert jnp.allclose(
        place_gain,
        jnp.array([[15.0789, 27.1937, 17.7725]]),
        atol=0.1,
    )
    assert jnp.allclose(
        lqr_result.K,
        jnp.array([[1.4020, 3.2270, 0.9646]]),
        atol=0.05,
    )

    return {
        "phi00": phi00,
        "gamma00": gamma00,
        "place_gain": np.asarray(place_gain),
        "lqr_gain": np.asarray(lqr_result.K),
    }


def main():
    result = run_example()
    print("SC42095 reference workflow")
    print(f"Phi[0,0]  = {result['phi00']:.6f}")
    print(f"Gamma[0,0]= {result['gamma00']:.6f}")
    print(f"place gain = {result['place_gain']}")
    print(f"lqr gain   = {result['lqr_gain']}")


if __name__ == "__main__":
    main()
# --8<-- [end:script]
