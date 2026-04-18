# --8<-- [start:script]
"""Linear Kalman filtering and RTS smoothing on a small discrete system."""

from __future__ import annotations

import jax

# --8<-- [start:setup]

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import contrax as cx

SYS = cx.dss(
    A=jnp.array([[1.0, 0.1], [0.0, 1.0]]),
    B=jnp.zeros((2, 1)),
    C=jnp.array([[1.0, 0.0]]),
    D=jnp.zeros((1, 1)),
    dt=0.1,
)
Q_NOISE = jnp.diag(jnp.array([1e-4, 5e-4]))
R_NOISE = jnp.array([[2.5e-3]])
YS = jnp.array(
    [
        [0.92],
        [0.95],
        [0.99],
        [1.03],
        [1.06],
        [1.05],
        [1.02],
        [1.00],
        [0.98],
        [1.01],
    ]
)
X0 = jnp.array([0.0, 0.0])
P0 = jnp.eye(2)
# --8<-- [end:setup]


# --8<-- [start:filter-and-smooth]
def run_example():
    filtered = cx.kalman(
        SYS,
        Q_noise=Q_NOISE,
        R_noise=R_NOISE,
        ys=YS,
        x0=X0,
        P0=P0,
    )
    smoothed = cx.rts(SYS, filtered, Q_noise=Q_NOISE)

    final_measurement = float(YS[-1, 0])
    final_filtered = float(filtered.x_hat[-1, 0])
    final_smoothed = float(smoothed.x_smooth[-1, 0])
    midpoint_index = 4
    midpoint_filtered = float(filtered.x_hat[midpoint_index, 0])
    midpoint_smoothed = float(smoothed.x_smooth[midpoint_index, 0])
    innovation_norm = float(jnp.linalg.norm(filtered.innovations))

    assert filtered.x_hat.shape == (10, 2)
    assert filtered.P.shape == (10, 2, 2)
    assert smoothed.x_smooth.shape == (10, 2)
    assert abs(final_filtered - final_measurement) < 0.1
    assert innovation_norm < 1.5

    return {
        "final_measurement": final_measurement,
        "final_filtered_position": final_filtered,
        "final_smoothed_position": final_smoothed,
        "midpoint_filtered_position": midpoint_filtered,
        "midpoint_smoothed_position": midpoint_smoothed,
        "final_velocity": float(filtered.x_hat[-1, 1]),
        "innovation_norm": innovation_norm,
        "filtered_cov_trace": float(jnp.trace(filtered.P[-1])),
    }


# --8<-- [end:filter-and-smooth]


def main():
    result = run_example()
    print("Kalman filtering and RTS smoothing")
    print(f"final measurement        = {result['final_measurement']:.6f}")
    print(f"mid filtered position    = {result['midpoint_filtered_position']:.6f}")
    print(f"mid smoothed position    = {result['midpoint_smoothed_position']:.6f}")
    print(f"final filtered position  = {result['final_filtered_position']:.6f}")
    print(f"final smoothed position  = {result['final_smoothed_position']:.6f}")
    print(f"final filtered velocity  = {result['final_velocity']:.6f}")
    print(f"innovation norm          = {result['innovation_norm']:.6f}")
    print(f"final covariance trace   = {result['filtered_cov_trace']:.6f}")


if __name__ == "__main__":
    main()
# --8<-- [end:script]
