# --8<-- [start:script]
"""Structured nonlinear estimation with a sampled port-Hamiltonian oscillator."""

from __future__ import annotations

import jax

# --8<-- [start:setup]
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr

import contrax as cx

DT = 0.1
DURATION = 3.0
NUM_STEPS = int(DURATION / DT)
NOISE_KEY = jr.PRNGKey(17)
X0_TRUE = jnp.array([1.0, -0.15])
X0_EST = jnp.array([0.75, 0.2])
P0 = 0.12 * jnp.eye(2)
Q_NOISE = 2e-4 * jnp.eye(2)
R_NOISE = jnp.array([[3e-3]])
MEASUREMENT_STD = jnp.sqrt(R_NOISE[0, 0])
SAMPLE_TIMES = jnp.arange(NUM_STEPS + 1, dtype=jnp.float64) * DT
INPUT_TIMES = SAMPLE_TIMES[:-1]
INPUTS = (0.18 * jnp.sin(1.1 * INPUT_TIMES) + 0.08 * jnp.cos(0.55 * INPUT_TIMES))[
    :, None
]


def hamiltonian(x):
    q, p = x
    return 0.5 * (1.2 * q**2 + p**2)


OBSERVE_POSITION = cx.block_observation((1, 1), (0,))

SYS_CONT = cx.phs_system(
    hamiltonian,
    R=lambda x: jnp.array([[0.0, 0.0], [0.0, 0.14]], dtype=x.dtype),
    G=lambda x: jnp.array([[0.0], [1.0]], dtype=x.dtype),
    output=OBSERVE_POSITION,
    state_dim=2,
    input_dim=1,
    output_dim=1,
)
SYS_DISC = cx.sample_system(SYS_CONT, DT)
# --8<-- [end:setup]


def input_at_step(step):
    return INPUTS[jnp.minimum(step, NUM_STEPS - 1)]


def simulate_truth():
    _, xs, _ = cx.simulate(
        SYS_DISC,
        X0_TRUE,
        lambda t, x: input_at_step(jnp.asarray(t / DT, dtype=jnp.int32)),
        num_steps=NUM_STEPS,
    )
    return xs[:-1]


def sample_measurements(x_true):
    noise = MEASUREMENT_STD * jr.normal(NOISE_KEY, (NUM_STEPS, 1), dtype=x_true.dtype)
    return x_true[:, :1] + noise


def summarize_rmse(filtered, smoothed, truth):
    filtered_q_rmse = float(
        jnp.sqrt(jnp.mean((filtered.x_hat[:, 0] - truth[:, 0]) ** 2))
    )
    smoothed_q_rmse = float(
        jnp.sqrt(jnp.mean((smoothed.x_smooth[:, 0] - truth[:, 0]) ** 2))
    )
    filtered_p_rmse = float(
        jnp.sqrt(jnp.mean((filtered.x_hat[:, 1] - truth[:, 1]) ** 2))
    )
    smoothed_p_rmse = float(
        jnp.sqrt(jnp.mean((smoothed.x_smooth[:, 1] - truth[:, 1]) ** 2))
    )
    return {
        "filtered_q_rmse": filtered_q_rmse,
        "smoothed_q_rmse": smoothed_q_rmse,
        "filtered_p_rmse": filtered_p_rmse,
        "smoothed_p_rmse": smoothed_p_rmse,
    }


# --8<-- [start:run-example]
def run_example():
    x_true = simulate_truth()
    ys = sample_measurements(x_true)

    filtered = cx.ukf(
        SYS_DISC,
        Q_noise=Q_NOISE,
        R_noise=R_NOISE,
        ys=ys,
        us=INPUTS,
        x0=X0_EST,
        P0=P0,
        alpha=0.5,
    )
    smoothed = cx.uks(SYS_DISC, filtered, Q_noise=Q_NOISE, us=INPUTS, alpha=0.5)
    innovation_diag, likelihood_diag = cx.ukf_diagnostics(filtered)
    structure_diag = cx.phs_diagnostics(SYS_CONT, X0_TRUE, INPUTS[0])
    rmse = summarize_rmse(filtered, smoothed, x_true)

    assert filtered.x_hat.shape == (NUM_STEPS, 2)
    assert filtered.predicted_measurements.shape == (NUM_STEPS, 1)
    assert smoothed.x_smooth.shape == (NUM_STEPS, 2)
    assert rmse["smoothed_q_rmse"] < rmse["filtered_q_rmse"]
    assert rmse["smoothed_p_rmse"] < rmse["filtered_p_rmse"]

    return {
        "times": INPUT_TIMES,
        "inputs": INPUTS[:, 0],
        "measurements": ys[:, 0],
        "true_q": x_true[:, 0],
        "filtered_q": filtered.x_hat[:, 0],
        "smoothed_q": smoothed.x_smooth[:, 0],
        "true_p": x_true[:, 1],
        "filtered_p": filtered.x_hat[:, 1],
        "smoothed_p": smoothed.x_smooth[:, 1],
        **rmse,
        "mean_nis": float(innovation_diag.mean_nis),
        "max_condition_number": float(
            innovation_diag.max_innovation_cov_condition_number
        ),
        "total_log_likelihood": float(likelihood_diag.total_log_likelihood),
        "skew_symmetry_residual": float(structure_diag.skew_symmetry_error),
        "dissipation_min_eigenvalue": float(structure_diag.min_dissipation_eigenvalue),
    }


# --8<-- [end:run-example]


def main():
    result = run_example()
    print("Structured nonlinear estimation")
    print(f"filtered q rmse         = {result['filtered_q_rmse']:.6f}")
    print(f"smoothed q rmse         = {result['smoothed_q_rmse']:.6f}")
    print(f"filtered p rmse         = {result['filtered_p_rmse']:.6f}")
    print(f"smoothed p rmse         = {result['smoothed_p_rmse']:.6f}")
    print(f"mean NIS                = {result['mean_nis']:.6f}")
    print(f"max innovation cond     = {result['max_condition_number']:.6f}")
    print(f"total log likelihood    = {result['total_log_likelihood']:.6f}")
    print(f"skew residual           = {result['skew_symmetry_residual']:.6e}")
    print(f"min dissipation eig     = {result['dissipation_min_eigenvalue']:.6f}")


if __name__ == "__main__":
    main()
# --8<-- [end:script]
