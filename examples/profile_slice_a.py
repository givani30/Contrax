"""JAX profiling harness for Slice A: PHS oscillator + UKF/UKS estimation.

Captures XLA compile/execute events via jax.profiler.trace, annotates the
three main sections (DGP, UKF, UKS) with TraceAnnotation, and reports
honest wall-clock times using block_until_ready. Runs both the Diffrax
reference backend and the RK4 fixed-step backend for direct comparison.

Run with XLA compile logging:
    JAX_LOG_COMPILES=1 uv run python examples/profile_slice_a.py

Inspect the trace in the Perfetto UI:
    https://ui.perfetto.dev  (load /tmp/contrax_slice_a/<timestamp>/trace.json.gz)
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_enable_x64", True)

import contrax as cx  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — mirror structured_nonlinear_estimation.py.
# ---------------------------------------------------------------------------
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


def input_at_step(step):
    return INPUTS[jnp.minimum(step, NUM_STEPS - 1)]


# ---------------------------------------------------------------------------
# Annotated workflow sections.
# ---------------------------------------------------------------------------


def _run_dgp(sys_disc: cx.NonlinearSystem) -> tuple:
    with jax.profiler.TraceAnnotation("DGP/simulate_truth"):
        _, xs, _ = cx.simulate(
            sys_disc,
            X0_TRUE,
            lambda t, x: input_at_step(jnp.asarray(t / DT, dtype=jnp.int32)),
            num_steps=NUM_STEPS,
        )
        x_true = xs[:-1]
        jax.block_until_ready(x_true)
    noise = MEASUREMENT_STD * jr.normal(NOISE_KEY, (NUM_STEPS, 1), dtype=X0_TRUE.dtype)
    ys = x_true[:, :1] + noise
    return x_true, ys


def _run_ukf(sys_disc: cx.NonlinearSystem, ys) -> cx.UKFResult:
    with jax.profiler.TraceAnnotation("UKF/forward_pass"):
        result = cx.ukf(
            sys_disc,
            ys,
            us=INPUTS,
            x0=X0_EST,
            P0=P0,
            Q=Q_NOISE,
            R=R_NOISE,
            solver=cx.solvers.UKF(alpha=0.5),
        )
        jax.block_until_ready(result.xs)
    return result


def _run_uks(sys_disc: cx.NonlinearSystem, filtered) -> cx.RTSResult:
    with jax.profiler.TraceAnnotation("UKS/backward_smoothing"):
        result = cx.uks(filtered)
        jax.block_until_ready(result.xs)
    return result


def _run_phs_diagnostics() -> cx.PHSStructureDiagnostics:
    with jax.profiler.TraceAnnotation("PHS/structure_diagnostics"):
        diag = cx.phs_diagnostics(SYS_CONT, X0_TRUE, INPUTS[0])
        jax.block_until_ready(diag.skew_symmetry_error)
    return diag


# ---------------------------------------------------------------------------
# Full timed pass for one backend.
# ---------------------------------------------------------------------------


def _timed_pass(sys_disc: cx.NonlinearSystem, label: str) -> dict:
    timings: dict = {"label": label}

    t0 = time.perf_counter()
    x_true, ys = _run_dgp(sys_disc)
    timings["dgp_ms"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    filtered = _run_ukf(sys_disc, ys)
    timings["ukf_ms"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    smoothed = _run_uks(sys_disc, filtered)
    timings["uks_ms"] = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    _run_phs_diagnostics()
    timings["phs_diag_ms"] = (time.perf_counter() - t0) * 1e3

    innovation_diag, likelihood_diag = cx.ukf_diagnostics(filtered)
    jax.block_until_ready(innovation_diag.mean_nis)

    timings["mean_nis"] = float(innovation_diag.mean_nis)
    timings["log_likelihood"] = float(likelihood_diag.total_log_likelihood)

    filtered_rmse = float(jnp.sqrt(jnp.mean((filtered.xs[:, 0] - x_true[:, 0]) ** 2)))
    smoothed_rmse = float(jnp.sqrt(jnp.mean((smoothed.xs[:, 0] - x_true[:, 0]) ** 2)))
    timings["filtered_q_rmse"] = filtered_rmse
    timings["smoothed_q_rmse"] = smoothed_rmse

    return timings


def _print_timings(t: dict) -> None:
    label = t["label"]
    total = t["dgp_ms"] + t["ukf_ms"] + t["uks_ms"] + t["phs_diag_ms"]
    print(f"  [{label}]")
    print(f"    DGP / simulate_truth  : {t['dgp_ms']:7.2f} ms")
    print(f"    UKF / forward pass    : {t['ukf_ms']:7.2f} ms")
    print(f"    UKS / backward smooth : {t['uks_ms']:7.2f} ms")
    print(f"    PHS / diagnostics     : {t['phs_diag_ms']:7.2f} ms")
    print(f"    Total                 : {total:7.2f} ms")
    print(f"    filtered q RMSE       : {t['filtered_q_rmse']:.6f}")
    print(f"    smoothed q RMSE       : {t['smoothed_q_rmse']:.6f}")
    print(f"    mean NIS              : {t['mean_nis']:.4f}")
    print(f"    total log-likelihood  : {t['log_likelihood']:.4f}")


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

TRACE_DIR = "/tmp/contrax_slice_a"


def main() -> None:
    print("Slice A — PHS oscillator UKF/UKS profiling")
    print(f"  trace output : {TRACE_DIR}")
    print(f"  steps        : {NUM_STEPS}  dt={DT}")
    print()

    sys_diffrax = cx.sample_system(SYS_CONT, DT)
    sys_rk4 = cx.sample_system(SYS_CONT, DT, method="rk4")

    # Warmup: absorb XLA compile cost for both backends before measuring.
    print("  warming up diffrax backend…")
    x_true, ys = _run_dgp(sys_diffrax)
    filtered = _run_ukf(sys_diffrax, ys)
    _run_uks(sys_diffrax, filtered)

    print("  warming up rk4 backend…")
    x_true, ys = _run_dgp(sys_rk4)
    filtered = _run_ukf(sys_rk4, ys)
    _run_uks(sys_rk4, filtered)
    _run_phs_diagnostics()

    print("  warmup done — starting traced passes\n")

    with jax.profiler.trace(TRACE_DIR, create_perfetto_link=False):
        t_diffrax = _timed_pass(sys_diffrax, "diffrax")
        t_rk4 = _timed_pass(sys_rk4, "rk4")

    print("Wall-clock timings (post-warmup, steady-state)")
    _print_timings(t_diffrax)
    print()
    _print_timings(t_rk4)

    total_diffrax = t_diffrax["dgp_ms"] + t_diffrax["ukf_ms"] + t_diffrax["uks_ms"]
    total_rk4 = t_rk4["dgp_ms"] + t_rk4["ukf_ms"] + t_rk4["uks_ms"]
    if total_rk4 > 0:
        speedup = total_diffrax / total_rk4
        print(f"\n  rk4 speedup vs diffrax (DGP+UKF+UKS): {speedup:.1f}×")

    print(f"\nTrace written to {TRACE_DIR}")
    print("Open in Perfetto: https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
