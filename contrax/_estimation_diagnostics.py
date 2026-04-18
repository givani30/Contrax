"""contrax._estimation_diagnostics — estimation diagnostics and health checks."""

import jax
import jax.numpy as jnp
from jax import Array

from contrax.types import (
    InnovationDiagnostics,
    KalmanResult,
    LikelihoodDiagnostics,
    RTSResult,
    SmootherDiagnostics,
    UKFResult,
)


def innovation_diagnostics(
    innovations: Array,
    innovation_covariances: Array,
) -> InnovationDiagnostics:
    """Summarize innovation magnitude and covariance conditioning.

    The returned NIS values are the per-step normalized innovation squared
    terms `v_k^T S_k^{-1} v_k`. They are useful for routine filter health
    inspection, but Contrax does not present them as a formal standalone
    consistency certificate.
    """

    def step_nis(innovation, covariance):
        solved = jnp.linalg.solve(covariance, innovation[:, None])
        return (innovation[None, :] @ solved).squeeze()

    nis = jax.vmap(step_nis)(innovations, innovation_covariances)
    innovation_norms = jnp.linalg.norm(innovations, axis=-1)
    innovation_cov_condition_numbers = jax.vmap(jnp.linalg.cond)(innovation_covariances)
    all_values = (
        jnp.isfinite(nis)
        & jnp.isfinite(innovation_norms)
        & jnp.isfinite(innovation_cov_condition_numbers)
    )
    return InnovationDiagnostics(
        nis=nis,
        innovation_norms=innovation_norms,
        innovation_cov_condition_numbers=innovation_cov_condition_numbers,
        mean_nis=jnp.mean(nis),
        max_nis=jnp.max(nis),
        mean_innovation_norm=jnp.mean(innovation_norms),
        max_innovation_norm=jnp.max(innovation_norms),
        max_innovation_cov_condition_number=jnp.max(innovation_cov_condition_numbers),
        nonfinite=jnp.logical_not(jnp.all(all_values)),
    )


def likelihood_diagnostics(log_likelihood_terms: Array) -> LikelihoodDiagnostics:
    """Summarize per-step innovation-form log-likelihood terms."""
    return LikelihoodDiagnostics(
        log_likelihood_terms=log_likelihood_terms,
        total_log_likelihood=jnp.sum(log_likelihood_terms),
        mean_log_likelihood=jnp.mean(log_likelihood_terms),
        min_log_likelihood=jnp.min(log_likelihood_terms),
        max_log_likelihood=jnp.max(log_likelihood_terms),
        nonfinite=jnp.logical_not(jnp.all(jnp.isfinite(log_likelihood_terms))),
    )


def ukf_diagnostics(
    result: UKFResult,
) -> tuple[InnovationDiagnostics, LikelihoodDiagnostics]:
    """Build the standard diagnostics pair for a UKF result."""
    return (
        innovation_diagnostics(result.innovations, result.innovation_covariances),
        likelihood_diagnostics(result.log_likelihood_terms),
    )


def innovation_rms(result: KalmanResult | UKFResult) -> Array:
    """Return the RMS innovation magnitude for a filtered result."""
    return jnp.sqrt(jnp.mean(result.innovations**2))


def smoother_diagnostics(
    smoothed: RTSResult,
    filtered: KalmanResult | UKFResult,
) -> SmootherDiagnostics:
    """Compute health diagnostics for an RTS or UKS smoother result.

    A numerically healthy smoother satisfies ``P_smooth ≤ P_filtered`` in the
    PSD sense at every step. This function checks that guarantee and surfaces
    the worst-case violation so callers can detect breakdown before trusting
    smoothed trajectories.

    Args:
        smoothed: Output of `rts()` or `uks()`.
        filtered: The forward filter result used to produce ``smoothed``.
            Accepts both `KalmanResult` and `UKFResult`.

    Returns:
        [SmootherDiagnostics][contrax.types.SmootherDiagnostics]: Diagnostic
            bundle. ``min_covariance_reduction < 0`` is the primary health flag.

    Examples:
        >>> import jax.numpy as jnp
        >>> import contrax as cx
        >>> sys = cx.dss(
        ...     jnp.array([[0.9]]), jnp.zeros((1, 1)),
        ...     jnp.array([[1.0]]), jnp.zeros((1, 1)), dt=1.0,
        ... )
        >>> filtered = cx.kalman(sys, 1e-3 * jnp.eye(1), 1e-2 * jnp.eye(1),
        ...                      jnp.zeros((10, 1)))
        >>> smoothed = cx.rts(sys, filtered, Q_noise=1e-3 * jnp.eye(1))
        >>> diag = cx.smoother_diagnostics(smoothed, filtered)
    """
    x_smooth = smoothed.x_smooth   # (T, n)
    P_smooth = smoothed.P_smooth   # (T, n, n)
    x_filt = filtered.x_hat        # (T, n)
    P_filt = filtered.P            # (T, n, n)

    def _sym(M: Array) -> Array:
        return (M + M.T) / 2

    # Min eigenvalue of (P_f - P_s): >= 0 means smoother reduced uncertainty.
    cov_reduction = jax.vmap(
        lambda Pf, Ps: jnp.min(jnp.linalg.eigvalsh(_sym(Pf) - _sym(Ps)))
    )(P_filt, P_smooth)

    state_corrections = jnp.linalg.norm(x_smooth - x_filt, axis=-1)

    smoothed_min_eig = jax.vmap(
        lambda Ps: jnp.min(jnp.linalg.eigvalsh(_sym(Ps)))
    )(P_smooth)

    nonfinite = jnp.logical_not(
        jnp.all(jnp.isfinite(x_smooth)) & jnp.all(jnp.isfinite(P_smooth))
    )

    return SmootherDiagnostics(
        covariance_reduction=cov_reduction,
        min_covariance_reduction=jnp.min(cov_reduction),
        state_corrections=state_corrections,
        max_state_correction=jnp.max(state_corrections),
        smoothed_min_eigenvalue=smoothed_min_eig,
        nonfinite=nonfinite,
    )


__all__ = [
    "InnovationDiagnostics",
    "LikelihoodDiagnostics",
    "SmootherDiagnostics",
    "innovation_diagnostics",
    "innovation_rms",
    "likelihood_diagnostics",
    "smoother_diagnostics",
    "ukf_diagnostics",
]
