"""contrax._estimation_diagnostics — estimation diagnostics and health checks."""

import jax
import jax.numpy as jnp
from jax import Array

from contrax.types import (
    InnovationDiagnostics,
    KalmanResult,
    LikelihoodDiagnostics,
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


__all__ = [
    "InnovationDiagnostics",
    "LikelihoodDiagnostics",
    "innovation_diagnostics",
    "innovation_rms",
    "likelihood_diagnostics",
    "ukf_diagnostics",
]
