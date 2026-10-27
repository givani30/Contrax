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


def _observations(values, has_measurement):
    flags = (
        jnp.ones(values.shape[0], dtype=bool)
        if has_measurement is None
        else jnp.asarray(has_measurement, dtype=bool)
    )
    if flags.shape != (values.shape[0],):
        raise ValueError("has_measurement must match the sequence length")
    return flags


def _masked_summary(values, flags):
    count = jnp.sum(flags)
    mean = jnp.sum(jnp.where(flags, values, 0)) / jnp.maximum(count, 1)
    minimum = jnp.min(jnp.where(flags, values, jnp.inf), initial=jnp.inf)
    maximum = jnp.max(jnp.where(flags, values, -jnp.inf), initial=-jnp.inf)
    return tuple(jnp.where(count > 0, v, jnp.nan) for v in (mean, minimum, maximum))


def innovation_diagnostics(
    innovations: Array,
    innovation_covariances: Array,
    *,
    has_measurement: Array | None = None,
) -> InnovationDiagnostics:
    """Summarize innovation magnitude and covariance conditioning.

    The returned NIS values are the per-step normalized innovation squared
    terms `v_k^T S_k^{-1} v_k`. They are useful for routine filter health
    inspection, but Contrax does not present them as a formal standalone
    consistency certificate. Optional whole-observation flags exclude missing
    entries from summaries and covariance solves. Missing per-step values are
    zero placeholders. With no observations, summaries are NaN and nonfinite
    remains false: absence of data is not a numerical failure.
    """
    flags = _observations(innovations, has_measurement)
    innovations = jnp.where(flags[:, None], innovations, 0)
    innovation_covariances = jnp.where(
        flags[:, None, None],
        innovation_covariances,
        jnp.eye(innovations.shape[-1], dtype=innovations.dtype),
    )

    def step_nis(innovation, covariance):
        solved = jnp.linalg.solve(covariance, innovation[:, None])
        return (innovation[None, :] @ solved).squeeze()

    nis = jax.vmap(step_nis)(innovations, innovation_covariances)
    innovation_norms = jnp.linalg.norm(innovations, axis=-1)
    innovation_cov_condition_numbers = jnp.where(
        flags, jax.vmap(jnp.linalg.cond)(innovation_covariances), 0
    )
    mean_nis, _, max_nis = _masked_summary(nis, flags)
    mean_norm, _, max_norm = _masked_summary(innovation_norms, flags)
    _, _, max_condition = _masked_summary(innovation_cov_condition_numbers, flags)
    all_values = (
        jnp.isfinite(nis)
        & jnp.isfinite(innovation_norms)
        & jnp.isfinite(innovation_cov_condition_numbers)
    )
    return InnovationDiagnostics(
        nis=nis,
        innovation_norms=innovation_norms,
        innovation_cov_condition_numbers=innovation_cov_condition_numbers,
        mean_nis=mean_nis,
        max_nis=max_nis,
        mean_innovation_norm=mean_norm,
        max_innovation_norm=max_norm,
        max_innovation_cov_condition_number=max_condition,
        nonfinite=jnp.logical_not(jnp.all(all_values)),
    )


def likelihood_diagnostics(
    log_likelihood_terms: Array,
    *,
    has_measurement: Array | None = None,
) -> LikelihoodDiagnostics:
    """Summarize observed log likelihoods, excluding optional missing entries.

    Missing terms are zero. With no observations, total is zero and mean/min/max
    are NaN; nonfinite flags numerical failure only in observed entries.
    """
    flags = _observations(log_likelihood_terms, has_measurement)
    log_likelihood_terms = jnp.where(flags, log_likelihood_terms, 0)
    mean, minimum, maximum = _masked_summary(log_likelihood_terms, flags)
    return LikelihoodDiagnostics(
        log_likelihood_terms=log_likelihood_terms,
        total_log_likelihood=jnp.sum(log_likelihood_terms),
        mean_log_likelihood=mean,
        min_log_likelihood=minimum,
        max_log_likelihood=maximum,
        nonfinite=jnp.logical_not(jnp.all(jnp.isfinite(log_likelihood_terms))),
    )


def ukf_diagnostics(
    result: UKFResult,
) -> tuple[InnovationDiagnostics, LikelihoodDiagnostics]:
    """Build the standard diagnostics pair for a UKF result."""
    return (
        innovation_diagnostics(
            result.innovations,
            result.innovation_covariances,
            has_measurement=result.has_measurement,
        ),
        likelihood_diagnostics(
            result.log_likelihood_terms,
            has_measurement=result.has_measurement,
        ),
    )


def innovation_rms(result: KalmanResult | UKFResult) -> Array:
    """Return observed innovation RMS; NaN if there are no observed components."""
    observed = jnp.where(result.has_measurement[:, None], result.innovations, 0)
    count = jnp.sum(result.has_measurement) * result.innovations.shape[-1]
    return jnp.where(
        count > 0, jnp.sqrt(jnp.sum(observed**2) / jnp.maximum(count, 1)), jnp.nan
    )


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
        >>> filtered = cx.kalman(
        ...     sys, jnp.zeros((10, 1)), Q=1e-3 * jnp.eye(1), R=1e-2 * jnp.eye(1)
        ... )
        >>> smoothed = cx.rts(sys, filtered, Q=0.001 * jnp.eye(1))
        >>> diag = cx.smoother_diagnostics(smoothed, filtered)
    """
    x_smooth = smoothed.xs  # (T, n)
    P_smooth = smoothed.Ps  # (T, n, n)
    x_filt = filtered.xs  # (T, n)
    P_filt = filtered.Ps  # (T, n, n)

    def _sym(M: Array) -> Array:
        return (M + M.T) / 2

    # Min eigenvalue of (P_f - P_s): >= 0 means smoother reduced uncertainty.
    cov_reduction = jax.vmap(
        lambda Pf, Ps: jnp.min(jnp.linalg.eigvalsh(_sym(Pf) - _sym(Ps)))
    )(P_filt, P_smooth)

    state_corrections = jnp.linalg.norm(x_smooth - x_filt, axis=-1)

    smoothed_min_eig = jax.vmap(lambda Ps: jnp.min(jnp.linalg.eigvalsh(_sym(Ps))))(
        P_smooth
    )

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
