"""Public result-bundle types."""

import equinox as eqx
from jax import Array


class LQRResult(eqx.Module):
    """Result bundle for LQR, DARE, and CARE computations.

    Attributes:
        K: Optimal state-feedback gain. Shape: `(m, n)`.
        S: Riccati solution matrix. Shape: `(n, n)`.
        poles: Closed-loop eigenvalues of the designed system. Shape: `(n,)`.
        residual_norm: Maximum absolute Riccati residual. Shape: scalar.
    """

    K: Array
    S: Array
    poles: Array
    residual_norm: Array


class KalmanResult(eqx.Module):
    """Result bundle for Kalman filter and EKF passes.

    Attributes:
        x_hat: Filtered state estimates over time. Shape: `(T, n)`.
        P: Filtered covariance matrices over time. Shape: `(T, n, n)`.
        innovations: Measurement residual sequence. Shape: `(T, p)`.
    """

    x_hat: Array
    P: Array
    innovations: Array


class UKFResult(eqx.Module):
    """Result bundle for unscented Kalman filtering.

    Attributes:
        x_hat: Filtered state estimates over time. Shape: `(T, n)`.
        P: Filtered covariance matrices over time. Shape: `(T, n, n)`.
        innovations: Measurement residual sequence. Shape: `(T, p)`.
        predicted_measurements: Predicted measurement means before each update.
            Shape: `(T, p)`.
        innovation_covariances: Innovation covariance matrices for each update.
            Shape: `(T, p, p)`.
        log_likelihood_terms: Per-step Gaussian innovation log-likelihood
            terms. Shape: `(T,)`.
        predicted_state_means: One-step predicted state means after each
            update. Shape: `(T, n)`.
        predicted_state_covariances: One-step predicted state covariances after
            each update. Shape: `(T, n, n)`.
        transition_cross_covariances: Cross-covariances between filtered
            states and one-step predictions. Shape: `(T, n, n)`.
    """

    x_hat: Array
    P: Array
    innovations: Array
    predicted_measurements: Array
    innovation_covariances: Array
    log_likelihood_terms: Array
    predicted_state_means: Array
    predicted_state_covariances: Array
    transition_cross_covariances: Array


class KalmanGainResult(eqx.Module):
    """Result bundle for steady-state Kalman estimator design.

    Attributes:
        K: Steady-state measurement-update gain. Shape: `(n, p)`.
        P: Steady-state predicted covariance. Shape: `(n, n)`.
        poles: Estimator error-dynamics eigenvalues. Shape: `(n,)`.
    """

    K: Array
    P: Array
    poles: Array


class RTSResult(eqx.Module):
    """Result bundle for Rauch-Tung-Striebel smoothing.

    Attributes:
        x_smooth: Smoothed state estimates over time. Shape: `(T, n)`.
        P_smooth: Smoothed covariance matrices over time. Shape: `(T, n, n)`.
    """

    x_smooth: Array
    P_smooth: Array


class MHEResult(eqx.Module):
    """Result bundle for a fixed-window moving-horizon-estimation solve.

    Attributes:
        xs: Estimated state trajectory over the window. Shape: `(T, n)`.
        x_hat: Terminal state estimate, equal to `xs[-1]`. Shape: `(n,)`.
            This is the estimate passed to downstream controllers or observers.
        final_cost: Final MHE objective value. Shape: scalar.
        solver_converged: Whether the underlying optimizer reported convergence.
            Shape: scalar bool. Note: LBFGS may report ``False`` even when the
            solution is numerically useful; check ``final_cost`` directly.
    """

    xs: Array
    x_hat: Array
    final_cost: Array
    solver_converged: Array


class PHSStructureDiagnostics(eqx.Module):
    """Structure-oriented diagnostics for a port-Hamiltonian state.

    Attributes:
        skew_symmetry_error: Maximum absolute residual in `J + J.T`.
            Shape: scalar.
        dissipation_symmetry_error: Maximum absolute residual in `R - R.T`.
            Shape: scalar.
        min_dissipation_eigenvalue: Smallest eigenvalue of the symmetrized
            dissipation matrix. Shape: scalar.
        storage_rate: Instantaneous Hamiltonian rate `dH/dt`. Shape: scalar.
        dissipation_power: Dissipated power `grad(H)^T R grad(H)`. Shape:
            scalar.
        supplied_power: Port power `grad(H)^T G u`. Shape: scalar.
        power_balance_residual: Residual in the standard PHS energy balance
            `supplied_power - dissipation_power - storage_rate`. Zero when
            `dH/dt = supply - dissipation` holds exactly. Shape: scalar.
    """

    skew_symmetry_error: Array
    dissipation_symmetry_error: Array
    min_dissipation_eigenvalue: Array
    storage_rate: Array
    dissipation_power: Array
    supplied_power: Array
    power_balance_residual: Array


class SmootherDiagnostics(eqx.Module):
    """Health diagnostics for RTS/UKS smoother output.

    A well-behaved smoother strictly reduces uncertainty: ``P_smooth ≤ P_filtered``
    in the positive-semidefinite sense. These diagnostics surface numerical
    breakdown before it affects downstream analysis.

    Attributes:
        covariance_reduction: Per-step minimum eigenvalue of
            ``P_filtered - P_smooth``. Negative values mean the smoother
            increased variance in some direction at that step.
        min_covariance_reduction: Scalar minimum over time. The primary
            health flag: negative means the smoother violated its own
            uncertainty-reduction guarantee.
        state_corrections: Per-step L2 norm of ``x_smooth - x_filtered``.
            Large corrections relative to filter uncertainty indicate a
            significant prior mismatch or divergence.
        max_state_correction: Scalar maximum correction over time.
        smoothed_min_eigenvalue: Per-step minimum eigenvalue of ``P_smooth``.
            Negative values indicate loss of positive-definiteness.
        nonfinite: True if any smoothed mean or covariance contains NaN/Inf.
    """

    covariance_reduction: Array
    min_covariance_reduction: Array
    state_corrections: Array
    max_state_correction: Array
    smoothed_min_eigenvalue: Array
    nonfinite: Array


class InnovationDiagnostics(eqx.Module):
    """Diagnostics derived from an innovation sequence and covariance model."""

    nis: Array
    innovation_norms: Array
    innovation_cov_condition_numbers: Array
    mean_nis: Array
    max_nis: Array
    mean_innovation_norm: Array
    max_innovation_norm: Array
    max_innovation_cov_condition_number: Array
    nonfinite: Array


class LikelihoodDiagnostics(eqx.Module):
    """Summary diagnostics for innovation-form log-likelihood terms."""

    log_likelihood_terms: Array
    total_log_likelihood: Array
    mean_log_likelihood: Array
    min_log_likelihood: Array
    max_log_likelihood: Array
    nonfinite: Array


__all__ = [
    "LQRResult",
    "KalmanGainResult",
    "KalmanResult",
    "UKFResult",
    "RTSResult",
    "MHEResult",
    "PHSStructureDiagnostics",
    "SmootherDiagnostics",
    "InnovationDiagnostics",
    "LikelihoodDiagnostics",
]
