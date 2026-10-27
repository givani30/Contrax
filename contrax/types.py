"""Public result-bundle types."""

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from contrax._filter_checks import state_status
from contrax._filter_types import (
    FilterPredictionResult,
    FilterUpdateResult,
    KalmanResult,
    UKFResult,
    _final_state,
)


class LyapunovResult(eqx.Module):
    """Solution and diagnostics for a Lyapunov equation or LTI Gramian.

    Attributes:
        X: Equation solution. Shape: `(n, n)`.
        status: Integer `contrax.solvers.SolverStatus` code. Shape: scalar.
        num_steps: Doubling iterations, or the term count for a finite discrete
            Gramian. Zero for an explicit dense solve or finite continuous integral.
        residual_norm: Frobenius norm of the equation residual.
        successful: Whether the validated equation residual meets tolerance.
    """

    X: Array
    status: Array
    num_steps: Array
    residual_norm: Array

    @property
    def successful(self) -> Array:
        return self.status == jnp.asarray(0)


class SimResult(eqx.Module):
    """Result bundle for simulation functions.

    Attributes:
        ts: Output sample times, also exposed as output_times. Shape: `(T,)`
            for discrete, `(N,)` for continuous.
        state_times: Times aligned with xs. Shape: `(T+1,)` for discrete,
            `(N,)` for continuous. Includes the terminal discrete state time.
        xs: State trajectory including the initial state. Shape: `(T+1, n)`
            for discrete, `(N, n)` for continuous.
        ys: Output trajectory. Shape: `(T, p)` or `(N, p)`.
        solver_status: Backend status code; zero when no backend failure occurred.
        status: Backend failure code, or numerical_failure for nonfinite output.
        successful: Whether the backend and output finiteness checks succeeded.
        num_steps: State advances or adaptive integration steps performed.
        final_state: Last state, shape `(n,)`; raises ValueError for empty xs.

    Supports tuple unpacking: ``ts, xs, ys = cx.simulate(...)``
    """

    ts: Array
    xs: Array
    ys: Array
    state_times: Array
    solver_status: Array = eqx.field(default_factory=lambda: jnp.asarray(0))
    num_steps: Array = eqx.field(default_factory=lambda: jnp.asarray(0))

    @property
    def status(self) -> Array:
        finite = (
            jnp.all(jnp.isfinite(self.ts), axis=-1)
            & jnp.all(jnp.isfinite(self.state_times), axis=-1)
            & jnp.all(jnp.isfinite(self.xs), axis=(-2, -1))
            & jnp.all(jnp.isfinite(self.ys), axis=(-2, -1))
        )
        return jnp.where(
            self.solver_status != 0, self.solver_status, jnp.where(finite, 0, 3)
        )

    @property
    def successful(self) -> Array:
        return self.status == 0

    @property
    def output_times(self) -> Array:
        return self.ts

    @property
    def final_state(self) -> Array:
        return _final_state(self.xs)

    def __iter__(self):
        return iter((self.ts, self.xs, self.ys))


class SampleResult(eqx.Module):
    """One discrete transition, including numerical integration diagnostics.

    Attributes:
        x: Terminal state. Shape: `(n,)`.
        status: Integer `contrax.solvers.SolverStatus` code.
        num_steps: Integration steps, or one for an ordinary discrete model.
        successful: Whether the transition finished with finite output.
    """

    x: Array
    status: Array
    num_steps: Array

    @property
    def successful(self) -> Array:
        return (self.status == 0) & jnp.all(jnp.isfinite(self.x), axis=-1)


class PlaceResult(eqx.Module):
    """Result bundle for pole placement.

    Attributes:
        K: State-feedback gain. Shape: `(m, n)`.
        poles: Achieved closed-loop eigenvalues. Shape: `(n,)`.

    Supports tuple unpacking: ``K, poles = cx.place(...)``
    """

    K: Array
    poles: Array

    def __iter__(self):
        return iter((self.K, self.poles))


class LQRResult(eqx.Module):
    """Result bundle for LQR, DARE, and CARE computations.

    Attributes:
        K: Optimal state-feedback gain. Shape: `(m, n)`.
        S: Riccati solution matrix. Shape: `(n, n)`.
        poles: Closed-loop eigenvalues of the designed system. Shape: `(n,)`.
            Evaluated on access; differentiation follows JAX eigenvalue limits.
        closed_loop: Closed-loop state matrix A-BK. Shape: `(n, n)`.
        residual_norm: Maximum absolute Riccati residual. Shape: scalar.
        status: Integer `contrax.solvers.SolverStatus` code. Shape: scalar.
        num_steps: Doubling iterations or accepted CARE refinement corrections.
        successful: Whether the residual, cost and stability checks passed.
    """

    K: Array
    S: Array
    closed_loop: Array
    residual_norm: Array
    status: Array
    num_steps: Array

    @property
    def poles(self) -> Array:
        # Evaluate only when requested: eager pole diagnostics must not restrict
        # higher-order derivatives of objectives that use just K or S.
        return jnp.linalg.eigvals(self.closed_loop)

    @property
    def successful(self) -> Array:
        return self.status == jnp.asarray(0)


class KalmanGainResult(eqx.Module):
    """Result bundle for steady-state Kalman estimator design.

    Attributes:
        K: Steady-state measurement-update gain. Shape: `(n, p)`.
        P: Steady-state predicted covariance. Shape: `(n, n)`.
        poles: Estimator error-dynamics eigenvalues. Shape: `(n,)`.
        closed_loop: Predicted-error transition matrix A(I-KC). Shape: `(n, n)`.
        status: Integer Riccati design status. Shape: scalar.
        residual_norm: Maximum absolute dual Riccati residual.
        num_steps: Iterations used by the dual Riccati design.
        successful: Whether the validated Riccati design succeeded.
    """

    K: Array
    P: Array
    closed_loop: Array
    status: Array
    residual_norm: Array
    num_steps: Array

    @property
    def poles(self) -> Array:
        return jnp.linalg.eigvals(self.closed_loop)

    @property
    def successful(self) -> Array:
        return self.status == jnp.asarray(0)


class RTSResult(eqx.Module):
    """Result bundle for Rauch-Tung-Striebel smoothing.

    Attributes:
        xs: Smoothed state estimates over time. Shape: `(T, n)`.
        Ps: Smoothed covariance matrices over time. Shape: `(T, n, n)`.
        final_state: Last smoothed state; raises ValueError for an empty pass.
        solver_status: Forward-pass or process-noise status, zero on success.
        status: Forward failure or invalid smoothed mean/covariance status.
        successful: Whether the forward and smoothing passes succeeded.
    """

    xs: Array
    Ps: Array
    solver_status: Array = eqx.field(default_factory=lambda: jnp.asarray(0))

    @property
    def status(self) -> Array:
        return state_status(self.xs, self.Ps, self.solver_status, trajectory=True)

    @property
    def successful(self) -> Array:
        return self.status == 0

    @property
    def final_state(self) -> Array:
        return _final_state(self.xs)


class MHEResult(eqx.Module):
    """Result bundle for a fixed-window moving-horizon-estimation solve.

    Attributes:
        xs: Estimated state trajectory over the window. Shape: `(N, n)`.
        final_state: Terminal state estimate, equal to `xs[-1]`. Shape: `(n,)`.
            This is the estimate passed to downstream controllers or observers.
        final_cost: Final MHE objective value. Shape: scalar.
        solver_converged: Whether the underlying optimizer reported convergence.
            Shape: scalar bool. False means convergence was not established;
            inspect status, cost reduction and gradient_norm before use.
        status: Integer `contrax.solvers.SolverStatus` code.
        num_steps: Optimizer iterations attempted.
        gradient_norm: Objective gradient norm at the returned trajectory.
        initial_cost: Objective value at the initial trajectory guess.
        successful: Whether convergence succeeded with a finite trajectory/cost.
    """

    xs: Array
    final_cost: Array
    solver_converged: Array
    status: Array
    num_steps: Array
    gradient_norm: Array
    initial_cost: Array

    @property
    def final_state(self) -> Array:
        return _final_state(self.xs)

    @property
    def successful(self) -> Array:
        return self.status == jnp.asarray(0)


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
    "LyapunovResult",
    "SimResult",
    "SampleResult",
    "PlaceResult",
    "LQRResult",
    "KalmanGainResult",
    "KalmanResult",
    "FilterPredictionResult",
    "FilterUpdateResult",
    "UKFResult",
    "RTSResult",
    "MHEResult",
    "PHSStructureDiagnostics",
    "SmootherDiagnostics",
    "InnovationDiagnostics",
    "LikelihoodDiagnostics",
]
