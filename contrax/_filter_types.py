"""Result bundles for recursive estimators."""

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from contrax._filter_checks import state_status


def _final_state(xs: Array) -> Array:
    if xs.shape[-2] == 0:
        raise ValueError("An empty trajectory has no final_state")
    return xs[..., -1, :]


class FilterPredictionResult(eqx.Module):
    """Mean, covariance, and status from one Kalman prediction step.

    Attributes:
        x: Predicted mean. Shape: `(n,)`.
        P: Predicted covariance. Shape: `(n, n)`.
        solver_status: Input or backend status code.
        status: Status after checking the mean and covariance.
        successful: Whether the prediction passed validation.
    """

    x: Array
    P: Array
    solver_status: Array

    @property
    def status(self) -> Array:
        return state_status(self.x, self.P, self.solver_status)

    @property
    def successful(self) -> Array:
        return self.status == 0

    def __iter__(self):
        return iter((self.x, self.P))


class FilterUpdateResult(eqx.Module):
    """Mean, covariance, innovation, and status from one Kalman update.

    Attributes:
        x: Updated mean. Shape: `(n,)`.
        P: Updated covariance. Shape: `(n, n)`.
        innovation: Measurement residual. Shape: `(p,)`; zero when the
            measurement is missing.
        solver_status: Input or backend status code.
        status: Status after checking the mean, covariance, and innovation.
        successful: Whether the update passed validation.
    """

    x: Array
    P: Array
    innovation: Array
    solver_status: Array

    @property
    def status(self) -> Array:
        status = state_status(self.x, self.P, self.solver_status)
        finite = jnp.all(jnp.isfinite(self.innovation), axis=-1)
        return jnp.where(status != 0, status, jnp.where(finite, 0, 3))

    @property
    def successful(self) -> Array:
        return self.status == 0

    def __iter__(self):
        return iter((self.x, self.P, self.innovation))


class KalmanResult(eqx.Module):
    """Result bundle for Kalman filter and EKF passes.

    Attributes:
        xs: Filtered state estimates over time. Shape: `(T, n)`.
        Ps: Filtered covariance matrices over time. Shape: `(T, n, n)`.
        innovations: Measurement residual sequence. Shape: `(T, p)`.
        has_measurement: Availability mask. Shape: `(T,)`.
        final_state: Last filtered state; raises ValueError for an empty pass.
        solver_status: Input or backend status, zero on success.
        status: Input/backend failure or invalid numerical output status.
        successful: Whether finite means and valid covariances were returned.
    """

    xs: Array
    Ps: Array
    innovations: Array
    has_measurement: Array
    solver_status: Array = eqx.field(default_factory=lambda: jnp.asarray(0))

    @property
    def status(self) -> Array:
        status = state_status(self.xs, self.Ps, self.solver_status, trajectory=True)
        finite = jnp.all(jnp.isfinite(self.innovations), axis=(-2, -1))
        return jnp.where(status != 0, status, jnp.where(finite, 0, 3))

    @property
    def successful(self) -> Array:
        return self.status == 0

    @property
    def final_state(self) -> Array:
        return _final_state(self.xs)


class UKFResult(eqx.Module):
    """Result bundle for unscented Kalman filtering.

    Attributes:
        xs: Filtered state estimates over time. Shape: `(T, n)`.
        Ps: Filtered covariance matrices over time. Shape: `(T, n, n)`.
        final_state: Last filtered state; raises ValueError for an empty pass.
        innovations: Measurement residual sequence. Shape: `(T, p)`.
        has_measurement: Availability mask. Shape: `(T,)`.
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
        solver_status: Input or backend status, zero on success.
        status: Input/backend failure or invalid output/prediction status.
        successful: Whether the validated filtering/prediction pass succeeded.
    """

    xs: Array
    Ps: Array
    innovations: Array
    has_measurement: Array
    predicted_measurements: Array
    innovation_covariances: Array
    log_likelihood_terms: Array
    predicted_state_means: Array
    predicted_state_covariances: Array
    transition_cross_covariances: Array
    solver_status: Array = eqx.field(default_factory=lambda: jnp.asarray(0))

    @property
    def status(self) -> Array:
        status = state_status(self.xs, self.Ps, self.solver_status, trajectory=True)
        finite = jnp.all(jnp.isfinite(self.log_likelihood_terms), axis=-1)
        finite = finite & jnp.all(jnp.isfinite(self.innovations), axis=(-2, -1))
        predictions = state_status(
            self.predicted_state_means,
            self.predicted_state_covariances,
            self.solver_status,
            trajectory=True,
        )
        valid = finite & (predictions == 0)
        return jnp.where(status != 0, status, jnp.where(valid, 0, 3))

    @property
    def successful(self) -> Array:
        return self.status == 0

    @property
    def final_state(self) -> Array:
        return _final_state(self.xs)
