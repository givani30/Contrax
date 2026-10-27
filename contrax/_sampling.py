"""Sample continuous nonlinear systems with explicit integration contracts."""

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from contrax._integration import _diffrax_status
from contrax.nonlinear import NonlinearSystem
from contrax.phs import PHSSystem
from contrax.types import SampleResult


def _continuous_input_fn(
    u: Array,
    tau: Array,
    dt: Array,
    interpolation: str,
) -> Array:
    if interpolation == "zoh":
        return u
    if interpolation == "foh":
        u0 = u[0]
        u1 = u[1]
        weight = tau / dt
        return (1.0 - weight) * u0 + weight * u1
    raise ValueError("input_interpolation must be 'zoh' or 'foh'.")


_FIXED_STEP_METHODS = frozenset({"euler", "midpoint", "rk4", "sv"})


def _fixed_step_dynamics(
    sys: "NonlinearSystem | PHSSystem",
    dt_arr: Array,
    method: str,
    n_substeps: int,
    input_interpolation: str,
) -> Callable:
    """Return a one-step discrete transition using a pure-JAX fixed-step integrator.

    The returned closure composes cleanly with jit and vmap. No Diffrax is
    involved; the integration graph is fully static so XLA sees a single
    fused kernel per call.

    Args:
        sys: Continuous source system. ``"sv"`` requires a SeparablePHSSystem.
        dt_arr: Sampling interval as a JAX scalar.
        method: One of ``"euler"``, ``"midpoint"``, ``"rk4"``, or ``"sv"``.
        n_substeps: Number of fixed sub-steps per sampling interval.
        input_interpolation: ``"zoh"`` or ``"foh"``.
    """

    def dynamics(t: "Array | float", x: Array, u: Array) -> Array:
        t_arr = jnp.asarray(t, dtype=x.dtype)  # noqa: F841 (used inside substep)
        h = dt_arr.astype(x.dtype) / n_substeps

        def substep(x_curr: Array, i: Array) -> tuple[Array, Array]:
            tau = i.astype(x.dtype) * h

            if method == "sv":
                # Reference: Leimkuhler & Reich (2004), "Simulating Hamiltonian
                # Dynamics", Cambridge University Press, Ch. 4.
                # Conservative part: Störmer-Verlet on H = V(q) + T(p).
                # Dissipative/input part: explicit half-steps via Strang splitting
                # (B(h/2) ∘ A(h) ∘ B(h/2)), where A is the symplectic flow and B
                # is the dissipation/input flow. B-steps are only emitted when R
                # or G is present; for purely conservative systems (R=None, G=None)
                # the result is exact Störmer-Verlet with no extra evaluations.
                u_curr = _continuous_input_fn(
                    u, tau, dt_arr.astype(x.dtype), input_interpolation
                )
                n_dofs = sys.n_dofs  # type: ignore[attr-defined]

                def _dissipation_half(x_: Array, h_: Array) -> Array:
                    grad_H = jax.grad(sys.H)(x_)  # type: ignore[attr-defined]
                    dim = x_.shape[0]
                    R_mat = (
                        sys.R(x_)  # type: ignore[attr-defined]
                        if sys.R is not None  # type: ignore[attr-defined]
                        else jnp.zeros((dim, dim), dtype=x_.dtype)
                    )
                    G_mat = (
                        sys.G(x_)  # type: ignore[attr-defined]
                        if sys.G is not None  # type: ignore[attr-defined]
                        else jnp.zeros((dim, u_curr.shape[0]), dtype=x_.dtype)
                    )
                    return x_ + h_ * (-R_mat @ grad_H + G_mat @ u_curr)

                has_dissipation = (
                    sys.R is not None or sys.G is not None  # type: ignore[attr-defined]
                )

                # B(h/2): half-step dissipation/input
                x1 = _dissipation_half(x_curr, 0.5 * h) if has_dissipation else x_curr
                q, p = x1[:n_dofs], x1[n_dofs:]

                # A(h): full Störmer-Verlet conservative step
                grad_V = jax.grad(sys.V)(q)  # type: ignore[attr-defined]
                p_half = p - 0.5 * h * grad_V
                grad_T = jax.grad(sys.T)(p_half)  # type: ignore[attr-defined]
                q_new = q + h * grad_T
                grad_V_new = jax.grad(sys.V)(q_new)  # type: ignore[attr-defined]
                p_new = p_half - 0.5 * h * grad_V_new
                x2 = jnp.concatenate([q_new, p_new])

                # B(h/2): half-step dissipation/input
                x_new = _dissipation_half(x2, 0.5 * h) if has_dissipation else x2

            else:

                def f(x_: Array, tau_: Array) -> Array:
                    u_ = _continuous_input_fn(
                        u, tau_, dt_arr.astype(x.dtype), input_interpolation
                    )
                    return sys.dynamics(t_arr + tau_, x_, u_)

                if method == "euler":
                    x_new = x_curr + h * f(x_curr, tau)
                elif method == "midpoint":
                    k1 = f(x_curr, tau)
                    k2 = f(x_curr + 0.5 * h * k1, tau + 0.5 * h)
                    x_new = x_curr + h * k2
                else:  # rk4
                    k1 = f(x_curr, tau)
                    k2 = f(x_curr + 0.5 * h * k1, tau + 0.5 * h)
                    k3 = f(x_curr + 0.5 * h * k2, tau + 0.5 * h)
                    k4 = f(x_curr + h * k3, tau + h)
                    x_new = x_curr + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            return x_new, jnp.zeros((), dtype=x.dtype)

        x_final, _ = jax.lax.scan(substep, x, jnp.arange(n_substeps, dtype=jnp.int32))
        return x_final

    return dynamics


def sample_system(
    sys: NonlinearSystem | PHSSystem,
    dt: float,
    *,
    input_interpolation: str = "zoh",
    method: str = "diffrax",
    n_substeps: int = 1,
    solver: Any = None,
    adjoint: Any = None,
    dt0: float | None = None,
    stepsize_controller: Any = None,
    max_steps: int = 100_000,
    throw: bool = False,
) -> NonlinearSystem:
    """Sample a continuous nonlinear system into a discrete transition model.

    Builds a discrete-time [NonlinearSystem][contrax.systems.NonlinearSystem]
    whose one-step dynamics integrate the continuous model over one sample
    interval using the chosen integrator.

    With `input_interpolation="zoh"`, each discrete input has shape `(m,)` and
    is held constant over the interval. With `input_interpolation="foh"`, each
    discrete input has shape `(2, m)` and is interpreted as the pair
    `(u_k, u_{k+1})`; use `foh_inputs()` to build those pairs from a sampled
    input sequence.

    **Integrator methods**

    `method="diffrax"` (default) uses adaptive Diffrax integration with Tsit5.
    For stiff dynamics, select an appropriate native implicit solver such as
    Kvaerno3. A native step-size controller sets integration tolerances. Compare
    execution costs at matched trajectory error before choosing a fixed method.

    `method="rk4"` uses a 4th-order Runge-Kutta scheme implemented as pure JAX
    arithmetic. The integration graph is fully static, so XLA compiles the entire
    step to a single fused kernel. 4th-order accurate: local error O(h⁵), global
    error O(h⁴) per sampling interval `dt / n_substeps`. No error control;
    the caller is responsible for choosing `dt` and `n_substeps` appropriate to
    the system's time constants. **Not symplectic** — energy drift accumulates
    over long simulation horizons, which matters for conservative PHS systems.

    `method="sv"` uses Störmer-Verlet with Strang splitting for dissipation
    and inputs (B(h/2) ∘ A(h) ∘ B(h/2), where A is the symplectic flow and B
    is the dissipation/input flow). **Requires a
    [SeparablePHSSystem][contrax.systems.SeparablePHSSystem]** — rejects any
    other system type. For purely conservative systems (R=None, G=None) this
    reduces to exact Störmer-Verlet with no extra gradient evaluations. Per
    substep it evaluates `∇V` twice and `∇T` once, compared to four full
    dynamics evaluations for RK4. **Symplectic** — the conservative part
    preserves phase-space volume exactly regardless of step count; the
    dissipative Strang half-steps are only approximately structure-preserving
    when R or G is nonzero. Reference: Leimkuhler & Reich (2004), *Simulating
    Hamiltonian Dynamics*, Cambridge University Press.

    `method="euler"` and `method="midpoint"` are 1st- and 2nd-order fixed-step
    baselines. They follow the same pure-JAX, no-Diffrax path as `"rk4"`.

    `n_substeps` subdivides `dt` into `n_substeps` equal sub-intervals. Larger
    values improve accuracy at the cost of proportionally more dynamics
    evaluations. Has no effect on `method="diffrax"` (adaptive methods choose
    their own step count).

    `solver`, `adjoint`, `dt0`, `stepsize_controller`, and a nondefault
    `max_steps` are Diffrax-only overrides; incompatible fixed-step
    configurations raise `ValueError`.

    Args:
        sys: Continuous source system. Must have ``dt=None``. ``"sv"`` further
            requires a [SeparablePHSSystem][contrax.systems.SeparablePHSSystem].
        dt: Sampling interval.
        input_interpolation: ``"zoh"`` (zero-order hold) or ``"foh"``
            (first-order hold, requires endpoint-pair inputs from
            `foh_inputs()`).
        method: Integration method. One of ``"diffrax"``, ``"rk4"``,
            ``"midpoint"``, ``"euler"``, or ``"sv"``. Default is ``"diffrax"``.
        n_substeps: Number of fixed sub-steps per sampling interval.
            Ignored by ``method="diffrax"``. Default is ``1``.
        solver: Diffrax solver override. Only valid with ``method="diffrax"``.
        adjoint: Diffrax adjoint override. Only valid with ``method="diffrax"``.
        dt0: Diffrax initial step-size hint. Only valid with ``method="diffrax"``.
        stepsize_controller: Native Diffrax step-size controller, defaulting to
            PIDController(rtol=1e-6, atol=1e-9). Diffrax-only.
        max_steps: Adaptive integration limit. Diffrax-only.
        throw: Raise on failed integration; otherwise step() returns diagnostics
            and dynamics() returns a nonfinite state on failure.

    Returns:
        [NonlinearSystem][contrax.systems.NonlinearSystem]: Discrete system
            with dt set and dynamics obtained from the chosen integrator.
            Its step(t, x, u) method returns SampleResult integration diagnostics.

    Examples:
        >>> import contrax as cx
        >>> import jax.numpy as jnp
        >>> field = cx.nonlinear_system(lambda t, x, u: -x + u)
        >>> sampled = cx.sample_system(field, 0.1, max_steps=256)
        >>> result = sampled.step(0.0, jnp.ones(1), jnp.zeros(1))
        >>> bool(result.successful)
        True
    """
    if sys.dt is not None:
        raise ValueError("sample_system() expects a continuous system with dt=None.")
    if input_interpolation not in {"zoh", "foh"}:
        raise ValueError("input_interpolation must be 'zoh' or 'foh'.")
    _valid_methods = {"diffrax"} | _FIXED_STEP_METHODS
    if method not in _valid_methods:
        raise ValueError(
            f"method must be one of {sorted(_valid_methods)!r}, got {method!r}."
        )
    if method in _FIXED_STEP_METHODS and (
        solver is not None
        or adjoint is not None
        or dt0 is not None
        or stepsize_controller is not None
        or max_steps != 100_000
    ):
        raise ValueError(
            f"solver, adjoint, and dt0 are Diffrax-only options and cannot be "
            f"used with method={method!r}."
        )
    if method == "sv":
        from contrax.phs import SeparablePHSSystem

        if not isinstance(sys, SeparablePHSSystem):
            raise ValueError(
                "method='sv' requires a SeparablePHSSystem with explicit V and T. "
                "Use separable_phs_system() to construct the system, or choose "
                "method='rk4' for a non-symplectic fixed-step alternative."
            )
    if not isinstance(n_substeps, int) or n_substeps < 1:
        raise ValueError(f"n_substeps must be >= 1, got {n_substeps}.")

    dt_arr = jnp.asarray(dt, dtype=float)
    if dt_arr.ndim != 0:
        raise ValueError("sample_system dt must be scalar")
    dt_arr = eqx.error_if(
        dt_arr,
        ~jnp.isfinite(dt_arr) | (dt_arr <= 0),
        "sample_system requires finite dt > 0",
    )
    if not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")

    def output(t: Array | float, x: Array, u: Array) -> Array:
        t = jnp.asarray(t, dtype=x.real.dtype)
        if input_interpolation == "foh":
            return sys.output(t, x, u[0])
        return sys.output(t, x, u)

    if method in _FIXED_STEP_METHODS:
        fixed_transition = _fixed_step_dynamics(
            sys, dt_arr, method, n_substeps, input_interpolation
        )

        def integrate(t, x, u):
            next_x = fixed_transition(t, x, u)
            result = SampleResult(
                next_x,
                jnp.where(jnp.all(jnp.isfinite(next_x)), 0, 3),
                jnp.asarray(n_substeps),
            )
            return (
                eqx.error_if(result, ~result.successful, "Sample integration failed")
                if throw
                else result
            )
    else:
        try:
            import diffrax
        except ImportError:
            raise ImportError("diffrax is required for sample_system(method='diffrax')")

        _solver = diffrax.Tsit5() if solver is None else solver
        _adjoint = diffrax.DirectAdjoint() if adjoint is None else adjoint

        def integrate(t: Array | float, x: Array, u: Array) -> SampleResult:
            local_dt = dt_arr.astype(x.dtype)
            local_dt0 = local_dt if dt0 is None else jnp.asarray(dt0, dtype=x.dtype)

            def vf(tau, x_tau, args):
                base_t, u_step = args
                u_tau = _continuous_input_fn(u_step, tau, local_dt, input_interpolation)
                return sys.dynamics(base_t + tau, x_tau, u_tau)

            solution = diffrax.diffeqsolve(
                diffrax.ODETerm(vf),
                _solver,
                t0=jnp.asarray(0.0, dtype=x.dtype),
                t1=local_dt,
                dt0=local_dt0,
                y0=x,
                args=(jnp.asarray(t, dtype=x.dtype), u),
                saveat=diffrax.SaveAt(t1=True),
                stepsize_controller=(
                    diffrax.PIDController(rtol=1e-6, atol=1e-9)
                    if stepsize_controller is None
                    else stepsize_controller
                ),
                adjoint=_adjoint,
                max_steps=max_steps,
                throw=False,
            )
            result = SampleResult(
                solution.ys[0], _diffrax_status(solution), solution.stats["num_steps"]
            )
            return (
                eqx.error_if(result, ~result.successful, "Sample integration failed")
                if throw
                else result
            )

    def dynamics(t, x, u):
        result = integrate(t, x, u)
        return jnp.where(result.successful, result.x, jnp.full_like(result.x, jnp.nan))

    return NonlinearSystem(
        dynamics=dynamics,
        observation=output,
        dt=dt_arr,
        state_dim=sys.state_dim,
        input_dim=sys.input_dim,
        _input_shape=(2, 0 if sys.input_dim is None else sys.input_dim)
        if input_interpolation == "foh"
        else None,
        output_dim=sys.output_dim,
        _step_fn=integrate,
    )
