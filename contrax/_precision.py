"""Private helpers for precision-sensitive solver contracts."""

import jax


def require_x64(function_name: str) -> None:
    """Raise if a precision-sensitive public API is used without x64 enabled."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            f"{function_name}() requires jax_enable_x64=True. "
            "Contrax does not enable float64 globally on import. "
            "Set it explicitly in your application before using this solver, "
            "for example:\n"
            "import jax\n"
            "jax.config.update('jax_enable_x64', True)"
        )
