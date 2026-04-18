# Contributing to Contrax

## Development setup

Contrax uses `uv` for environment management.

```bash
git clone https://github.com/givani30/Contrax.git
cd Contrax
uv sync --group dev
```

## Running tests

```bash
uv run pytest tests/ -q
```

All tests must pass before submitting a pull request. Tests are numerical —
key solvers are checked against Octave where possible.

## Code style

Ruff handles formatting and linting:

```bash
uv run ruff check .
uv run ruff format .
```

Install pre-commit hooks to run these automatically:

```bash
uv run pre-commit install
```

## JAX rules

- Enable float64 before any computation.
- Use `jax.lax.scan` for time loops, not Python `for`.
- No `jax.scipy.linalg.schur` — use Hamiltonian eigendecomposition instead.
- No `BacksolveAdjoint` for dissipative systems.
- See `CLAUDE.md` (internal) for the full set of JAX constraints.

## Adding a numerical algorithm

Any non-trivial algorithm needs a reference comment:

```python
# Reference: Author (year), "Title", Journal.
```

Add an Octave verification command in the test docstring when the test
compares against specific expected values.

## Pull requests

- Keep PRs focused. One logical change per PR.
- Update docstrings, `__all__`, and `__init__.py` re-exports for any new
  public API.
- Add or extend tests to cover the new behavior.
- Run `uv run pytest tests/ -q` and `uv run ruff check .` before opening
  the PR.
