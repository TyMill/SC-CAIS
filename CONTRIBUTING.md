# Contributing

1. Use Python 3.13+.
2. Install the development extra: `python -m pip install -e '.[dev]'`.
3. Add tests for all behaviour changes.
4. Run `pytest`, Ruff and strict mypy before opening a pull request.
5. Keep agent policies independent from the runtime assurance kernel.
6. Preserve deterministic seeds in benchmark additions.
7. New disruptions must state whether they modify physical state, observation state, or both.
