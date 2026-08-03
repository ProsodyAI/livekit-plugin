# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The test suite uses synthetic payloads and does not require credentials or a
network connection.

## Checks

```bash
ruff check .
mypy livekit_plugins_prosodyai
python -m pytest
python -m build
```

Add a focused test for every public behavior change. Keep transport details
inside the package and make acoustic measurements available through the typed
conversation surface.
