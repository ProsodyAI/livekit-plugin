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

## Maintainer automation

Pull requests and pushes to `dev` or `main` run the same CI matrix. After a
green `dev` push, the promotion workflow opens or updates the `dev` to `main`
pull request only when `PROSODYAI_RELEASE_TOKEN` is configured. It verifies
that the current `dev` head exactly matches the SHA that passed CI. It cannot
merge the pull request.

Use a fine-grained release token restricted to `ProsodyAI/livekit` with
`Contents: read` and `Pull requests: write`. If the token is absent, the
workflow emits a notice and stays green.

After every green `main` push, CI sends a `prosodyai_livekit_main_updated`
repository dispatch to `ProsodyAI/prosodyai` with these payload fields:

- `livekit_sha`: the exact merged `main` commit
- `source_repository`: `ProsodyAI/livekit`
- `source_ref`: `main`

The public repository needs a `PROSODYAI_ROOT_DISPATCH_TOKEN` Actions secret.
Use a fine-grained token restricted to `ProsodyAI/prosodyai` with only the
target repository's `Contents: write` permission, which GitHub requires to
create a repository dispatch. The root repository must listen for the event
and record or propose the new `livekit_sha`. If the secret is absent, CI emits
a notice and stays green so the root repository can poll as a fallback.
