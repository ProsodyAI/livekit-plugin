"""The vendored wire vocabulary must match the repo's canonical source.

The plugin publishes separately, so it ships its own copy of ``wire.py``.
Inside the monorepo this test proves the copy is byte-identical to
``shared/wire.py``; outside the monorepo (an installed distribution) the
canonical source is absent and the check skips.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PLUGIN_COPY = (
    Path(__file__).resolve().parents[1]
    / "livekit"
    / "plugins"
    / "prosodyai"
    / "wire.py"
)


def _canonical_source() -> Path | None:
    for parent in PLUGIN_COPY.parents:
        candidate = parent / "shared" / "wire.py"
        if candidate.exists():
            return candidate
    return None


def test_vendored_wire_matches_the_canonical_source() -> None:
    source = _canonical_source()
    if source is None:
        pytest.skip("canonical shared/wire.py unavailable outside the monorepo")
    assert PLUGIN_COPY.read_bytes() == source.read_bytes(), (
        "the plugin's wire.py drifted from shared/wire.py; "
        "run: python ci/sync_wire.py"
    )
