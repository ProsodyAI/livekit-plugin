"""Consume acoustic measurements from an existing LiveKit audio track."""

from __future__ import annotations

import os

from livekit import rtc
from livekit.plugins import prosodyai


async def observe_track(track: rtc.AudioTrack) -> None:
    """Analyze one track and print measured acoustic state as it arrives."""

    analyzer = prosodyai.ProsodyAnalyzer(api_key=os.environ["PROSODY_API_KEY"])

    async for event in analyzer.analyze_track(track):
        window = event.window
        level = window.get_feature("rms_dbfs")
        pitch = window.get_feature("f0_median_hz")
        level_change = window.get_delta("rms_db_change")
        print(f"{window.speaker_id} level={level!r} pitch={pitch!r} level_change={level_change!r}")

    conversation = analyzer.conversation
    for turn in conversation.get_turns(final_only=True):
        print(f"{turn.speaker_id}: {turn.text}")
