<p align="center">
  <a href="https://prosodyai.app">
    <img src="https://prosodyai.app/logo.png" alt="ProsodyAI" width="88" />
  </a>
</p>

<h1 align="center">ProsodyAI for LiveKit Agents</h1>

<p align="center"><strong>Speech to speech infrastructure.</strong></p>

<p align="center">
  <a href="https://prosodyai.app">Product</a> ·
  <a href="https://prosodyai.app/docs/livekit">Docs</a> ·
  <a href="https://prosodyai.app/docs/reference">API reference</a> ·
  <a href="https://github.com/ProsodyAI/livekit/issues">Issues</a>
</p>

`livekit-plugins-prosodyai` adds real-time acoustic speech analysis and recording-local
diarization to LiveKit Agents. Pass it a subscribed LiveKit audio track and receive ordered
transcript turns, speaker-scoped acoustic measurements, frame trajectories, and same-speaker
deltas.

This is an auxiliary LiveKit Agents plugin. It does not replace the `stt`, `llm`, or `tts`
component selected for an `AgentSession`.

## Install

The package is public but has not completed its first PyPI release. Install the current `main`
commit directly:

```bash
python -m pip install \
  "livekit-plugins-prosodyai @ git+https://github.com/ProsodyAI/livekit.git@main"
```

The audio encoder requires the `ffmpeg` executable on `PATH`.

## Authenticate

Pass an organization API key directly or set it in the environment:

```bash
export PROSODY_API_KEY="your-api-key"
```

## Analyze a LiveKit track

LiveKit plugins use the `livekit.plugins.<provider>` namespace:

```python
import os

from livekit import rtc
from livekit.plugins import prosodyai


async def analyze_track(track: rtc.AudioTrack) -> prosodyai.Conversation:
    analyzer = prosodyai.ProsodyAnalyzer(
        api_key=os.environ["PROSODY_API_KEY"],
        session_id="call-12345",
    )

    async for event in analyzer.analyze_track(track):
        window = event.window
        print(window.speaker_id)
        print(window.get_feature("rms_dbfs"))
        print(window.get_delta("rms_db_change"))

        for point in window.get_frame_series("f0_hz"):
            print(point.timestamp_ms, point.value)

    return analyzer.conversation
```

`track` is a subscribed `rtc.AudioTrack` from the room your agent has joined. The analyzer owns
audio capture, encoding, authentication, reconnects, event ordering, and its typed
`Conversation`.

## Conversation API

```python
conversation.get_transcript()
conversation.get_turns()
conversation.get_turn(0)
conversation.get_speakers()

conversation.get_acoustics()
conversation.get_acoustics("speaker_0")
conversation.get_acoustic_window(0)

conversation.get_feature_series("rms_dbfs")
conversation.get_frame_series("voiced_probability", "speaker_0")
conversation.get_deltas("speaker_0")
```

The typed surface exposes:

- final and interim transcript turns on the same analysis clock
- recording-local `speaker_id` values
- window summaries for RMS and peak level, pitch, pitch range and slope, spectral tilt, voicing,
  pauses, clipping, and voice-onset rate
- frame-level acoustic trajectories at the rate provided by the API
- signed `acoustic_change` values and the exact same-speaker reference used for each delta

Unavailable measurements are `None`, not zero. `acoustic_change.reference` states what each
signed delta is measured against.

## Speaker scope

`speaker_0`, `speaker_1`, and similar labels belong to one recording. They support turns, timing,
and same-speaker acoustic change inside that conversation. This public plugin does not return
voiceprints, embedding vectors, or durable identity across rooms.

## Configuration

```python
prosodyai.ProsodyAnalyzer(
    api_key=None,  # falls back to PROSODY_API_KEY
    base_url="https://api.prosodyai.app",
    session_id=None,  # generated when omitted
    sample_rate=16_000,
    source="livekit",
    max_reconnects=3,
)
```

The API key is redacted from `repr(analyzer)`.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for tests, package checks, and the protected release path.
Report security issues through [SECURITY.md](SECURITY.md).

## License

MIT © [Prosody AI, Inc.](https://prosodyai.app)
