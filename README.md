# ProsodyAI for LiveKit

Turn a LiveKit audio track into an ordered conversation with speaker turns,
recording-local diarization, acoustic measurements, and same-speaker change.

`ProsodyAnalyzer` owns a `Conversation`. As audio is analyzed, the conversation
accumulates transcript turns and acoustic windows. Each window contains summary
features plus its Mimi-aligned frame trajectory at 12.5 Hz.

## Install

The package is installed from GitHub until its first PyPI release:

```bash
python -m pip install \
  'livekit-plugins-prosodyai @ git+https://github.com/ProsodyAI/livekit.git'
```

The audio encoder requires `ffmpeg` on `PATH`.

## Analyze a LiveKit track

```python
import os

from livekit_plugins_prosodyai import ProsodyAnalyzer

analyzer = ProsodyAnalyzer(api_key=os.environ["PROSODY_API_KEY"])

async for event in analyzer.analyze_track(audio_track):
    window = event.window

    print(window.speaker_id)
    print(window.get_feature("rms_dbfs"))
    print(window.get_delta("rms_db_change"))

    for point in window.get_frame_series("f0_hz"):
        print(point.timestamp_ms, point.value)

conversation = analyzer.conversation

for turn in conversation.get_turns(final_only=True):
    print(turn.speaker_id, turn.text)

for speaker in conversation.get_speakers():
    print(speaker.speaker_id, speaker.talk_ms, speaker.turn_count)
```

`analyze_track()` handles audio capture, encoding, authentication, and the
ordered analysis stream. Your application works with typed acoustic events and
the conversation that those events update.

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

Window-level acoustic state includes:

- RMS and peak level in dBFS
- median pitch, pitch range, and pitch slope
- spectral tilt
- voiced ratio and pause ratio
- clipping ratio
- voice onset rate

Frame series preserve the model's 12.5 Hz timing for:

- RMS level
- pitch
- spectral tilt
- voiced probability
- voice-activity-boundary probability

Unavailable measurements are `None`, not zero. `acoustic_change.reference`
states what each signed delta is measured against.

## Speaker scope

`speaker_0`, `speaker_1`, and similar labels belong to one recording. They are
useful for turns, timing, and same-speaker acoustic change inside that
conversation. The package does not return voiceprints, embedding vectors, or a
durable identity for a person.

## Authentication

Pass an API key directly or set `PROSODY_API_KEY`:

```python
analyzer = ProsodyAnalyzer()
```

The API key is held privately by the analyzer and redacted from `repr()`.

## Development

```bash
python -m pip install -e '.[dev]'
ruff check .
mypy livekit_plugins_prosodyai
python -m pytest
python -m build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).
