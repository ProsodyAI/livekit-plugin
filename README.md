# ProsodyAI for LiveKit Agents

**Full-duplex speech agents with persistent speaker identity**

[![PyPI](https://img.shields.io/pypi/v/livekit-plugins-prosodyai)](https://pypi.org/project/livekit-plugins-prosodyai/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/livekit-plugins-prosodyai/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Plugin for [LiveKit Agents](https://docs.livekit.io/agents/) that connects a
room to the ProsodyAI speech model. The model is full-duplex in the sense the
speech-to-speech literature uses: the agent listens and speaks at the same
time on one continuous connection, and turn-taking, overlap, and barge-in
emerge from the model. Speaker identity and conversation events arrive on
the same stream.

[Product](https://prosodyai.app) ·
[Docs](https://prosodyai.app/docs) ·
[LiveKit Agents](https://docs.livekit.io/agents/) ·
[PyPI](https://pypi.org/project/livekit-plugins-prosodyai/)

## Install

```bash
pip install livekit-plugins-prosodyai[duplex]
export PROSODYAI_API_KEY=psk_...
```

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import prosodyai

model = prosodyai.RealtimeModel()
session = AgentSession(llm=model)

await session.start(room=ctx.room)
```

`RealtimeModel` sends continuous room audio to ProsodyAI and returns generated
audio, streaming transcripts, identity updates, and conversation events.

## Speaker identity

Conversation-local diarization labels look like `speaker_0`. When the model
resolves an enrolled caller, it emits a durable `person_id` and display name.
Returning callers resume their saved speaker state.

```python
realtime = model.sessions[-1]


@realtime.on("prosody_identity")
def on_identity(event):
    print(event.speaker_id, event.person_id, event.display_name)
```

## Events

| Event | |
| --- | --- |
| `prosody_transcript` | Committed words with `speaker_id` and word-level timestamps (`start_ms`, `end_ms`) |
| `prosody_event` | Speaker change, new speaker, or identity resolved |
| `prosody_identity` | Returning person committed |
| `prosody_text` | Generated text stream |

```python
@realtime.on("prosody_event")
def on_model_event(event):
    print(event.to_dict())
```

## Lower-level bridge

For workers that publish and subscribe to tracks directly:

```python
from livekit.plugins.prosodyai import (
    FullDuplexBridge,
    FullDuplexBridgeConfig,
    GatewayConnection,
)

connection = GatewayConnection.from_environment()
bridge = FullDuplexBridge(
    FullDuplexBridgeConfig(
        url=connection.url,
        api_key=connection.api_key,
        room_sample_rate=24_000,
        publish_sample_rate=24_000,
    )
)

await bridge.run(
    uplink_pcm16(),
    on_downlink_pcm16=publish_pcm16,
    on_event=handle_gateway_event,
)
```

`uplink_pcm16()` yields little-endian mono PCM16. The bridge returns the same
format for publication.

## Speech backends

The bridge drives its speaking loop through the `SpeechBackend` protocol.
PersonaPlex on the ProsodyAI gateway is the default. Another
speech-to-speech loop (an OpenAI Realtime-style, Gemini Live-style, or
Moshi-style session) plugs in as a class with five members: a
`capabilities` property returning a frozen `SpeechBackendCapabilities`
(`full_duplex`, `accepts_voice_prompt`, `accepts_role_prompt`,
`sample_rate`), `open(config)` taking a `SpeechSessionConfig`,
`send_audio(samples)` taking mono float32 at the declared rate,
`receive()` yielding `SessionOpened`, `SpeechAudio`, and `SpeechText`
items in production order, and `close()`.

```python
from livekit.plugins.prosodyai import FullDuplexBridge, FullDuplexBridgeConfig

bridge = FullDuplexBridge(
    FullDuplexBridgeConfig(
        room_sample_rate=16_000, publish_sample_rate=16_000, role_prompt="You are a concierge."
    ),
    backend=MyRealtimeBackend(),
)
```

Capabilities are declared facts. A turn-based backend declares
`full_duplex=False` and the bridge carries that declaration as-is; the
bridge contains no turn detector. A prompt on the config that the
backend's capabilities exclude raises `BackendCapabilityError` at
construction. Speaker identity, transcripts, and conversation events are
ProsodySSM readouts on the gateway session; they arrive with the default
PersonaPlex backend.

## License

MIT © [ProsodyAI](https://prosodyai.app)
