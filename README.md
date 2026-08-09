<p align="center">
  <a href="https://prosodyai.app">
    <img src="https://prosodyai.app/logo.png" alt="ProsodyAI" width="88" />
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://livekit.io">
    <img src="https://avatars.githubusercontent.com/u/69438833?s=176" alt="LiveKit" width="88" />
  </a>
</p>

<h1 align="center">ProsodyAI for LiveKit Agents</h1>

<p align="center"><strong>Full-duplex voice agents with persistent speaker identity.</strong></p>

<p align="center">
  <a href="https://prosodyai.app">Product</a> ·
  <a href="https://prosodyai.app/docs">Docs</a> ·
  <a href="https://docs.livekit.io/agents/">LiveKit Agents</a> ·
  <a href="https://pypi.org/project/livekit-plugins-prosodyai/">PyPI</a>
</p>

<p align="center">
  <a href="https://docs.livekit.io/agents/"><img src="https://img.shields.io/badge/LiveKit-Agents-1FD5F9?logo=livekit&logoColor=white" alt="LiveKit Agents" /></a>
  <a href="https://pypi.org/project/livekit-plugins-prosodyai/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license" /></a>
</p>

One continuous speech model carries the whole call. The agent listens and
speaks on the same open connection, with no turn detector and no
STT → LLM → TTS pipeline: audio advances the model's recurrence, and response
timing emerges from continuous generation. Every voice on the call advances
its own recurrent speaker state, so the agent knows who is speaking, notices
when the floor changes, and recognizes a returning caller across hangups.

## Install

```bash
pip install "livekit-plugins-prosodyai[duplex]"
export PROSODYAI_API_KEY=psk_...
```

The duplex transport uses `sphn` for 24 kHz Opus audio.

## A full-duplex agent in six lines

`RealtimeModel` plugs the ProsodyAI speech model into a LiveKit
`AgentSession`. The gateway owns speech generation, response timing,
barge-in, speaker lanes, and memory.

```python
from livekit.agents import AgentSession
from livekit.plugins import prosodyai

model = prosodyai.RealtimeModel()
session = AgentSession(llm=model)

await session.start(room=ctx.room)
```

The model advertises continuous full-duplex capabilities to LiveKit. Audio
advances the model recurrence for the life of the session.

## Persistent speaker identity

Identity is a committed model fact on the wire. Recording-local lanes look
like `speaker_0` and `speaker_1`; a resolved `person_id` is durable across
sessions for the organization that enrolled the voice. When a known voice
comes back, the model resumes that person's state and says so.

```python
realtime = model.sessions[-1]


@realtime.on("prosody_identity")
def on_identity(event):
    print(event.speaker_id, event.person_id, event.display_name)
```

## Consume model events

Each `RealtimeSession` emits typed events from the gateway:

- `prosody_transcript` carries committed words with `speaker_id`, `start_ms`,
  and `end_ms`.
- `prosody_event` carries `SpeakerChangeEvent`, `NewSpeakerEvent`, or
  `IdentityResolvedEvent`.
- `prosody_identity` announces a committed returning person.
- `prosody_text` carries the model's generated text stream.

```python
@realtime.on("prosody_event")
def on_model_event(event):
    print(event.to_dict())
```

Speaker events are model commitments. Their timestamps point to the relevant
audio position, including cases where the model resolves a lane after more
audio arrives.

## Use the bridge directly

`FullDuplexBridge` is the lower-level integration for workers that publish and
subscribe to LiveKit tracks themselves.

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

`uplink_pcm16()` yields little-endian mono PCM16. The bridge converts it to the
gateway's Opus stream and returns little-endian mono PCM16 for publication.

## License

MIT © [ProsodyAI](https://prosodyai.app)
