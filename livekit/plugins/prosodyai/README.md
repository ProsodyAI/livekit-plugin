# livekit.plugins.prosodyai

Public LiveKit Agents plugin: full-duplex speech with persistent speaker
identity. `FullDuplexBridge` carries PCM to the gateway. `RealtimeModel`
adapts that bridge for an `AgentSession`.

Gateway audio is 24 kHz Opus in ~80 ms frames. This package publishes
20 ms PCM frames so LiveKit RTP stays fed.
