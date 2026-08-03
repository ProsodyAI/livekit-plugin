# Security policy

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do
not open a public issue with an API key, audio sample, transcript, or security
report.

Include the affected version, a minimal reproduction, and the impact you
observed. ProsodyAI will acknowledge the report and coordinate remediation and
disclosure privately.

## API keys and audio

Load `PROSODY_API_KEY` from the environment or a secret manager. The analyzer
redacts the key from its representation, but application logs and exception
handlers remain the developer's responsibility.

Audio supplied to `ProsodyAnalyzer.analyze_track()` is sent to the configured
ProsodyAI API for analysis. Use the production HTTPS endpoint unless you are
testing against a trusted local service.

The public SDK exposes recording-local speaker labels. It does not expose raw
speaker embeddings or durable speaker identity.
