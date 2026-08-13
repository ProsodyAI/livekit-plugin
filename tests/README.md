# Plugin tests

```bash
cd packages/livekit-plugins-prosodyai
python -m pytest
```

`test_realtime_model.py` needs `sphn` (the `duplex` extra). Wire tests
skip the canonical-source check outside the monorepo.
