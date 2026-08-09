"""PCM resample and format helpers for full-duplex bridging."""

from __future__ import annotations

import numpy as np


def pcm16_le_to_float32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def float32_to_pcm16_le(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def resample_float32(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    n = int(samples.shape[0])
    target = max(1, int(round(n * dst_rate / src_rate)))
    x_old = np.arange(n, dtype=np.float64)
    x_new = np.linspace(0.0, n - 1, target, dtype=np.float64)
    return np.interp(x_new, x_old, samples.astype(np.float64)).astype(np.float32)


def resample_pcm16_le(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    if src_rate == dst_rate or not pcm:
        return pcm
    return float32_to_pcm16_le(
        resample_float32(pcm16_le_to_float32(pcm), src_rate, dst_rate)
    )
