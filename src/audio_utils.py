"""
Shared audio helpers -- used by both run_agent.py (local mic) and server.py
(browser client over WebSocket). Pulled out so the two entry points don't
duplicate resampling / PCM<->float conversion logic.

Zero framework dependencies beyond numpy/scipy.
"""

import base64
import math

import numpy as np
from scipy.signal import resample_poly

SAMPLE_RATE = 16000  # what native_brain / Gemma 4 audio input expects


def int2float(pcm16: np.ndarray) -> np.ndarray:
    """int16 PCM samples -> float32 in [-1, 1]."""
    return pcm16.astype(np.float32) / 32768.0


def float2int16(audio: np.ndarray) -> np.ndarray:
    """float32 in [-1, 1] -> int16 PCM samples (clipped)."""
    return np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)


def resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample float32 audio to 16kHz using a polyphase (Fourier-consistent)
    filter -- Google's own Gemma 4 audio guide specifically recommends a
    Fourier-based resampler (scipy.signal.resample / librosa 'scipy') over
    naive linear interpolation for best ASR/understanding quality."""
    if sr == SAMPLE_RATE:
        return audio.astype(np.float32)
    g = math.gcd(sr, SAMPLE_RATE)
    return resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)


def pcm16_bytes_to_b64(pcm16: np.ndarray) -> str:
    return base64.b64encode(pcm16.tobytes()).decode("ascii")


def b64_to_pcm16_bytes(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.int16)


def float_audio_to_b64_pcm16(audio: np.ndarray) -> str:
    return pcm16_bytes_to_b64(float2int16(audio))


def b64_pcm16_to_float_audio(b64: str) -> np.ndarray:
    return int2float(b64_to_pcm16_bytes(b64))


MAX_AUDIO_SECONDS = 30  # hard cap confirmed in Gemma 4's audio docs


def clip_to_max_audio_len(
    audio: np.ndarray, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """Gemma 4's audio input is documented to support a maximum clip length
    of 30 seconds. VAD end-of-turn detection should normally keep utterances
    well under that, but if someone talks for a long time without pausing,
    silently sending >30s of audio into the processor is asking for either a
    truncation you don't control or a hard error. Trim defensively to the
    most recent 30 seconds (people rarely need the very start of a long
    ramble to be understood) and log so it's visible during debugging."""
    max_samples = MAX_AUDIO_SECONDS * sample_rate
    if len(audio) > max_samples:
        return audio[-max_samples:]
    return audio
