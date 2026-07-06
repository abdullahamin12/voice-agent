"""
Persistent Qwen3-TTS worker -- STREAMING version, base64-PCM protocol.
Runs in venv-voice.

=== WHAT CHANGED IN THIS PASS, AND WHY (verified against the real
faster-qwen3-tts README, not guessed) ===

1. FIXED BUG: non_streaming_mode=False -> non_streaming_mode=None.
   The library's own README is explicit about this: `non_streaming_mode`
   uses `None` as a sentinel meaning "keep upstream's default for this
   method". For `generate_custom_voice_streaming` specifically, that
   upstream default is **True**. The previous code passed `False` outright,
   which *overrides* the sentinel and forces the model into the wrong mode
   for the CustomVoice model type -- exactly the divergence the old code's
   own comment flagged as unverified. It's now fixed to match the tested
   upstream default.

2. NO MORE TEMP WAV FILES. The old version wrote every audio chunk to a
   NamedTemporaryFile on disk, printed the path, and left the caller to
   read+delete it -- except nothing ever deleted it, so temp files leaked
   for the life of the process, and every chunk paid for a disk round-trip
   that the streaming API's whole point was to avoid. This version
   base64-encodes the raw int16 PCM bytes directly into the stdout protocol
   line. No disk I/O, no leftover files, no read races.

Protocol (stdin -> stdout), one JSON line per message:
  IN:  {"text": "...", "instruct": "warm and reassuring"}
  OUT: one line per audio chunk, the instant it's ready:
       {"audio_b64": "<base64 int16 PCM mono>", "sample_rate": 24000}
  OUT: "<<END>>" once all chunks for that request have been sent.
  OUT: "<<ERROR>> <message>" if synthesis failed (still followed by
       "<<END>>" so the caller's read loop always terminates).

`instruct` is a real Qwen3-TTS CustomVoice feature for natural-language
control of emotion/tone/prosody. native_brain.py fills it in from the
model's own TONE field.
"""

import base64
import json
import sys

import numpy as np
import torch
from faster_qwen3_tts import FasterQwen3TTS

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SPEAKER = "Aiden"
LANGUAGE = "auto"

# ~667ms of audio per chunk on the fast CUDA-graph path. Drop to 2-4 for
# lower latency on a strong GPU (RTX 4090-class), at the cost of a bit more
# decode overhead per chunk -- see faster-qwen3-tts's own chunk-size
# benchmark table.
CHUNK_SIZE = 8


def log(msg: str) -> None:
    # stderr only -- stdout is reserved for the JSON protocol
    print(msg, file=sys.stderr, flush=True)


log("⏳ Loading Qwen3-TTS into VRAM...")
model = FasterQwen3TTS.from_pretrained(
    MODEL_ID,
    device="cuda",
    dtype=torch.bfloat16,
    attn_implementation="eager",  # repo's own default
    backend="torch",  # CUDA-graph fast path
)
log("✅ Qwen3-TTS ready (streaming mode).")


def synthesize_streaming(text: str, instruct: str | None):
    """Generator: yields (pcm16_bytes, sample_rate) for each audio chunk."""
    chunk_iter = model.generate_custom_voice_streaming(
        text=text,
        speaker=SPEAKER,
        language=LANGUAGE,
        instruct=instruct or None,
        chunk_size=CHUNK_SIZE,
        # None = "use the library's own tested default for this method"
        # (True, for CustomVoice) -- see note (1) above. Do NOT hardcode
        # False here; that was the bug.
        non_streaming_mode=None,
    )
    for wav_chunk, sr, _timing in chunk_iter:
        if hasattr(wav_chunk, "detach"):
            wav_chunk = wav_chunk.detach().cpu().numpy()
        wav_chunk = np.asarray(wav_chunk, dtype=np.float32).squeeze()
        if wav_chunk.size == 0:
            continue
        wav_chunk = np.clip(wav_chunk, -1.0, 1.0)
        pcm16 = (wav_chunk * 32767.0).astype(np.int16)
        yield pcm16.tobytes(), sr


log("🎙️  Worker loop ready, waiting for requests on stdin...")
try:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            text = request["text"]
            instruct = request.get("instruct")
        except Exception as e:  # noqa: BLE001
            log(f"❌ Bad request line: {e}")
            print("<<END>>", flush=True)
            continue

        if not text:
            print("<<END>>", flush=True)
            continue

        try:
            chunk_count = 0
            for pcm_bytes, sr in synthesize_streaming(text, instruct):
                out = {
                    "audio_b64": base64.b64encode(pcm_bytes).decode("ascii"),
                    "sample_rate": sr,
                }
                print(json.dumps(out), flush=True)
                chunk_count += 1
            if chunk_count == 0:
                log("❌ TTS produced no audio chunks")
            print("<<END>>", flush=True)
        except Exception as e:  # noqa: BLE001
            log(f"❌ TTS error: {e}")
            print(f"<<ERROR>> {e}", flush=True)
            print("<<END>>", flush=True)
except KeyboardInterrupt:
    pass
