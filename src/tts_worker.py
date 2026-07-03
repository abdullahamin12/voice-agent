"""
Persistent Qwen3-TTS worker -- STREAMING version.
Runs in venv-voice (transformers==4.57.3).

Why this changed: the old version fully drained
model.generate_custom_voice_streaming() into one big buffer, wrote ONE wav
file, and only then printed its path. That throws away the whole point of
the streaming API -- the CUDA-graph backend's own benchmarks show ~150-450ms
time-to-first-audio (152-159ms on an RTX 4090, ~415ms on a much weaker RTX
3050 -- confirmed from the library's own README/blog), but the old code made
you wait for the ENTIRE reply to finish synthesizing before a single word
played.

Protocol (stdin -> stdout), one line per message:
  IN:  {"text": "...", "instruct": "warm and reassuring"}   (JSON, one line)
  OUT: one line per audio chunk, printed the moment that chunk is ready --
       the path to a small wav file holding just that chunk. The caller
       should start playing chunk 1 as soon as it arrives instead of
       waiting for the rest.
  OUT: "<<END>>" once all chunks for that request have been sent.
  OUT: "<<ERROR>> <message>" if synthesis failed for that request (still
       followed by "<<END>>" so the caller's read loop always terminates).

`instruct` is a real Qwen3-TTS CustomVoice feature for natural-language
control of emotion/tone/prosody -- confirmed CustomVoice (not just
VoiceDesign) supports a speaker + style-instruction combination.
native_brain.py fills it in from the model's own TONE field, which is what
makes replies sound expressive instead of flat and monotone.

NOTE: non_streaming_mode=False below diverges from CustomVoice's documented
upstream default (True). Benchmarks show near-identical speed either way --
if voice quality/behavior seems off, try passing non_streaming_mode=None
(the default) instead.
"""

import json
import sys
import tempfile

import numpy as np
import soundfile as sf
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
    # stderr only -- stdout is reserved for the wav-path protocol
    print(msg, file=sys.stderr, flush=True)


log("⏳ Loading Qwen3-TTS into VRAM...")
model = FasterQwen3TTS.from_pretrained(
    MODEL_ID,
    device="cuda",
    dtype=torch.bfloat16,
    attn_implementation="eager",  # repo's own default -- see comments in the original file
    backend="torch",  # CUDA-graph fast path (matches: no qwentts-cpp-python installed = no ggml)
)
log("✅ Qwen3-TTS ready (streaming mode).")


def synthesize_streaming(text: str, instruct: str | None):
    """Generator: yields a wav file path for each audio chunk as it's produced."""
    chunk_iter = model.generate_custom_voice_streaming(
        text=text,
        speaker=SPEAKER,
        language=LANGUAGE,
        instruct=instruct or None,
        chunk_size=CHUNK_SIZE,
        non_streaming_mode=False,
    )
    for wav_chunk, sr, _timing in chunk_iter:
        if hasattr(wav_chunk, "detach"):
            wav_chunk = wav_chunk.detach().cpu().numpy()
        wav_chunk = np.asarray(wav_chunk, dtype=np.float32).squeeze()
        if wav_chunk.size == 0:
            continue
        wav_chunk = np.clip(wav_chunk, -1.0, 1.0)
        tmp = tempfile.NamedTemporaryFile(
            prefix="agent_chunk_", suffix=".wav", delete=False
        )
        sf.write(tmp.name, wav_chunk, sr, format="WAV")
        yield tmp.name


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
            for wav_path in synthesize_streaming(text, instruct):
                print(wav_path, flush=True)
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
