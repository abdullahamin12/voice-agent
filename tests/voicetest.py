"""
Optimized real-time Qwen3-TTS streaming using faster-qwen3-tts
(CUDA graph backend — no Flash Attention / vLLM / Triton required).

Fixes vs the plain qwen_tts version:
  1. Uses the package's real streaming generator (generate_custom_voice_streaming),
     which yields audio chunks WHILE generating — not one full blocking call per sentence.
  2. Uses StreamPlayer instead of sd.play(..., blocking=True) per chunk, so playback
     doesn't restart/gap between chunks.
  3. Keeps producer (GPU) and consumer (audio) on separate threads so generation of
     the next chunk overlaps with playback of the current one.
  4. Does a warm-up pass before timing, since first-call CUDA graph capture and
     kernel init cost is one-time and shouldn't be counted as latency.
  5. Removes the global torch.set_default_dtype / cudnn.benchmark hacks — the
     package's CUDA graph capture already handles the performance-critical path;
     those globals only add noise/risk here.
"""

import queue
import threading
import time
import warnings
import logging

import numpy as np
import sounddevice as sd

from faster_qwen3_tts import FasterQwen3TTS

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SPEAKER = "aiden"  # run `faster-qwen3-tts custom --model ... --list-speakers` to see valid IDs
LANGUAGE = "English"
CHUNK_SIZE = 4  # steps per yielded audio chunk. 2-4 = lowest latency on desktop GPUs.

# ALSA's "default" device can silently route to a disconnected HDMI output on
# NVIDIA desktops instead of your real speakers. Set this to whichever device
# index actually produced sound in your sd.play() test (commonly 9=pulse or
# 4=analog jack on this machine). None = let sounddevice decide (not recommended
# if you've confirmed default is wrong).
OUTPUT_DEVICE = 9


class StreamPlayer:
    """
    Minimal gapless streaming player: keeps ONE sounddevice.OutputStream open
    and pushes chunks into it as they arrive. Fallback to default if device 9 fails.
    """

    def __init__(self, channels=1, dtype="float32"):
        self._stream = None
        self._channels = channels
        self._dtype = dtype

    def _ensure_stream(self, samplerate):
        if self._stream is None:
            try:
                # Try your preferred custom device index first
                self._stream = sd.OutputStream(
                    samplerate=samplerate,
                    channels=self._channels,
                    dtype=self._dtype,
                    device=OUTPUT_DEVICE,
                )
            except Exception as e:
                print(f"\n⚠️ [Warning] Custom device {OUTPUT_DEVICE} failed: {e}")
                print("Fallback: Sniffing system default audio output...")
                # Automatic fallback to whatever the OS dictates as default
                self._stream = sd.OutputStream(
                    samplerate=samplerate,
                    channels=self._channels,
                    dtype=self._dtype,
                    device=None,
                )
            self._stream.start()

    def __call__(self, audio_chunk, samplerate):
        if hasattr(audio_chunk, "detach"):  # torch tensor
            audio_chunk = audio_chunk.detach().cpu().numpy()
        audio_chunk = np.asarray(audio_chunk, dtype=np.float32).squeeze()
        if audio_chunk.ndim == 1:
            audio_chunk = audio_chunk.reshape(-1, 1)
        if audio_chunk.size == 0:
            return
        self._ensure_stream(samplerate)
        self._stream.write(audio_chunk)

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass


def playback_worker():
    """
    Background worker thread with explicit error printing.
    Prevents silent crashes from locking the main thread.
    """
    try:
        player = StreamPlayer()
        while True:
            item = audio_queue.get()
            if item is None:
                audio_queue.task_done()
                break
            audio_chunk, sr = item
            player(audio_chunk, sr)
            audio_queue.task_done()
    except Exception as thread_error:
        # Crucial Fix: Print the real underlying reason the thread died
        print(f"\n❌ [CRITICAL] Playback Thread crashed: {thread_error}")
        import traceback

        traceback.print_exc()
    finally:
        try:
            player.close()
        except Exception:
            pass


text_chunks = [
    "This is a demonstration of true decoupled streaming.",
    "Notice how the audio plays seamlessly,",
    "while the GPU immediately works on the next segment.",
    "This architecture eliminates the sequential hardware bottleneck.",
]

print("Booting Qwen3-TTS (CUDA graph backend) into VRAM...")
model = FasterQwen3TTS.from_pretrained(
    MODEL_ID
)  # no torch_dtype kwarg — package manages precision internally
print("Engine online.\n")

# ---------------------------------------------------------
# Warm-up: first call pays for CUDA graph capture + kernel JIT.
# Don't let this pollute your TTFA measurement.
# ---------------------------------------------------------
print("Warming up (one throwaway generation)...")
_ = list(
    model.generate_custom_voice_streaming(
        text="Warm up.",
        language=LANGUAGE,
        speaker=SPEAKER,
        chunk_size=CHUNK_SIZE,
    )
)
print("Warm-up complete.\n")

# ---------------------------------------------------------
# Producer/consumer: GPU generation vs playback on separate threads,
# so the next sentence's inference overlaps the current sentence's audio.
# ---------------------------------------------------------
audio_queue = queue.Queue()


def playback_worker():
    player = StreamPlayer()
    try:
        while True:
            item = audio_queue.get()
            if item is None:
                break
            audio_chunk, sr = item
            player(audio_chunk, sr)
            audio_queue.task_done()
    finally:
        player.close()


playback_thread = threading.Thread(target=playback_worker, daemon=True)
playback_thread.start()

print("🚀 Launching inference pipeline...")
total_start = time.perf_counter()
first_audio_logged = False

for i, text_segment in enumerate(text_chunks):
    seg_start = time.perf_counter()
    for audio_chunk, sr, timing in model.generate_custom_voice_streaming(
        text=text_segment,
        language=LANGUAGE,
        speaker=SPEAKER,
        chunk_size=CHUNK_SIZE,
        max_new_tokens=1024,  # guards against the known EOS-miss infinite-generation bug
    ):
        if not first_audio_logged:
            ttfa = (time.perf_counter() - total_start) * 1000
            print(f"[Metrics] Time to First Audio: {ttfa:.2f} ms")
            raw = (
                audio_chunk.detach().cpu().numpy()
                if hasattr(audio_chunk, "detach")
                else np.asarray(audio_chunk)
            )
            print(
                f"[Debug] chunk dtype={raw.dtype} shape={raw.shape} min={raw.min():.4f} max={raw.max():.4f}"
            )
            try:
                import soundfile as sf

                sf.write("/tmp/debug_first_chunk.wav", raw.squeeze(), sr)
                print(
                    "[Debug] wrote /tmp/debug_first_chunk.wav — play it separately to confirm audio data is valid"
                )
            except Exception as e:
                print(f"[Debug] could not write debug wav: {e}")
            first_audio_logged = True
        audio_queue.put((audio_chunk, sr))

    seg_time = time.perf_counter() - seg_start
    print(f"⚡ Segment {i + 1} generated in {seg_time:.3f}s")

audio_queue.put(None)
print("⏳ Generation done — waiting for playback to finish draining the queue...")
playback_thread.join()

print(f"\n✅ Pipeline completed in {time.perf_counter() - total_start:.2f}s")
