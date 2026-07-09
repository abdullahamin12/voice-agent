"""
Optimized real-time Qwen3-TTS streaming using faster-qwen3-tts
(CUDA graph backend — Live Gapless Resampling Engine).
"""

import queue
import threading
import time
import warnings
import logging
import traceback

import numpy as np
import sounddevice as sd
import torch
from faster_qwen3_tts import FasterQwen3TTS

# ---------------------------------------------------------
# 1. System & Environment Optimization
# ---------------------------------------------------------
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
SPEAKER = "aiden"
LANGUAGE = "English"

# ---------------------------------------------------------
# 2. Dynamic Live Audio Player (Resampling Fallback)
# ---------------------------------------------------------
audio_queue = queue.Queue()


class LiveStreamPlayer:
    """
    Handles streaming playback dynamically. If the hardware rejects the model's
    native 24kHz stream, it intercepts the error and resamples audio on-the-fly.
    """

    def __init__(self, channels=1):
        self._stream = None
        self._channels = channels
        self.target_sr = None

    def _ensure_stream(self, native_sr):
        if self._stream is not None:
            return

        # Attempt 1: Direct playback at native 24kHz sample rate
        try:
            self._stream = sd.OutputStream(
                samplerate=native_sr,
                channels=self._channels,
                dtype="float32",
                device=None,  # Use default system speaker
            )
            self._stream.start()
            self.target_sr = native_sr
            print(f"🔊 Audio device initialized natively at {native_sr}Hz.")
        except Exception as e:
            # Attempt 2: Target device rejected 24kHz. Force fallback to standard 44.1kHz hardware rate.
            print(
                f"\n⚠️ Hardware rejected {native_sr}Hz ({e}). Activating live resampler to 44100Hz..."
            )
            self.target_sr = 44100
            self._stream = sd.OutputStream(
                samplerate=self.target_sr,
                channels=self._channels,
                dtype="float32",
                device=None,
            )
            self._stream.start()

    def __call__(self, audio_chunk, native_sr):
        if hasattr(audio_chunk, "detach"):
            audio_chunk = audio_chunk.detach().cpu().numpy()

        audio_chunk = np.asarray(audio_chunk, dtype=np.float32).squeeze()
        if audio_chunk.size == 0:
            return

        self._ensure_stream(native_sr)

        # On-the-fly linear resampling if hardware rate mismatched
        if self.target_sr != native_sr:
            duration = len(audio_chunk) / native_sr
            num_target_samples = int(duration * self.target_sr)
            audio_chunk = np.interp(
                np.linspace(0, len(audio_chunk), num_target_samples, endpoint=False),
                np.arange(len(audio_chunk)),
                audio_chunk,
            )

        if audio_chunk.ndim == 1:
            audio_chunk = audio_chunk.reshape(-1, 1)

        # Non-blocking write directly to hardware sound card ring-buffers
        self._stream.write(audio_chunk)

    def close(self):
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass


def playback_worker():
    """
    Consumes chunks asynchronously, allowing immediate playback *while*
    the GPU continues generating subsequent blocks.
    """
    player = LiveStreamPlayer()
    try:
        while True:
            item = audio_queue.get()
            if item is None:
                audio_queue.task_done()
                break
            audio_chunk, sr = item
            player(audio_chunk, sr)
            audio_queue.task_done()
    except Exception as thread_error:
        print(f"\n❌ [CRITICAL] Playback Thread crashed: {thread_error}")
        traceback.print_exc()
    finally:
        player.close()


# Start background audio playback thread
playback_thread = threading.Thread(target=playback_worker, daemon=True)
playback_thread.start()

# ---------------------------------------------------------
# 3. Model Engine Load & Warm-up
# ---------------------------------------------------------
print("Booting Faster-Qwen3-TTS Engine into VRAM...")
model = FasterQwen3TTS.from_pretrained(MODEL_ID, torch_dtype="bfloat16", device="cuda")
print("Engine online.\n")

# ---------------------------------------------------------
# 4. Pipeline Execution
# ---------------------------------------------------------
text_chunks = [
    "This is a demonstration of true decoupled streaming.",
    "Notice how the audio plays seamlessly,",
    "while the GPU immediately works on the next segment.",
    "This architecture eliminates the sequential hardware bottleneck.",
]

print("🚀 Launching inference pipeline...")
total_start = time.perf_counter()
first_audio_logged = False

with torch.inference_mode():
    for i, text_segment in enumerate(text_chunks):
        seg_start = time.perf_counter()

        # Use voice cloning stream API
        audio_streamer = model.generate_voice_clone_streaming(
            text=text_segment,
            language=LANGUAGE,
            instruction="Speaking clearly and confidently at a brisk pace.",
            chunk_size=8,  # Keeps processing fluid
        )

        for chunk_index, (wav_chunk, sr, timing) in enumerate(audio_streamer):
            if not first_audio_logged:
                ttfa = (time.perf_counter() - total_start) * 1000
                print(f"\n🔥 [Metrics] True Time to First Audio (TTFA): {ttfa:.2f} ms")
                print("--------------------------------------------------")
                first_audio_logged = True

            # Feed data to queue immediately while loop runs
            audio_queue.put((wav_chunk, sr))

        seg_time = time.perf_counter() - seg_start
        print(f"⚡ Segment {i + 1} generated and queued in {seg_time:.3f}s")

# ---------------------------------------------------------
# 5. Clean Asynchronous Teardown
# ---------------------------------------------------------
print("⏳ Generation complete. Streaming playback loop draining hardware buffer...")

# Wait until background audio queue is empty
audio_queue.join()

# Give soundcard physical room to push remaining soundwaves out
time.sleep(1.5)

# Safe teardown sequence
audio_queue.put(None)
playback_thread.join()

print(
    f"\n✅ Pipeline completed successfully in {time.perf_counter() - total_start:.2f}s"
)
