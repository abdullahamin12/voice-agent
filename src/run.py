"""
Terminal voice agent -- no network required except for the optional web
search step, no Docker, no browser.
Run with the venv-brain Python interpreter:
    .venv-brain/bin/python run_agent.py

Loads Gemma-4-E4B locally, launches the Qwen3-TTS worker as a PERSISTENT
subprocess in venv-voice (loaded once, kept warm), listens on your mic via
Silero VAD, and speaks the reply back through your speakers.

STREAMING PIPELINE -- this is the main fix vs. the earlier version:
  1. native_brain.respond_to_audio() streams Gemma's reply out sentence by
     sentence instead of generating the whole answer before anything
     happens.
  2. Each finished sentence is sent to the TTS worker immediately.
  3. The TTS worker streams audio chunks back as they're generated (not as
     one big file at the end) -- playback of chunk 1 starts while later
     chunks, and later sentences, are still being produced.

Previously NOTHING happened until BOTH the full LLM answer AND the full TTS
clip existed -- that serial wait is what made the agent feel slow. LLM
generation and TTS generation still share one GPU, so they're not truly
running at the same instant, but this removes the "wait for 100% before
starting anything" bottleneck, which is what actually drives the time you
wait before you hear the first word.

KNOWN COSMETIC ISSUE: each TTS chunk is resampled 24kHz->16kHz
independently before queuing (correct math, via resample_poly), which can
introduce faint clicks at chunk boundaries since each chunk is resampled in
isolation rather than as one continuous stream. Not blocking -- just
something to listen for.

>>> ADJUST THE TWO PATHS BELOW to match your actual folders. <<<
"""

import json
import math
import subprocess
import threading
import time as _time
from queue import Queue

import numpy as np
import torch
import soundfile as sf
from scipy.signal import resample_poly

from vad_iterator import VADIterator
from local_audio_streamer import LocalAudioStreamer, AUDIO_RESPONSE_DONE
import native_brain

# ---- CONFIG: change these two paths to your actual setup ----
VENV_VOICE_PYTHON = "/home/nauyan/voice-agent-pipeline/.venv-voice/bin/python"
TTS_WORKER_SCRIPT = "/home/nauyan/voice-agent-pipeline/src/tts_worker.py"

SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # samples per block (~32ms @ 16kHz) -- matches what Silero VAD expects


def int2float(sound: np.ndarray) -> np.ndarray:
    return sound.astype(np.float32) / 32768.0


def resample_to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Qwen3-TTS's native output rate is not 16kHz, and the mic/speaker
    stream below runs a single duplex line at 16kHz, so this conversion is
    still needed. resample_poly is a lighter-weight, faster call per chunk
    than librosa.resample was in the old per-reply version -- if you want
    the model's full native audio quality with zero resampling, the next
    step is splitting the duplex stream into an independent input stream
    (16kHz, for VAD) and output stream (TTS's native rate)."""
    if sr == SAMPLE_RATE:
        return audio
    g = math.gcd(sr, SAMPLE_RATE)
    return resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)


class TTSClient:
    """Thin wrapper around the persistent TTS subprocess's streaming protocol."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self.proc = proc

    def speak_sentence(
        self, text: str, instruct: str | None, output_queue: Queue
    ) -> None:
        """Sends one sentence and streams its audio chunks straight into
        output_queue as they're produced -- does NOT wait for the whole
        sentence's audio to finish synthesizing before queuing chunk 1."""
        request = json.dumps({"text": text, "instruct": instruct})
        self.proc.stdin.write(request + "\n")
        self.proc.stdin.flush()

        while True:
            line = self.proc.stdout.readline()
            if not line:
                # worker died -- avoid spinning forever
                print("❌ TTS worker pipe closed unexpectedly")
                return
            line = line.strip()
            if not line:
                continue
            if line == "<<END>>":
                return
            if line.startswith("<<ERROR>>"):
                print(f"❌ TTS worker: {line}")
                continue

            wav_path = line
            try:
                audio, sr = sf.read(wav_path, dtype="float32")
            except Exception as e:  # noqa: BLE001
                print(f"❌ Could not read TTS chunk {wav_path}: {e}")
                continue
            audio = resample_to_16k(audio, sr)
            audio_int16 = np.clip(audio * 32768, -32768, 32767).astype(np.int16)
            for i in range(0, len(audio_int16), CHUNK_SIZE):
                block = audio_int16[i : i + CHUNK_SIZE]
                if len(block) < CHUNK_SIZE:
                    block = np.pad(block, (0, CHUNK_SIZE - len(block)))
                output_queue.put(block)


def main() -> None:
    print("⏳ Starting Qwen3-TTS worker (venv-voice, persistent subprocess)...")
    tts_proc = subprocess.Popen(
        [VENV_VOICE_PYTHON, "-u", TTS_WORKER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,  # its own logs print straight to your terminal
        text=True,
        bufsize=1,
    )

    # Fail loudly if the TTS worker died on launch (wrong path, import error,
    # etc.) instead of silently continuing as if it were running.
    _time.sleep(1.0)
    if tts_proc.poll() is not None:
        raise RuntimeError(
            f"TTS worker process exited immediately (code {tts_proc.returncode}). "
            f"Check TTS_WORKER_SCRIPT path: {TTS_WORKER_SCRIPT}"
        )
    tts = TTSClient(tts_proc)

    print("⏳ Loading Silero VAD...")
    silero_model, _ = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", trust_repo=True, skip_validation=True
    )
    vad = VADIterator(
        silero_model,
        threshold=0.5,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=500,  # end-of-turn latency knob -- lower = snappier, riskier
    )

    input_queue: Queue = Queue()
    output_queue: Queue = Queue()
    should_listen = threading.Event()
    should_listen.set()

    streamer = LocalAudioStreamer(
        input_queue, output_queue, should_listen, list_play_chunk_size=CHUNK_SIZE
    )
    streamer_thread = threading.Thread(target=streamer.run, daemon=True)
    streamer_thread.start()

    print("✅ Agent ready. Speak into your mic (Ctrl+C to stop).\n")

    try:
        while True:
            raw_bytes = input_queue.get()

            if not should_listen.is_set():
                continue

            audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
            audio_float32 = int2float(audio_int16)

            speech = vad(torch.from_numpy(audio_float32))
            if not speech:
                continue

            utterance = torch.cat(speech).cpu().numpy()
            print(
                f"🎤 Heard {len(utterance) / SAMPLE_RATE:.2f}s of speech, thinking..."
            )
            should_listen.clear()

            try:
                spoke_anything = False
                for event in native_brain.respond_to_audio(
                    utterance, sampling_rate=SAMPLE_RATE
                ):
                    if event["type"] == "sentence" and event["text"]:
                        print(f"🤖 {event['text']}")
                        tts.speak_sentence(
                            event["text"], event.get("tone"), output_queue
                        )
                        spoke_anything = True

                if spoke_anything:
                    output_queue.put(
                        AUDIO_RESPONSE_DONE
                    )  # streamer re-enables should_listen on dequeue
                else:
                    should_listen.set()
            except Exception as e:  # noqa: BLE001
                print(f"❌ Brain error: {e}")
                should_listen.set()

    except KeyboardInterrupt:
        print("\n👋 Stopping agent...")
        streamer.stop_event.set()
        tts_proc.terminate()
        # Hard exit: sd.Stream's close() can hang on some Linux audio
        # backends, which left the previous process alive and still
        # holding ~15GB of VRAM. os._exit skips all cleanup and forces
        # the OS/CUDA driver to reclaim everything immediately.
        import os

        os._exit(0)


if __name__ == "__main__":
    main()
