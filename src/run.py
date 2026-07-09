"""
Terminal voice agent -- BARGE-IN SUPPORT, base64-PCM TTS protocol.

Changed vs. the previous version:
  - TTSClient now speaks tts_worker.py's new JSON/base64 protocol (no more
    reading wav files off disk per chunk -- see tts_worker.py's docstring).
  - Resampling / PCM<->float helpers moved to audio_utils.py so run_agent.py
    and server.py don't duplicate them.
Barge-in logic, VAD setup, and the overall turn-taking loop are unchanged.
"""

import json
import subprocess
import threading
import time as _time
from queue import Queue

import numpy as np
import torch

from vad_iterator import VADIterator
from local_audio_streamer import LocalAudioStreamer, AUDIO_RESPONSE_DONE
from audio_utils import (
    SAMPLE_RATE,
    int2float,
    resample_to_16k,
    b64_to_pcm16_bytes,
)
import native_brain

# ---- CONFIG ----
VENV_VOICE_PYTHON = "/home/nauyan/voice-agent-pipeline/.venv-voice/bin/python"
TTS_WORKER_SCRIPT = "/home/nauyan/voice-agent-pipeline/src/tts_worker.py"

CHUNK_SIZE = 512


class TTSClient:
    """Thin wrapper around the persistent TTS subprocess's streaming protocol."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self.proc = proc

    def speak_sentence(
        self, text: str, instruct: str | None, output_queue: Queue
    ) -> None:
        request = json.dumps({"text": text, "instruct": instruct})
        self.proc.stdin.write(request + "\n")
        self.proc.stdin.flush()

        while True:
            line = self.proc.stdout.readline()
            if not line:
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

            try:
                chunk = json.loads(line)
                pcm16 = b64_to_pcm16_bytes(chunk["audio_b64"])
                sr = chunk["sample_rate"]
            except Exception as e:  # noqa: BLE001
                print(f"❌ Could not decode TTS chunk: {e}")
                continue

            audio_float = int2float(pcm16)
            audio_float = resample_to_16k(audio_float, sr)
            audio_int16 = (audio_float * 32768.0).clip(-32768, 32767).astype(np.int16)
            for i in range(0, len(audio_int16), CHUNK_SIZE):
                block = audio_int16[i : i + CHUNK_SIZE]
                if len(block) < CHUNK_SIZE:
                    block = np.pad(block, (0, CHUNK_SIZE - len(block)))
                output_queue.put(block)


def check_barge_in(
    input_queue: Queue, barge_vad: VADIterator, min_speech_ms: int = 250
) -> bool:
    """Drain mic chunks and feed to the barge‑in VAD.
    Returns True if the user has been speaking for >= min_speech_ms."""
    from queue import Empty

    MIN_SAMPLES = SAMPLE_RATE * min_speech_ms // 1000
    while not input_queue.empty():
        try:
            raw_bytes = input_queue.get_nowait()
        except Empty:
            break
        audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        _ = barge_vad(torch.from_numpy(audio_float32))
        if barge_vad.triggered and barge_vad.active_speech_samples >= MIN_SAMPLES:
            return True
    return False


def main() -> None:
    print("⏳ Starting Qwen3-TTS worker (venv-voice, persistent subprocess)...")
    tts_proc = subprocess.Popen(
        [VENV_VOICE_PYTHON, "-u", TTS_WORKER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )

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
        min_silence_duration_ms=500,
    )

    barge_vad = VADIterator(
        silero_model,
        threshold=0.7,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=100,
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
                    if check_barge_in(input_queue, barge_vad):
                        print("🛑 Barge‑in detected – stopping reply.")
                        streamer.flush_output()
                        break

                    if event["type"] == "sentence" and event["text"]:
                        print(f"🤖 {event['text']}")
                        tts.speak_sentence(
                            event["text"], event.get("tone"), output_queue
                        )
                        if check_barge_in(input_queue, barge_vad):
                            print("🛑 Barge‑in during TTS – silencing.")
                            streamer.flush_output()
                            break
                        spoke_anything = True

                if spoke_anything:
                    output_queue.put(AUDIO_RESPONSE_DONE)
                else:
                    should_listen.set()

                barge_vad.reset_states()

            except Exception as e:
                print(f"❌ Brain error: {e}")
                should_listen.set()

    except KeyboardInterrupt:
        print("\n👋 Stopping agent...")
        streamer.stop_event.set()
        tts_proc.terminate()
        import os

        os._exit(0)


if __name__ == "__main__":
    main()
