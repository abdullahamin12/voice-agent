"""
WebSocket server -- lets you talk to the agent from a browser tab (client.html)
instead of only the terminal mic. Runs in venv-brain (it imports native_brain
directly, same as run_agent.py) and manages the TTS worker subprocess itself.

Uses the `websockets` library (current stable API, verified: `async def
handler(websocket)` with no `path` argument is the modern signature --
websockets.serve() is a stable async-context-manager entry point).

PROTOCOL (JSON text frames over the WebSocket):

  Client -> Server
    {"type": "audio_chunk", "audio_b64": "<int16 PCM16 16kHz mono, base64>"}
    {"type": "reset"}                      -- clear conversation memory

  Server -> Client
    {"type": "status", "state": "listening" | "thinking" | "speaking"}
    {"type": "caption", "role": "assistant", "text": "..."}
    {"type": "audio_chunk", "audio_b64": "...", "sample_rate": 24000}
    {"type": "turn_done"}
    {"type": "error", "message": "..."}

DESIGN NOTES / KNOWN LIMITATIONS (stated plainly rather than glossed over):

  - Gemma 4 + Qwen3-TTS are both single-GPU, single-instance resources in
    this setup. All turns, across all connected browser tabs, are serialized
    through one lock (`_gpu_lock`). That's correct for "stream on my
    laptop" -- it is NOT a multi-tenant server.

  - The Silero VAD model is loaded once and shared across connections for
    speed. Because Python's asyncio event loop is single-threaded and the
    VAD call itself contains no `await`, concurrent connections can't
    corrupt each other's turn mid-call -- but if two people genuinely talk
    at the same time from two tabs, their VAD hidden state will interleave
    and accuracy will suffer. Fine for one person, one laptop, one browser
    tab at a time (the stated goal); not fine for a real multi-user deploy.

  - Unlike run_agent.py, this version does not implement barge-in over the
    socket. Interrupting a live model.generate() call cleanly needs extra
    machinery (a cancellation-aware generation loop); it's a reasonable
    follow-up, not something bolted on here to keep this correct rather
    than half-working.
"""

import asyncio
import json
import queue as pyqueue
import subprocess
import threading
import time as _time

import numpy as np
import torch
import websockets

from vad_iterator import VADIterator
from audio_utils import SAMPLE_RATE, b64_pcm16_to_float_audio
import native_brain

# ---- CONFIG ----
VENV_VOICE_PYTHON = "/home/nauyan/voice-agent-pipeline/.venv-voice/bin/python"
TTS_WORKER_SCRIPT = "/home/nauyan/voice-agent-pipeline/src/tts_worker.py"
HOST = "0.0.0.0"
PORT = 8765

_gpu_lock = threading.Lock()  # serializes Gemma + Qwen3-TTS access, see docstring
_tts: "TTSClient | None" = None
_silero_model = None


class TTSClient:
    """Same JSON/base64 protocol as run_agent.py's TTSClient, but yields
    (audio_b64, sample_rate) pairs instead of pushing PCM onto a local
    playback queue -- the caller here forwards them over a websocket."""

    def __init__(self, proc: subprocess.Popen) -> None:
        self.proc = proc

    def synthesize(self, text: str, instruct: str | None):
        request = json.dumps({"text": text, "instruct": instruct})
        self.proc.stdin.write(request + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
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
                yield chunk["audio_b64"], chunk["sample_rate"]
            except Exception as e:  # noqa: BLE001
                print(f"❌ Could not decode TTS chunk: {e}")


def start_tts_worker() -> TTSClient:
    proc = subprocess.Popen(
        [VENV_VOICE_PYTHON, "-u", TTS_WORKER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )
    _time.sleep(1.0)
    if proc.poll() is not None:
        raise RuntimeError(
            f"TTS worker process exited immediately (code {proc.returncode}). "
            f"Check TTS_WORKER_SCRIPT path: {TTS_WORKER_SCRIPT}"
        )
    return TTSClient(proc)


def process_turn_blocking(utterance: np.ndarray, out_q: "pyqueue.Queue") -> None:
    """Runs in a worker thread (kept off the asyncio event loop, since both
    model.generate() and TTS synthesis are blocking calls). Pushes dict
    events onto out_q; a None sentinel signals the turn is complete."""
    try:
        with _gpu_lock:
            for event in native_brain.respond_to_audio(
                utterance, sampling_rate=SAMPLE_RATE
            ):
                if event["type"] == "sentence" and event["text"]:
                    out_q.put(
                        {"type": "caption", "role": "assistant", "text": event["text"]}
                    )
                    for audio_b64, sr in _tts.synthesize(
                        event["text"], event.get("tone")
                    ):
                        out_q.put(
                            {
                                "type": "audio_chunk",
                                "audio_b64": audio_b64,
                                "sample_rate": sr,
                            }
                        )
    except Exception as e:  # noqa: BLE001
        out_q.put({"type": "error", "message": str(e)})
    finally:
        out_q.put(None)


async def handle_turn(websocket, utterance: np.ndarray) -> None:
    await websocket.send(json.dumps({"type": "status", "state": "thinking"}))

    out_q: "pyqueue.Queue" = pyqueue.Queue()
    thread = threading.Thread(
        target=process_turn_blocking, args=(utterance, out_q), daemon=True
    )
    thread.start()

    loop = asyncio.get_running_loop()
    sent_first_audio = False
    while True:
        item = await loop.run_in_executor(None, out_q.get)
        if item is None:
            break
        if item["type"] == "audio_chunk" and not sent_first_audio:
            await websocket.send(json.dumps({"type": "status", "state": "speaking"}))
            sent_first_audio = True
        await websocket.send(json.dumps(item))

    await websocket.send(json.dumps({"type": "turn_done"}))
    await websocket.send(json.dumps({"type": "status", "state": "listening"}))


async def handler(websocket) -> None:
    print("🔌 Client connected")
    vad = VADIterator(
        _silero_model,
        threshold=0.5,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=500,
    )
    await websocket.send(json.dumps({"type": "status", "state": "listening"}))
    try:
        async for message in websocket:
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                continue

            mtype = msg.get("type")
            if mtype == "reset":
                native_brain.reset_memory()
                vad.reset_states()
                await websocket.send(
                    json.dumps({"type": "status", "state": "listening"})
                )
                continue

            if mtype != "audio_chunk":
                continue

            audio_float = b64_pcm16_to_float_audio(msg["audio_b64"])
            speech = vad(torch.from_numpy(audio_float))
            if speech is None:
                continue

            utterance = torch.cat(speech).cpu().numpy()
            print(f"🎤 Heard {len(utterance) / SAMPLE_RATE:.2f}s, thinking...")
            try:
                await handle_turn(websocket, utterance)
            except Exception as e:  # noqa: BLE001
                print(f"❌ Turn error: {e}")
                await websocket.send(json.dumps({"type": "error", "message": str(e)}))
                await websocket.send(
                    json.dumps({"type": "status", "state": "listening"})
                )
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        print("🔌 Client disconnected")


async def main() -> None:
    global _tts, _silero_model

    print("⏳ Starting Qwen3-TTS worker...")
    _tts = start_tts_worker()

    print("⏳ Loading Silero VAD...")
    _silero_model, _ = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", trust_repo=True, skip_validation=True
    )

    print(f"✅ Serving on ws://{HOST}:{PORT}  (open client.html and point it here)")
    async with websockets.serve(handler, HOST, PORT, max_size=2**22):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
