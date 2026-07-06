"""
Local full-duplex mic/speaker streamer -- ALWAYS records mic.
Source: adapted from huggingface/speech-to-speech.

Unchanged from your version -- the always-record full-duplex design plus
flush_output() for barge-in was already correct.
"""

import logging
import threading
import time
from queue import Empty, Queue

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

AUDIO_RESPONSE_DONE = "AUDIO_RESPONSE_DONE"


class LocalAudioStreamer:
    def __init__(
        self,
        input_queue: Queue,
        output_queue: Queue,
        should_listen: threading.Event,
        list_play_chunk_size: int = 512,
    ) -> None:
        self.list_play_chunk_size = list_play_chunk_size
        self.stop_event = threading.Event()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.should_listen = should_listen

    def flush_output(self) -> int:
        """Immediately drop all queued audio. Call on barge‑in."""
        dropped = 0
        while True:
            try:
                self.output_queue.get_nowait()
                dropped += 1
            except Empty:
                break
        return dropped

    def run(self) -> None:
        dither = np.random.randint(
            -1, 2, size=(self.list_play_chunk_size, 1), dtype=np.int16
        )

        def callback(indata, outdata, frames, time_info, status) -> None:
            if self.stop_event.is_set():
                outdata[:] = 0 * outdata
                return
            # ----- ALWAYS record the mic, no matter what -----
            pcm = np.ascontiguousarray(indata, dtype=np.int16)
            self.input_queue.put(pcm.tobytes())
            # ----- Playback is a completely independent decision -----
            if self.output_queue.empty():
                outdata[:] = dither
                return
            try:
                audio_chunk = self.output_queue.get_nowait()
                if isinstance(audio_chunk, np.ndarray):
                    outdata[:] = audio_chunk[:, np.newaxis]
                elif audio_chunk == AUDIO_RESPONSE_DONE:
                    self.should_listen.set()
                    logger.debug("Response complete, listening re-enabled")
                    outdata[:] = 0 * outdata
                else:
                    outdata[:] = 0 * outdata
            except Exception:
                outdata[:] = 0 * outdata

        logger.debug("Available devices:")
        logger.debug(sd.query_devices())
        with sd.Stream(
            samplerate=16000,
            dtype="int16",
            channels=1,
            callback=callback,
            blocksize=self.list_play_chunk_size,
        ):
            logger.info("Starting local audio stream (true full duplex)")
            print("🎙️  Mic/speaker stream started (mic stays live during playback).")
            while not self.stop_event.is_set():
                time.sleep(0.001)
            print("Stopping recording")
