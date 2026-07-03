"""
Local full-duplex mic/speaker streamer.
Source: adapted from huggingface/speech-to-speech (connections/local_audio_streamer.py).
Only change from the original: the two imports that pulled in the repo's
internal pipeline framework are replaced with a local sentinel constant.
Runs in venv-brain.

Behavior: while the output_queue is empty, it records mic audio into
input_queue. The moment there's audio queued in output_queue, it switches
to playing that instead (so it doesn't record its own voice back).
"""

import logging
import threading
import time
from queue import Queue

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# Sentinel pushed onto output_queue to mark "agent finished talking" --
# the streamer re-enables should_listen the moment it dequeues this.
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

    def run(self) -> None:
        # Pre-generate a static dither buffer (±1 LSB, -96 dB) to keep the
        # audio sink active without calling numpy inside the real-time callback.
        dither = np.random.randint(
            -1, 2, size=(self.list_play_chunk_size, 1), dtype=np.int16
        )

        def callback(indata, outdata, frames, time_info, status) -> None:
            if self.stop_event.is_set():
                outdata[:] = 0 * outdata
                return

            if self.output_queue.empty():
                pcm = np.ascontiguousarray(indata, dtype=np.int16)
                self.input_queue.put(pcm.tobytes())
                outdata[:] = dither
            else:
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
            logger.info("Starting local audio stream")
            print("🎙️  Mic/speaker stream started.")
            while not self.stop_event.is_set():
                time.sleep(0.001)
            print("Stopping recording")
