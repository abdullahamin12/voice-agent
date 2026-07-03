"""
Silero VAD stream wrapper.
Source: adapted from huggingface/speech-to-speech (VAD/vad_iterator.py),
which itself adapts https://github.com/snakers4/silero-vad
Zero framework dependencies -- only torch. Runs in venv-brain.

Feed it 512-sample (32ms @ 16kHz) float32 chunks one at a time via __call__.
Returns None while still listening/silent. Returns the full list of
speech-chunk tensors the moment it confirms the user has stopped talking.
"""

from collections import deque

import torch


class VADIterator:
    def __init__(
        self,
        model,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 30,
    ) -> None:
        """
        model: preloaded Silero VAD model (torch.hub.load("snakers4/silero-vad", "silero_vad"))
        threshold: speech probability above this counts as speech (0.5 is a good default)
        sampling_rate: 8000 or 16000 only
        min_silence_duration_ms: how long silence must persist before we
            declare "user is done talking" -- this is your end-of-turn latency knob.
            Lower = snappier but risks cutting people off. 400-600ms is a good MVP range.
        speech_pad_ms: keep this much audio *before* the detected speech start,
            so we don't clip the first syllable
        """
        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.is_speaking = False
        self.buffer: list[torch.Tensor] = []
        self.prefix_buffer: list[torch.Tensor] = []
        self.active_speech_samples = 0
        self.last_utterance_active_speech_samples = 0
        self._pre_speech_buffer: deque[torch.Tensor] = deque()
        self._pre_speech_samples = 0

        if sampling_rate not in [8000, 16000]:
            raise ValueError(
                "VADIterator does not support sampling rates other than [8000, 16000]"
            )

        self.min_silence_samples = int(sampling_rate * min_silence_duration_ms / 1000)
        self.speech_pad_samples = int(sampling_rate * speech_pad_ms / 1000)
        self.reset_states()

    def reset_states(self) -> None:
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0
        self.buffer = []
        self.prefix_buffer = []
        self.active_speech_samples = 0
        self.last_utterance_active_speech_samples = 0
        self._pre_speech_buffer.clear()
        self._pre_speech_samples = 0

    def _num_samples(self, chunk: torch.Tensor) -> int:
        return len(chunk[0]) if chunk.dim() == 2 else len(chunk)

    def _trim_pre_speech_buffer(self) -> None:
        while (
            self.speech_pad_samples > 0
            and self._pre_speech_buffer
            and self._pre_speech_samples > self.speech_pad_samples
        ):
            first = self._pre_speech_buffer[0]
            first_samples = self._num_samples(first)
            excess = self._pre_speech_samples - self.speech_pad_samples

            if excess >= first_samples:
                self._pre_speech_buffer.popleft()
                self._pre_speech_samples -= first_samples
                continue

            if first.dim() == 2:
                self._pre_speech_buffer[0] = first[:, excess:]
            else:
                self._pre_speech_buffer[0] = first[excess:]
            self._pre_speech_samples -= excess

    def _remember_pre_speech(self, chunk: torch.Tensor) -> None:
        if self.speech_pad_samples <= 0:
            self._pre_speech_buffer.clear()
            self._pre_speech_samples = 0
            return

        self._pre_speech_buffer.append(chunk)
        self._pre_speech_samples += self._num_samples(chunk)
        self._trim_pre_speech_buffer()

    def _speech_buffer(self) -> list[torch.Tensor]:
        if not self.prefix_buffer:
            return list(self.buffer)
        return [*self.prefix_buffer, *self.buffer]

    def speech_buffer(self) -> list[torch.Tensor]:
        return self._speech_buffer()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> list[torch.Tensor] | None:
        if not torch.is_tensor(x):
            try:
                x = torch.Tensor(x)
            except Exception:
                raise TypeError("Audio cannot be casted to tensor. Cast it manually")

        window_size_samples = len(x[0]) if x.dim() == 2 else len(x)
        self.current_sample += window_size_samples

        speech_prob = self.model(x, self.sampling_rate).item()

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            self.prefix_buffer = list(self._pre_speech_buffer)
            self._pre_speech_buffer.clear()
            self._pre_speech_samples = 0
            self.buffer.append(x)
            self.active_speech_samples = window_size_samples
            self.last_utterance_active_speech_samples = 0
            return None

        if not self.triggered:
            self._remember_pre_speech(x)
            return None

        if self.triggered:
            self.buffer.append(x)
            if speech_prob >= self.threshold - 0.15:
                self.active_speech_samples += window_size_samples
                if self.temp_end:
                    self.temp_end = 0
                    return None

            if speech_prob < self.threshold - 0.15:
                if not self.temp_end:
                    self.temp_end = self.current_sample
                if self.current_sample - self.temp_end < self.min_silence_samples:
                    return None

                # End of speech confirmed
                self.temp_end = 0
                self.triggered = False
                spoken_utterance = self.speech_buffer()
                self.last_utterance_active_speech_samples = self.active_speech_samples
                self.active_speech_samples = 0
                self.buffer = []
                self.prefix_buffer = []
                return spoken_utterance

        return None
