"""
Gemma-4-E4B native audio brain -- STREAMING + MEMORY + SEARCH version.
Runs in venv-brain.

=== WHAT CHANGED IN THIS PASS, AND WHY (verified against Google's own docs
and real transformers source, not guessed) ===

1. transformers>=5.10.1, not >=5.5.0.
   Google's own Gemma 4 audio guide (ai.google.dev/gemma/docs/capabilities/audio)
   and function-calling guide both pin `pip install "transformers>=5.10.1"`.
   5.5.0 predates the Gemma4 model integration landing in transformers and
   will not import Gemma4Processor at all.

2. ONE-STEP generation input instead of two.
   The previous version called `processor.apply_chat_template(..., tokenize=False)`
   to get a text string, then called `processor(text=prompt, audio=audios,
   sampling_rate=sampling_rate, ...)` by hand. That second call's `sampling_rate`
   kwarg was never confirmed against Gemma4's actual audio feature extractor
   signature -- and checking the real extractor
   (transformers/models/gemma4/*), its `__call__` takes `raw_speech` with NO
   `sampling_rate` parameter (Gemma 4 always assumes pre-resampled 16kHz
   mono float32, per the model card's own audio encoding section). Passing
   an unexpected kwarg like that is a plausible source of a hard crash on
   every audio turn.
   The documented, working pattern (Google's own audio-capabilities guide,
   and a hands-on community walkthrough that runs this exact call) is to put
   the raw numpy audio array straight into the message content and call:
       processor.apply_chat_template(messages, tokenize=True,
                                      add_generation_prompt=True,
                                      return_dict=True, return_tensors="pt")
   which returns a ready-to-generate() dict and handles audio feature
   extraction internally. That's what this version does -- for both the
   audio-carrying first pass AND the text-only search follow-up pass, so
   there's only one code path instead of two.

3. Sampling parameters now match Google's documented standard config for
   Gemma 4 (`temperature=1.0, top_p=0.95, top_k=64`) instead of an
   unsourced `temperature=0.7`.

4. MAX_NEW_TOKENS raised 260 -> 640. This is almost certainly the actual
   cause of "bot doesn't reply fully": one generation pass has to fit
   LOG + TONE + SEARCH + the entire REPLY inside a single token budget, and
   260 tokens barely covers a couple of sentences once the three header
   fields are subtracted. Streaming means the extra tokens don't add to
   time-to-first-audio, only to total length.

5. `dtype=` instead of the now-deprecated `torch_dtype=` (confirmed: recent
   transformers emits "`torch_dtype` is deprecated! Use `dtype` instead!").

6. Model class fallback reordered. Both `AutoModelForImageTextToText` (used
   consistently across the model card, the audio guide, and the
   HF Transformers guide) and `AutoModelForMultimodalLM` (used in Google's
   own function-calling guide) are real, currently-documented entry points
   for Gemma 4 -- Google's docs are simply inconsistent about which one they
   show. Trying both is correct defensive coding, not guesswork.

7. Bounded conversation memory (was unbounded) so a long-running server
   session doesn't grow the prompt forever.

8. Audio is defensively clipped to Gemma 4's documented 30-second input cap
   before it ever reaches the processor (see audio_utils.clip_to_max_audio_len).

Everything else (LOG/TONE/SEARCH/REPLY structured output, per-sentence
streaming, one-round web search) is unchanged in spirit -- just made to run
on a corrected, source-verified inference path.

>>> ADJUST MODEL_PATH BELOW to your actual local weights folder. <<<
"""

import re
import threading
from collections import deque
from typing import Iterator, Optional

import torch

from audio_utils import clip_to_max_audio_len
from web_search import search_web

# ---- CONFIG: change this to your actual absolute path ---
MODEL_PATH = "/home/nauyan/voice-agent-pipeline/models/gemma-4-E4B-it"

MAX_NEW_TOKENS = 640  # was 260 -- see note (4) above
MAX_MEMORY_TURNS = 60  # was unbounded -- bounds prompt growth on long sessions

# Google's documented standardized sampling config for Gemma 4 (all sizes).
SAMPLING_KWARGS = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=64)

SYSTEM_PROMPT = (
    "You are a warm, emotionally present voice companion having a real spoken "
    "conversation, not writing text. Never use markdown, asterisks, bullet points, "
    "or emojis -- your output goes straight to a text-to-speech engine.\n\n"
    "You keep track of the conversation (shown below as prior turns) and, when you "
    "genuinely need current information you don't know -- news, prices, scores, "
    "today's date, anything time-sensitive -- you can request a web search instead "
    "of guessing or making something up.\n\n"
    "Always answer using exactly this structure, one field per line, in this order, "
    "and nothing before or after it:\n"
    "LOG: a short sentence summarizing what the user just said, for your own memory\n"
    "TONE: 2-4 words for how this reply should sound, e.g. warm and reassuring, "
    "playful and excited, calm and matter-of-fact\n"
    "SEARCH: a short search query, ONLY if you truly need current information -- "
    "otherwise write exactly NONE\n"
    "REPLY: the exact words you will say out loud. Speak like a real person -- "
    "contractions, warmth, personality. Give a genuinely complete, useful answer "
    "unless the user clearly just wants something quick.\n\n"
    "If SEARCH is not NONE, end your response right after the SEARCH line -- you'll "
    "be given the results and a chance to write REPLY afterward."
    # NOTE: deliberately no "<|think|>" token here. Gemma 4's E2B/E4B variants
    # are the documented exception that emits nothing extra when thinking is
    # left disabled (larger variants would emit an empty <|channel>thought
    # block even when disabled) -- so this system prompt correctly keeps
    # thinking off with zero wasted tokens for this model size.
)

print(f"⏳ Loading offline multimodal brain from {MODEL_PATH}...")

from transformers import AutoProcessor, TextIteratorStreamer  # noqa: E402

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = getattr(processor, "tokenizer", processor)

# Both class names below are real, currently-documented Gemma 4 entry points
# (Google's own docs are simply inconsistent about which one they show on
# which page) -- trying both, in the order most consistently documented, is
# correct defensive loading, not a guess.
model = None
_load_errors = []
for _cls_name in (
    "AutoModelForImageTextToText",
    "AutoModelForMultimodalLM",
    "AutoModelForCausalLM",
):
    try:
        import transformers as _tf

        _cls = getattr(_tf, _cls_name)
        model = _cls.from_pretrained(
            MODEL_PATH,
            device_map="cuda",
            dtype=torch.bfloat16,  # `dtype=`, not deprecated `torch_dtype=`
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        print(f"✅ Loaded with {_cls_name}")
        break
    except Exception as e:  # noqa: BLE001
        _load_errors.append(f"{_cls_name}: {e}")
        continue

if model is None:
    raise RuntimeError(
        "Could not load Gemma-4-E4B with any known multimodal class. Errors:\n"
        + "\n".join(_load_errors)
    )

print("✅ Multimodal Brain Ready.\n")

_memory: deque[tuple[str, str]] = deque(maxlen=MAX_MEMORY_TURNS)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def reset_memory() -> None:
    """Call this to start a fresh conversation (e.g. on a wake-word reset)."""
    _memory.clear()


def _field(text: str, label: str, next_label: str) -> str:
    i = text.find(label)
    if i == -1:
        return ""
    i += len(label)
    j = text.find(next_label, i)
    if j == -1:
        j = len(text)
    return text[i:j].strip()


def _build_messages(audio_array) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for log_text, reply_text in _memory:
        messages.append(
            {"role": "user", "content": [{"type": "text", "text": log_text}]}
        )
        messages.append({"role": "assistant", "content": reply_text})
    messages.append(
        {
            "role": "user",
            # Text BEFORE audio -- confirmed current in Gemma 4's own model
            # card ("For optimal performance with multimodal inputs ...
            # place Audio content after the text in your prompt").
            "content": [
                {"type": "text", "text": "Listen to the following audio and respond."},
                {"type": "audio", "audio": audio_array},
            ],
        }
    )
    return messages


def _generate_stream(messages: list) -> Iterator[str]:
    """Runs model.generate() in a background thread and yields decoded text
    deltas as they're produced. Single-step chat-template call -- see note
    (2) in the module docstring for why the old two-step version was risky."""
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=MAX_NEW_TOKENS,
        **SAMPLING_KWARGS,
    )
    thread = threading.Thread(
        target=model.generate, kwargs=generate_kwargs, daemon=True
    )
    thread.start()
    for delta in streamer:
        yield delta
    thread.join()


def _run_pass(messages: list) -> Iterator[tuple[str, str]]:
    """Generator: yields (sentence, tone) for each REPLY sentence the moment
    it's ready. Its return value (accessed via StopIteration.value when
    driven manually) is (log, tone, search_query, full_reply_text)."""
    full_text = ""
    reply_start = None
    emitted_upto = 0
    log_text = tone_text = search_text = ""

    for delta in _generate_stream(messages):
        full_text += delta

        if reply_start is None:
            idx = full_text.find("REPLY:")
            if idx == -1:
                continue
            reply_start = idx + len("REPLY:")
            emitted_upto = reply_start
            log_text = _field(full_text, "LOG:", "TONE:")
            tone_text = _field(full_text, "TONE:", "SEARCH:")
            search_text = _field(full_text, "SEARCH:", "REPLY:")

        unsent = full_text[emitted_upto:]
        while True:
            m = _SENTENCE_END.search(unsent)
            if not m:
                break
            cut = m.end()
            sentence, unsent = unsent[:cut], unsent[cut:]
            if sentence.strip():
                yield sentence.strip(), tone_text
            emitted_upto += cut

    if reply_start is not None:
        tail = full_text[emitted_upto:].strip()
        if tail:
            yield tail, tone_text
        reply_text = full_text[reply_start:].strip()
    else:
        # The model ignored the LOG/TONE/SEARCH/REPLY format entirely (small
        # models sometimes drift) -- speak whatever it said rather than
        # staying silent.
        raw = full_text.strip()
        if raw:
            yield raw, ""
        reply_text = raw

    return log_text, tone_text, search_text, reply_text


def respond_to_audio(audio_array, sampling_rate: int = 16000) -> Iterator[dict]:
    """Main entry point. `audio_array` must already be float32 mono at
    `sampling_rate` (16kHz) -- both run_agent.py and server.py resample
    before calling this.

    Yields events as the reply is produced:
      {"type": "sentence", "text": ..., "tone": ...}  -- ready to speak now
      {"type": "done"}                                 -- reply finished

    Handles conversation memory and an optional one-round web search
    internally, so the caller never has to know about either.
    """
    if sampling_rate != 16000:
        raise ValueError(
            "respond_to_audio expects 16kHz audio; resample before calling "
            "(see audio_utils.resample_to_16k)."
        )
    audio_array = clip_to_max_audio_len(audio_array, sampling_rate)

    messages = _build_messages(audio_array)
    gen = _run_pass(messages)

    log_text = tone_text = search_query = reply_text = ""
    while True:
        try:
            sentence, tone = next(gen)
        except StopIteration as stop:
            log_text, tone_text, search_query, reply_text = stop.value
            break
        yield {"type": "sentence", "text": sentence, "tone": tone or None}

    if search_query and search_query.strip().upper() != "NONE":
        print(f"🔎 Searching: {search_query}")
        results = search_web(search_query)
        followup = messages + [
            {
                "role": "assistant",
                "content": f"LOG: {log_text}\nTONE: {tone_text}\nSEARCH: {search_query}",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Search results:\n{results}\n\n"
                            "The user can't see these results. Using them, write REPLY "
                            "only (skip LOG/TONE/SEARCH) in the same spoken style."
                        ),
                    }
                ],
            },
        ]

        full = ""
        sentences: list[str] = []
        for delta in _generate_stream(followup):
            full += delta
            while True:
                m = _SENTENCE_END.search(full)
                if not m:
                    break
                sentence, full = full[: m.end()], full[m.end() :]
                sentence = sentence.strip()
                if sentence:
                    sentences.append(sentence)
                    yield {
                        "type": "sentence",
                        "text": sentence,
                        "tone": tone_text or None,
                    }
        if full.strip():
            sentences.append(full.strip())
            yield {"type": "sentence", "text": full.strip(), "tone": tone_text or None}

        reply_text = " ".join(sentences)

    if log_text:
        _memory.append((log_text, reply_text))
    yield {"type": "done"}


if __name__ == "__main__":
    import librosa

    test_file = "data/audio_logs/test.wav"
    try:
        audio_array, _ = librosa.load(test_file, sr=16000)
        for event in respond_to_audio(audio_array, sampling_rate=16000):
            if event["type"] == "sentence":
                print(f"🤖 [{event['tone']}] {event['text']}")
    except Exception as e:  # noqa: BLE001
        print(f"❌ Error: {e}")
