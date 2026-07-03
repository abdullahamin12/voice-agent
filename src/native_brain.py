"""
Gemma-4-E4B native audio brain -- STREAMING + MEMORY + SEARCH version.
Runs in venv-brain (transformers>=5.5.0).

What changed vs. the earlier version, and why:

  - STREAMING GENERATION. The old version called model.generate() and blocked
    until the entire reply existed before returning anything. This version
    uses TextIteratorStreamer in a background thread and yields each
    completed SENTENCE the moment it's ready, so run_agent.py can hand it to
    the TTS worker immediately instead of waiting for the whole answer.

  - STRUCTURED OUTPUT (LOG / TONE / SEARCH / REPLY). One generation pass now
    does four jobs at once instead of needing separate round trips:
      LOG    -- a one-line memory note (what the user said)
      TONE   -- 2-4 words fed to Qwen3-TTS's `instruct` parameter, which is
                a real feature of the CustomVoice model for controlling
                emotion/prosody -- this is what makes replies sound
                expressive instead of flat.
      SEARCH -- a query string, or NONE, letting the model ask for current
                information instead of guessing.
      REPLY  -- the actual words to speak.

  - LIGHTWEIGHT TEXT MEMORY. Conversation history persists as (log, reply)
    text pairs, NOT raw audio. Keeping raw audio in history would make every
    later turn slower (more audio tokens to re-encode on every single turn)
    -- text is cheap and keeps latency flat across a long conversation.

  - MAX_NEW_TOKENS raised from 100 to 260, and the old "keep it brief"
    system-prompt instruction is gone. That combination is why answers felt
    thin before. Streaming means the extra tokens no longer delay the first
    spoken word -- they only add to total reply length.

KNOWN UNVERIFIED RISK: the search follow-up pass below calls the processor
with NO audio input at all (text-only). Whether Gemma 4's multimodal
processor accepts that cleanly hasn't been confirmed against your exact
build -- this is the single most likely place to throw an error on first
run of a search-triggering turn.

>>> ADJUST MODEL_PATH BELOW to your actual local weights folder. <<<
"""

import re
import threading
from collections import deque
from typing import Iterator, Optional

import torch

from web_search import search_web

# ---- CONFIG: change this to your actual absolute path ----
MODEL_PATH = "/home/nauyan/voice-agent-pipeline/models/gemma-4-E4B-it"

MAX_NEW_TOKENS = 260  # was 100 -- see note above
MAX_MEMORY_TURNS = None  # text-only turns kept in context

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
)

print(f"⏳ Loading offline multimodal brain from {MODEL_PATH}...")

from transformers import AutoProcessor, TextIteratorStreamer  # noqa: E402

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
# Some multimodal processors nest the text tokenizer as .tokenizer, some are
# usable directly -- fall back gracefully either way.
tokenizer = getattr(processor, "tokenizer", processor)

# Robustness: the exact audio-capable class name has moved around across
# recent transformers dev builds. Try the officially documented one first,
# fall back if your installed version names it differently.
model = None
_load_errors = []
for _cls_name in (
    "AutoModelForMultimodalLM",
    "AutoModelForImageTextToText",
    "AutoModelForCausalLM",
):
    try:
        import transformers as _tf

        _cls = getattr(_tf, _cls_name)
        model = _cls.from_pretrained(
            MODEL_PATH,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
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
            # Text BEFORE audio in the content list -- this matches Google's
            # own gemma-4-E4B-it model card example, confirmed current.
            "content": [
                {"type": "text", "text": "Listen to the following audio and respond."},
                {"type": "audio", "audio": audio_array},
            ],
        }
    )
    return messages


def _generate_stream(messages, audios, sampling_rate: Optional[int]) -> Iterator[str]:
    """Runs model.generate() in a background thread and yields decoded text
    deltas as they're produced, instead of blocking until generation ends."""
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    proc_kwargs = dict(text=prompt, return_tensors="pt")
    if audios is not None:
        proc_kwargs["audio"] = audios
        proc_kwargs["sampling_rate"] = sampling_rate
    inputs = processor(**proc_kwargs).to("cuda")

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.1,
    )
    thread = threading.Thread(
        target=model.generate, kwargs=generate_kwargs, daemon=True
    )
    thread.start()
    for delta in streamer:
        yield delta
    thread.join()


def _run_pass(messages, audios, sampling_rate) -> Iterator[tuple[str, str]]:
    """Generator: yields (sentence, tone) for each REPLY sentence the moment
    it's ready. Its return value (accessed via StopIteration.value when
    driven manually) is (log, tone, search_query, full_reply_text)."""
    full_text = ""
    reply_start = None
    emitted_upto = 0
    log_text = tone_text = search_text = ""

    for delta in _generate_stream(messages, audios, sampling_rate):
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
    """Main entry point.

    Yields events as the reply is produced:
      {"type": "sentence", "text": ..., "tone": ...}  -- ready to speak now
      {"type": "done"}                                 -- reply finished

    Handles conversation memory and an optional one-round web search
    internally, so the caller (run_agent.py) never has to know about either.
    """
    messages = _build_messages(audio_array)
    gen = _run_pass(messages, audio_array, sampling_rate)

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
        for delta in _generate_stream(followup, None, None):
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
