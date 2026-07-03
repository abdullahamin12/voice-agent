"""
Free web search for the voice agent.
Uses DuckDuckGo via the `ddgs` package (the actively-maintained successor to
the old `duckduckgo_search` package, confirmed current on PyPI, v9.x as of
this writing) -- no API key, no billing.
    pip install ddgs
Runs in venv-brain (imported by native_brain.py).
"""

from ddgs import DDGS


def search_web(query: str, max_results: int = 4, timeout: int = 8) -> str:
    """Return a short, LLM-ready digest of the top results.

    Never raises -- on any failure it returns an explanatory string instead,
    so a flaky network connection degrades the reply gracefully rather than
    crashing the whole turn.
    """
    try:
        with DDGS(timeout=timeout) as ddgs:
            results = list(
                ddgs.text(query, max_results=max_results, safesearch="moderate")
            )
    except Exception as e:  # noqa: BLE001
        return f"(search failed: {e})"

    if not results:
        return "(no results found)"

    lines = []
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        body = (r.get("body") or "").strip()
        # Keep it short -- this text goes straight into a small model's
        # context window, and a 4B model does not need five paragraphs per
        # source to answer a factual question.
        if len(body) > 240:
            body = body[:240].rsplit(" ", 1)[0] + "..."
        lines.append(f"{i}. {title}: {body}")
    return "\n".join(lines)
