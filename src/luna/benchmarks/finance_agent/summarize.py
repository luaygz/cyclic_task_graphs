"""Token-bounded recursive summarization ported from the pending private fix."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

MAX_SUMMARIZER_INPUT_TOKENS = 24_000


def chunk_text_by_tokens(text: str, max_tokens: int = MAX_SUMMARIZER_INPUT_TOKENS) -> list[str]:
    if not text:
        return [text]
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model("gpt-4o")
        tokens = encoding.encode(text)
        return [encoding.decode(tokens[index : index + max_tokens]) for index in range(0, len(tokens), max_tokens)]
    except ImportError:
        approximate_characters = max_tokens * 4
        return [text[index : index + approximate_characters] for index in range(0, len(text), approximate_characters)]


async def summarize_content(
    content: str,
    prompt: str,
    summarize_once: Callable[[str, str, str], Awaitable[str]],
) -> str:
    chunks = chunk_text_by_tokens(content)
    if len(chunks) == 1:
        return await summarize_once(chunks[0], prompt, "document")
    summaries = [
        await summarize_once(chunk, prompt, f"chunk {index} of {len(chunks)}")
        for index, chunk in enumerate(chunks, start=1)
    ]
    combined = "\n\n".join(summaries)
    for level in range(1, 5):
        next_chunks = chunk_text_by_tokens(combined)
        if len(next_chunks) == 1:
            return await summarize_once(combined, prompt, "combined chunk summaries")
        combined = "\n\n".join(
            [
                await summarize_once(chunk, prompt, f"recursive level {level}, chunk {index}")
                for index, chunk in enumerate(next_chunks, start=1)
            ]
        )
    raise RuntimeError("recursive summaries did not converge below the 24,000-token bound")

