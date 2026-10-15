"""Errors shared by the LLM client, the environments and the runner."""

from __future__ import annotations


class BackendUnavailableError(RuntimeError):
    """An external service cannot serve this run: an LLM provider or a tool backend has no credits or
    quota left, or rejects the API key or the model.

    Not a tool error: the agent never sees it. The runner records the episode as an error and starts no
    further episodes, since every one of them would fail the same way.
    """
