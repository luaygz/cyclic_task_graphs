"""Prompt templates live next to this module as Markdown files.

Templates use `$name` placeholders (string.Template), so JSON and braces in prompts need no escaping.
"""

from __future__ import annotations

from functools import cache
from importlib.resources import files
from string import Template


@cache
def load(name: str) -> Template:
    """Load `<name>.md` from this package, e.g. load("router") or load("benchmarks/textcraft")."""
    return Template(files("ctg.prompts").joinpath(f"{name}.md").read_text(encoding="utf-8"))


def render(name: str, **values: str) -> str:
    """Fill a template. Missing placeholders raise KeyError."""
    return load(name).substitute(values)
