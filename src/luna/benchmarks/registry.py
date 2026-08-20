"""Lazy benchmark registry: one adapter never imports another's dependencies."""

from __future__ import annotations

import importlib
from typing import Any

_REGISTRY = {
    "alfworld": "luna.benchmarks.alfworld.adapter:ALFWorldAdapter",
    "textcraft": "luna.benchmarks.textcraft.adapter:TextCraftAdapter",
    "finance_agent": "luna.benchmarks.finance_agent.adapter:FinanceAgentAdapter",
}


def benchmark_names() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def get_adapter_class(name: str) -> type[Any]:
    try:
        target = _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark {name!r}; choose one of {', '.join(_REGISTRY)}") from exc
    module_name, class_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

