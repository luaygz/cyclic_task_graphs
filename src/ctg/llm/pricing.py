"""Call pricing and the persistent spend ledger.

Every model call (including failed attempts) is appended to one JSONL ledger, so cumulative spend
survives across runs. Spend caps are checked against the ledger before each call.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from ctg.config import REPO_ROOT, ModelSpec, Provider, Role
from ctg.llm.base import LLMCallRecord

DEFAULT_LEDGER_PATH = REPO_ROOT / ".ctg" / "spend.jsonl"


def has_prices(model: ModelSpec) -> bool:
    return model.input_usd_per_mtok is not None and model.output_usd_per_mtok is not None


def token_cost_usd(
    model: ModelSpec, input_tokens: int, cached_input_tokens: int, output_tokens: int
) -> float:
    """Cost from the preset's per-million-token prices.

    `input_tokens` includes the cached ones; those are billed at the cached rate (the input rate when the
    preset has no cached price). Reasoning tokens are part of `output_tokens`.
    """
    input_rate, output_rate = model.input_usd_per_mtok, model.output_usd_per_mtok
    if input_rate is None or output_rate is None:
        raise ValueError(f"model {model.name!r} has no prices; add them to configs/models.yaml")
    cached_rate = (
        model.cached_input_usd_per_mtok if model.cached_input_usd_per_mtok is not None else input_rate
    )
    uncached = input_tokens - cached_input_tokens
    return (
        uncached * input_rate + cached_input_tokens * cached_rate + output_tokens * output_rate
    ) / 1_000_000


class SpendEntry(BaseModel):
    """One ledger line."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    provider: Provider
    model: str
    role: Role
    run: str | None
    cost_usd: float
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int

    @classmethod
    def from_record(cls, record: LLMCallRecord, run: str | None) -> SpendEntry:
        return cls(
            timestamp=record.started_at,
            provider=record.provider,
            model=record.model,
            role=record.role,
            run=run,
            cost_usd=record.cost_usd,
            input_tokens=record.input_tokens,
            cached_input_tokens=record.cached_input_tokens,
            output_tokens=record.output_tokens,
            reasoning_tokens=record.reasoning_tokens,
        )


class SpendLedger:
    """Append-only JSONL spend log with running totals per (provider, run).

    Totals are refreshed from the file before they are read, so lines appended by other processes
    sharing the ledger count towards the caps too. Each line is written with a single append, and only
    complete lines are read, so concurrent writers never produce or observe half a record.
    """

    def __init__(self, path: Path = DEFAULT_LEDGER_PATH) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._offset = 0
        self._totals: defaultdict[tuple[Provider, str | None], float] = defaultdict(float)
        with self._lock:
            self._catch_up()

    def append(self, record: LLMCallRecord, run: str | None) -> None:
        line = SpendEntry.from_record(record, run).model_dump_json() + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)
            self._catch_up()

    def total(self, provider: Provider | None = None, run: str | None = None) -> float:
        """Recorded spend in USD, optionally restricted to one provider and/or one run."""
        with self._lock:
            self._catch_up()
            return sum(
                cost
                for (entry_provider, entry_run), cost in self._totals.items()
                if (provider is None or entry_provider == provider) and (run is None or entry_run == run)
            )

    def _catch_up(self) -> None:
        try:
            with self.path.open("rb") as f:
                f.seek(self._offset)
                data = f.read()
        except FileNotFoundError:
            return
        complete = data[: data.rfind(b"\n") + 1]
        for line in complete.splitlines():
            if not line.strip():
                continue
            try:
                entry = SpendEntry.model_validate_json(line)
            except ValidationError as exc:
                raise ValueError(f"corrupt line in spend ledger {self.path}: {line!r}") from exc
            self._totals[(entry.provider, entry.run)] += entry.cost_usd
        self._offset += len(complete)
