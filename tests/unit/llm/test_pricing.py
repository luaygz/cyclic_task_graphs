"""Price-table costs and the JSONL spend ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ctg.config import ModelSpec, Provider, Role
from ctg.llm.base import LLMCallRecord
from ctg.llm.pricing import SpendLedger, token_cost_usd

MODEL = ModelSpec(name="m", input_usd_per_mtok=2.0, cached_input_usd_per_mtok=0.5, output_usd_per_mtok=8.0)


def record(cost: float, provider: Provider = Provider.OPENAI) -> LLMCallRecord:
    return LLMCallRecord(
        role=Role.ROUTER, model="m", provider=provider, cost_usd=cost, started_at=datetime.now(UTC)
    )


def test_cached_input_tokens_are_billed_at_the_cached_rate() -> None:
    cost = token_cost_usd(MODEL, input_tokens=1_000_000, cached_input_tokens=400_000, output_tokens=100_000)

    assert cost == pytest.approx(0.6 * 2.0 + 0.4 * 0.5 + 0.1 * 8.0)


def test_cached_tokens_fall_back_to_the_input_rate_without_a_cached_price() -> None:
    model = MODEL.model_copy(update={"cached_input_usd_per_mtok": None})

    assert token_cost_usd(model, 1_000_000, 400_000, 0) == pytest.approx(2.0)


def test_unpriced_model_cannot_be_costed() -> None:
    with pytest.raises(ValueError, match="no prices"):
        token_cost_usd(ModelSpec(name="unpriced"), 10, 0, 10)


def test_totals_filter_by_provider_and_run(tmp_path: Path) -> None:
    ledger = SpendLedger(tmp_path / "nested" / "spend.jsonl")
    ledger.append(record(1.0), "a")
    ledger.append(record(2.0, Provider.OPENROUTER), "a")
    ledger.append(record(4.0), "b")
    ledger.append(record(8.0), None)

    assert ledger.total() == pytest.approx(15.0)
    assert ledger.total(Provider.OPENAI) == pytest.approx(13.0)
    assert ledger.total(run="a") == pytest.approx(3.0)
    assert ledger.total(Provider.OPENROUTER, run="b") == 0.0


def test_ledger_sees_spend_appended_by_another_writer(tmp_path: Path) -> None:
    path = tmp_path / "spend.jsonl"
    reader, writer = SpendLedger(path), SpendLedger(path)

    writer.append(record(0.75), "run")

    assert reader.total(run="run") == pytest.approx(0.75)
    assert SpendLedger(path).total() == pytest.approx(0.75)


def test_incomplete_trailing_line_is_ignored_until_finished(tmp_path: Path) -> None:
    path = tmp_path / "spend.jsonl"
    ledger = SpendLedger(path)
    ledger.append(record(1.0), None)
    line = path.read_text().splitlines()[0].replace('"cost_usd":1.0', '"cost_usd":2.0')
    half = len(line) // 2

    with path.open("a") as f:
        f.write(line[:half])
    assert ledger.total() == pytest.approx(1.0)

    with path.open("a") as f:
        f.write(line[half:] + "\n")
    assert ledger.total() == pytest.approx(3.0)


def test_corrupt_ledger_line_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "spend.jsonl"
    path.write_text('{"cost_usd": "lots"}\n')

    with pytest.raises(ValueError, match="corrupt line"):
        SpendLedger(path)
